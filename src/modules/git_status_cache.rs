//! Persistent cache for `git_status` results.
//!
//! Computing the worktree status of a repository whose index stat cache is
//! stale requires re-statting and re-hashing every tracked file, which costs
//! hundreds of milliseconds on big repositories. Git normally heals this by
//! persisting the refreshed stat information back into the index, but starship
//! deliberately must not do that: `GIT_OPTIONAL_LOCKS=0` is set on every git
//! invocation precisely so that a prompt renderer never creates
//! `.git/index.lock` or rewrites the index (either could block or corrupt a
//! concurrent user git command), and gitoxide never writes the index either.
//! The result is that on such repositories every prompt render re-pays the
//! full scan cost, forever.
//!
//! This module memoizes the `git_status` counts in starship's own cache
//! directory instead, keyed by a fingerprint that proves the result cannot
//! have changed: per-file stat data plus index blob ids, the untracked file
//! set, `.gitignore`/`.gitattributes` mtimes, the index mtime, `HEAD`, and the
//! computation-affecting configuration. Any change at all is a cache miss and
//! falls back to the full computation, so a stale entry can only ever cost
//! time, never correctness. The repository itself is only ever *read*: the
//! index is never written and no lock files are created.
//!
//! A cheap probe (comparing the stat of a small sample of index entries
//! against the index's recorded stat) detects whether the index stat cache is
//! warm; when it is, the persistent cache is bypassed entirely so that
//! repositories git already handles quickly see no behavior change.

use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Version of the on-disk format. Bump when `CacheFile` changes incompatibly.
const CACHE_VERSION: u32 = 1;
/// Repositories with more files than this are not cached.
const MAX_CACHED_FILES: usize = 100_000;
/// Number of index entries probed to decide whether the index stat cache is warm.
const STALENESS_SAMPLE_SIZE: usize = 32;

/// Counts computed by `get_repo_status`, the expensive part worth memoizing.
/// (`ahead`/`behind` are re-validated cheaply on every hit instead of trusted.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub(crate) struct StatusCounts {
    pub conflicted: usize,
    pub deleted: usize,
    pub renamed: usize,
    pub modified: usize,
    pub staged: usize,
    pub typechanged: usize,
    pub untracked: usize,
    pub worktree_added: usize,
    pub worktree_deleted: usize,
    pub worktree_modified: usize,
    pub worktree_typechanged: usize,
    pub index_added: usize,
    pub index_deleted: usize,
    pub index_modified: usize,
    pub index_typechanged: usize,
}

/// A cached `git_status` result with its upstream tracking info.
#[derive(Debug, Clone)]
pub(crate) struct CachedStatus {
    pub counts: StatusCounts,
    pub ahead: Option<usize>,
    pub behind: Option<usize>,
}

/// Stat data for a single worktree file, plus the index blob it was compared
/// against (empty for untracked files).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct FileEntry {
    /// Workdir-relative path with `/` separators.
    path: String,
    dev: u64,
    ino: u64,
    /// Full mode bits from the filesystem (unix only; synthesized elsewhere).
    mode: u32,
    size: u64,
    mtime_secs: i64,
    mtime_nanos: u32,
    ctime_secs: i64,
    ctime_nanos: u32,
    /// Hex object id of the index blob; empty for untracked files.
    blob: String,
    tracked: bool,
}

/// A metadata file whose content influences status computation
/// (`.gitignore`, `.gitattributes`, `info/exclude`, `core.excludesFile`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct MetaFile {
    /// Workdir-relative path (with `/` separators) or absolute path.
    path: String,
    mtime_secs: i64,
    mtime_nanos: u32,
    size: u64,
}

/// Everything a cached result depends on, minus the result itself.
/// Two fingerprints with equal keys and equal file lists observed a repository
/// in an indistinguishable state, so the status computation would produce the
/// same counts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CacheKey {
    version: u32,
    /// Key of the starship-side computation-affecting config (see
    /// `status_config_key` in `git_status.rs`).
    config_key: String,
    /// `HEAD` object id in hex, or empty when unborn.
    head: String,
    index_mtime_secs: i64,
    index_mtime_nanos: u32,
    index_size: u64,
    meta_files: Vec<MetaFile>,
    info_exclude: Option<MetaFile>,
    excludes_file: Option<MetaFile>,
    show_untracked: String,
    core_filemode: String,
    core_ignorecase: String,
    diff_renames: String,
    diff_renamelimit: String,
}

/// The working fingerprint of the current repository state.
#[derive(Debug, Clone)]
pub(crate) struct Fingerprint {
    key: CacheKey,
    files: Vec<FileEntry>,
}

/// What is stored on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheFile {
    key: CacheKey,
    /// Seconds/nanos since the epoch when the entry was recorded. A file's
    /// fingerprint is only trusted when its mtime is strictly older than this
    /// (the racy-clean guard, mirroring git).
    recorded_at_secs: i64,
    recorded_at_nanos: u32,
    counts: StatusCounts,
    ahead: Option<usize>,
    behind: Option<usize>,
    files: Vec<FileEntry>,
}

/// Outcome of consulting the persistent cache.
#[derive(Debug)]
pub(crate) enum LookupOutcome {
    /// Usable cached result. The fingerprint is included so the caller can
    /// re-store it after recomputing (e.g. when upstream tracking changed).
    Hit {
        status: CachedStatus,
        fingerprint: Fingerprint,
    },
    /// No usable entry; the fingerprint is included so the caller can store
    /// the fresh computation without walking the worktree twice.
    Miss { fingerprint: Fingerprint },
    /// Do not consult the cache and do not store: the index stat cache looks
    /// warm (today's behavior is already fast), caching is disabled, or the
    /// repository is not eligible.
    Skip,
}

/// Consult the persistent cache for this repository.
///
/// Returns [`LookupOutcome::Skip`] when the index stat cache looks warm (the
/// normal case, where the cache would only add overhead), when caching is
/// disabled via `STARSHIP_GIT_STATUS_CACHE_OFF`, or when the repository is
/// not eligible (submodules, unreadable index, too many files, ...).
/// Otherwise walks the worktree once and compares the fingerprint against the
/// stored entry.
pub(crate) fn lookup(
    git_dir: &Path,
    workdir: &Path,
    gix_repo: &gix::Repository,
    config_key: &str,
) -> LookupOutcome {
    if std::env::var_os("STARSHIP_GIT_STATUS_CACHE_OFF").is_some() {
        log::debug!("git_status cache: disabled via STARSHIP_GIT_STATUS_CACHE_OFF");
        return LookupOutcome::Skip;
    }
    let index = match gix_repo.index_or_empty() {
        Ok(index) => index,
        Err(e) => {
            log::debug!("git_status cache: cannot read index: {e}");
            return LookupOutcome::Skip;
        }
    };
    // Submodule dirtiness is not visible in the parent worktree stat data, so
    // a stat fingerprint cannot prove a submodule is unchanged. Never cache
    // repositories with submodules.
    if index.entries().iter().any(|e| e.mode.is_submodule()) {
        log::debug!("git_status cache: repository has submodules, skipping");
        return LookupOutcome::Skip;
    }
    match stat_cache_warm(&index, workdir) {
        None => {
            log::debug!("git_status cache: cannot probe index stat cache");
            return LookupOutcome::Skip;
        }
        Some(true) => {
            // Warm: git/gitoxide already answer quickly; bypass the cache so
            // these repositories see no behavior change at all.
            return LookupOutcome::Skip;
        }
        Some(false) => {
        }
    }
    let fingerprint = match Fingerprint::compute(git_dir, workdir, gix_repo, &index, config_key) {
        Some(fingerprint) => {
            fingerprint
        }
        None => {
            return LookupOutcome::Skip;
        }
    };
    let stored: CacheFile = match load(git_dir) {
        Some(stored) => {
            stored
        }
        None => {
            log::debug!("git_status cache: miss (no entry)");
            return LookupOutcome::Miss { fingerprint };
        }
    };
    if stored.key != fingerprint.key {
        log::debug!("git_status cache: miss (key changed)");
        return LookupOutcome::Miss { fingerprint };
    }
    if stored.files != fingerprint.files {
        log::debug!("git_status cache: miss (worktree changed)");
        return LookupOutcome::Miss { fingerprint };
    }
    // Racy-clean guard, mirroring git: only trust a file's fingerprint when
    // its mtime is strictly older than the moment the entry was recorded. A
    // file touched at/after recording may have changed within the same
    // timestamp tick, which stat data alone cannot reveal.
    let racy = fingerprint.files.iter().any(|f| {
        is_racy(
            (f.mtime_secs, f.mtime_nanos),
            (stored.recorded_at_secs, stored.recorded_at_nanos),
        )
    });
    if racy {
        log::debug!("git_status cache: miss (racy timestamps)");
        return LookupOutcome::Miss { fingerprint };
    }
    log::debug!("git_status cache: hit");
    LookupOutcome::Hit {
        status: CachedStatus {
            counts: stored.counts,
            ahead: stored.ahead,
            behind: stored.behind,
        },
        fingerprint,
    }
}

/// Whether a file mtime is too close to the entry's record time to trust:
/// anything not strictly older may have changed within the same timestamp
/// tick, which stat data alone cannot reveal.
fn is_racy(mtime: (i64, u32), recorded_at: (i64, u32)) -> bool {
    mtime >= recorded_at
}

/// Store the result of a fresh computation under `fingerprint`.
/// Only ever writes to starship's own cache directory, never to the repository.
pub(crate) fn store(git_dir: &Path, fingerprint: &Fingerprint, status: &CachedStatus) {
    let path = match cache_file_path(git_dir) {
        Some(path) => path,
        None => return,
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let entry = CacheFile {
        key: fingerprint.key.clone(),
        recorded_at_secs: now.as_secs() as i64,
        recorded_at_nanos: now.subsec_nanos(),
        counts: status.counts,
        ahead: status.ahead,
        behind: status.behind,
        files: fingerprint.files.clone(),
    };
    let bytes = match serde_json::to_vec(&entry) {
        Ok(bytes) => bytes,
        Err(e) => {
            log::debug!("git_status cache: cannot serialize entry: {e}");
            return;
        }
    };
    if let Some(parent) = path.parent()
        && let Err(e) = fs::create_dir_all(parent)
    {
        log::debug!("git_status cache: cannot create cache dir: {e}");
        return;
    }
    // Atomic write: readers only ever see complete entries.
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    if let Err(e) = fs::write(&tmp, &bytes) {
        log::debug!("git_status cache: cannot write entry: {e}");
        return;
    }
    if let Err(e) = fs::rename(&tmp, &path) {
        log::debug!("git_status cache: cannot publish entry: {e}");
        let _ = fs::remove_file(&tmp);
    }
}

/// Probe whether the index stat cache looks warm: compare a sample of index
/// entries' recorded stat against the live filesystem. Returns `None` when
/// the probe cannot run. Either answer is safe: the probe only decides whether
/// consulting the persistent cache is worthwhile, never the result itself.
fn stat_cache_warm(index: &gix::index::File, workdir: &Path) -> Option<bool> {
    let entries = index.entries();
    if entries.is_empty() {
        return Some(true);
    }
    let step = (entries.len() / STALENESS_SAMPLE_SIZE).max(1);
    for entry in entries.iter().step_by(step).take(STALENESS_SAMPLE_SIZE) {
        let rel = std::str::from_utf8(entry.path(index)).ok()?;
        let md = fs::symlink_metadata(workdir.join(rel)).ok()?;
        if !stat_matches_index(&md, &entry.stat) {
            return Some(false);
        }
    }
    Some(true)
}

/// Whether live filesystem stat matches what the index recorded.
fn stat_matches_index(md: &fs::Metadata, stat: &gix::index::entry::Stat) -> bool {
    if md.len() > u32::MAX as u64 || stat.size != md.len() as u32 {
        return false;
    }
    let Some((mtime_s, mtime_ns, ctime_s, ctime_ns)) = file_times(md) else {
        return false;
    };
    if stat.mtime.secs as i64 != mtime_s || stat.mtime.nsecs != mtime_ns {
        return false;
    }
    if stat.ctime.secs as i64 != ctime_s || stat.ctime.nsecs != ctime_ns {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // gix truncates dev/ino to u32 in the index; compare the same way.
        if stat.dev != md.dev() as u32 || stat.ino != md.ino() as u32 {
            return false;
        }
    }
    true
}

/// (mtime_secs, mtime_nanos, ctime_secs, ctime_nanos) for a file.
fn file_times(md: &fs::Metadata) -> Option<(i64, u32, i64, u32)> {
    let mtime = md.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Some((
            mtime.as_secs() as i64,
            mtime.subsec_nanos(),
            md.ctime(),
            md.ctime_nsec() as u32,
        ))
    }
    #[cfg(not(unix))]
    {
        let ctime = md.created().ok()?.duration_since(UNIX_EPOCH).ok()?;
        Some((
            mtime.as_secs() as i64,
            mtime.subsec_nanos(),
            ctime.as_secs() as i64,
            ctime.subsec_nanos(),
        ))
    }
}

#[cfg(unix)]
fn dev_ino_mode(md: &fs::Metadata) -> (u64, u64, u32) {
    use std::os::unix::fs::MetadataExt;
    (md.dev(), md.ino(), md.mode())
}

#[cfg(not(unix))]
fn dev_ino_mode(md: &fs::Metadata) -> (u64, u64, u32) {
    // std does not expose dev/ino on non-unix; synthesize mode bits from the
    // file type. Precision loss here only ever causes extra cache misses.
    let ft = md.file_type();
    let mode = if ft.is_symlink() {
        0o120000
    } else if ft.is_dir() {
        0o040000
    } else {
        0o100644
    };
    (0, 0, mode)
}

impl Fingerprint {
    /// Compute the fingerprint of everything the status computation depends
    /// on. Returns `None` when the repository is not eligible for caching.
    /// Only reads the repository; never writes.
    fn compute(
        git_dir: &Path,
        workdir: &Path,
        gix_repo: &gix::Repository,
        index: &gix::index::File,
        config_key: &str,
    ) -> Option<Self> {
        let git_config = gix_repo.config_snapshot();

        let head = gix_repo
            .head_id()
            .ok()
            .map(|id| id.to_hex().to_string())
            .unwrap_or_default();

        let index_md = fs::metadata(index.path()).ok()?;
        let index_mtime = index_md
            .modified()
            .ok()?
            .duration_since(UNIX_EPOCH)
            .ok()?;

        // Index entries: path -> blob id, for change detection on stage/unstage.
        let mut index_paths: HashSet<String> = HashSet::new();
        let mut files: Vec<FileEntry> = Vec::new();
        for entry in index.entries() {
            let rel = std::str::from_utf8(entry.path(index)).ok()?;
            let rel = rel.replace('\\', "/");
            index_paths.insert(rel.clone());
            // Absent files (deletions) are represented by omission: the entry
            // simply won't be in `files`, which differs from a cached entry.
            let md = match fs::symlink_metadata(workdir.join(&rel)) {
                Ok(md) => md,
                Err(_) => continue,
            };
            let Some((mtime_s, mtime_ns, ctime_s, ctime_ns)) = file_times(&md) else {
                return None;
            };
            let (dev, ino, mode) = dev_ino_mode(&md);
            files.push(FileEntry {
                path: rel,
                dev,
                ino,
                mode,
                size: md.len(),
                mtime_secs: mtime_s,
                mtime_nanos: mtime_ns,
                ctime_secs: ctime_s,
                ctime_nanos: ctime_ns,
                blob: entry.id.to_hex().to_string(),
                tracked: true,
            });
        }

        // Worktree walk for untracked files (and to notice deletions/renames).
        // The walk is deliberately not ignore-aware: any new file at all is a
        // fingerprint change (a conservative miss), which keeps this sound
        // without reimplementing ignore logic.
        let mut meta_files: Vec<MetaFile> = Vec::new();
        let mut stack = vec![workdir.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let read_dir = match fs::read_dir(&dir) {
                Ok(read_dir) => read_dir,
                Err(_) => continue,
            };
            for dir_entry in read_dir {
                let dir_entry = match dir_entry {
                    Ok(dir_entry) => dir_entry,
                    Err(_) => continue,
                };
                let path = dir_entry.path();
                let file_type = match dir_entry.file_type() {
                    Ok(file_type) => file_type,
                    Err(_) => continue,
                };
                // Never descend into or record the git dir itself; its files
                // (locks, logs) change constantly without affecting status.
                if dir == workdir && path.file_name().is_some_and(|name| name == ".git") {
                    continue;
                }
                if file_type.is_dir() && !file_type.is_symlink() {
                    stack.push(path);
                    continue;
                }
                let rel = match path.strip_prefix(workdir).ok()?.to_str() {
                    Some(rel) => rel.replace('\\', "/"),
                    None => return None,
                };
                if index_paths.contains(&rel) {
                    continue;
                }
                let md = match fs::symlink_metadata(&path) {
                    Ok(md) => md,
                    Err(_) => continue,
                };
                let Some((mtime_s, mtime_ns, ctime_s, ctime_ns)) = file_times(&md) else {
                    return None;
                };
                if rel == ".gitignore"
                    || rel.ends_with("/.gitignore")
                    || rel == ".gitattributes"
                    || rel.ends_with("/.gitattributes")
                {
                    meta_files.push(MetaFile {
                        path: rel.clone(),
                        mtime_secs: mtime_s,
                        mtime_nanos: mtime_ns,
                        size: md.len(),
                    });
                }
                let (dev, ino, mode) = dev_ino_mode(&md);
                files.push(FileEntry {
                    path: rel,
                    dev,
                    ino,
                    mode,
                    size: md.len(),
                    mtime_secs: mtime_s,
                    mtime_nanos: mtime_ns,
                    ctime_secs: ctime_s,
                    ctime_nanos: ctime_ns,
                    blob: String::new(),
                    tracked: false,
                });
                if files.len() > MAX_CACHED_FILES {
                    log::debug!("git_status cache: too many files, skipping");
                    return None;
                }
            }
        }
        files.sort_by(|a, b| a.path.cmp(&b.path));
        meta_files.sort_by(|a, b| a.path.cmp(&b.path));

        let info_exclude = meta_file_at(&git_dir.join("info").join("exclude"));
        let excludes_file = git_config
            .string("core.excludesFile")
            .map(|value| value.to_string())
            .and_then(|value| {
            let path = PathBuf::from(expand_tilde(&value));
            meta_file_at(&path).map(|mut meta| {
                // Record the configured value too: pointing at a different
                // file must invalidate even if mtimes collide.
                meta.path = format!("core.excludesFile={value}");
                meta
            })
        });

        Some(Fingerprint {
            key: CacheKey {
                version: CACHE_VERSION,
                config_key: config_key.to_owned(),
                head,
                index_mtime_secs: index_mtime.as_secs() as i64,
                index_mtime_nanos: index_mtime.subsec_nanos(),
                index_size: index_md.len(),
                meta_files,
                info_exclude,
                excludes_file,
                show_untracked: config_string(&git_config, "status.showUntrackedFiles"),
                core_filemode: config_string(&git_config, "core.fileMode"),
                core_ignorecase: config_string(&git_config, "core.ignoreCase"),
                diff_renames: config_string(&git_config, "diff.renames"),
                diff_renamelimit: config_string(&git_config, "diff.renameLimit"),
            },
            files,
        })
    }
}

/// Load the stored entry for a repository, if any. Corrupt entries are treated
/// as absent (and replaced on the next store).
fn load(git_dir: &Path) -> Option<CacheFile> {
    let path = cache_file_path(git_dir)?;
    let bytes = fs::read(&path).ok()?;
    let entry: CacheFile = serde_json::from_slice(&bytes).ok()?;
    if entry.key.version != CACHE_VERSION {
        return None;
    }
    Some(entry)
}

/// Read a string value from the git config snapshot, lossily at worst.
/// Only used for cache-key purposes, where an approximation is safe: the
/// value is only compared for equality against itself, so an approximation
/// can only cause extra cache misses, never wrong hits.
fn config_string(config: &gix::config::Snapshot<'_>, key: &str) -> String {
    config
        .string(key)
        .map(|value| value.to_string())
        .unwrap_or_default()
}

fn meta_file_at(path: &Path) -> Option<MetaFile> {
    let md = fs::symlink_metadata(path).ok()?;
    if !md.is_file() {
        return None;
    }
    let (mtime_s, mtime_ns, _, _) = file_times(&md)?;
    Some(MetaFile {
        path: path.to_string_lossy().replace('\\', "/"),
        mtime_secs: mtime_s,
        mtime_nanos: mtime_ns,
        size: md.len(),
    })
}

fn expand_tilde(value: &str) -> String {
    if let Some(rest) = value.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest).to_string_lossy().into_owned();
    }
    value.to_owned()
}

fn cache_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|dir| {
        dir.join("starship")
            .join("git-status-cache")
            .join(format!("v{CACHE_VERSION}"))
    })
}

/// Cache file for a repository, named by a hash of its canonical git dir.
/// Hash collisions only ever cause extra cache misses, never wrong results.
fn cache_file_path(git_dir: &Path) -> Option<PathBuf> {
    let dir = cache_dir()?;
    let canonical = git_dir
        .canonicalize()
        .unwrap_or_else(|_| git_dir.to_path_buf());
    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    Some(dir.join(format!("{:016x}.json", hasher.finish())))
}

/// Exposed for tests.
#[cfg(test)]
pub(crate) fn cache_file_path_for_test(git_dir: &Path) -> Option<PathBuf> {
    cache_file_path(git_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::process::Command;

    fn git(dir: &Path, args: &[&str]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(dir)
            // Isolate from the developer's git config.
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn init_repo() -> io::Result<tempfile::TempDir> {
        let dir = tempfile::tempdir()?;
        git(dir.path(), &["init", "-b", "main"]);
        git(dir.path(), &["config", "user.email", "test@test"]);
        git(dir.path(), &["config", "user.name", "test"]);
        git(dir.path(), &["config", "commit.gpgsign", "false"]);
        fs::create_dir_all(dir.path().join("src"))?;
        fs::write(dir.path().join("src").join("a.txt"), "hello\n")?;
        fs::write(dir.path().join("src").join("b.txt"), "world\n")?;
        git(dir.path(), &["add", "-A"]);
        git(dir.path(), &["commit", "-qm", "init"]);
        Ok(dir)
    }

    fn open_repo(dir: &tempfile::TempDir) -> gix::Repository {
        gix::open(dir.path()).expect("open repo")
    }

    fn git_dir_of(repo: &gix::Repository) -> PathBuf {
        repo.git_dir().to_path_buf()
    }

    fn workdir_of(repo: &gix::Repository) -> PathBuf {
        repo.workdir().expect("workdir").to_path_buf()
    }

    /// Make the index stat cache stale without changing any content:
    /// backdate every tracked file deterministically (POSIX `touch -t`).
    /// A fixed old timestamp also keeps the racy-clean guard deterministic.
    #[cfg(unix)]
    fn stale_stat_cache(dir: &tempfile::TempDir) {
        let output = Command::new("git")
            .args(["ls-files", "-z"])
            .current_dir(dir.path())
            .output()
            .expect("git ls-files");
        assert!(output.status.success());
        for path in output.stdout.split(|b| *b == 0).filter(|p| !p.is_empty()) {
            let path = dir.path().join(std::str::from_utf8(path).unwrap());
            let touched = Command::new("touch")
                .args(["-t", "200101010000"])
                .arg(&path)
                .output()
                .expect("touch");
            assert!(touched.status.success());
        }
    }

    fn lookup_for_test(repo: &gix::Repository) -> LookupOutcome {
        lookup(&git_dir_of(repo), &workdir_of(repo), repo, "test-config-key")
    }

    fn dummy_status() -> CachedStatus {
        CachedStatus {
            counts: StatusCounts {
                modified: 2,
                ..Default::default()
            },
            ahead: Some(1),
            behind: Some(0),
        }
    }

    #[test]
    fn is_racy_predicate() {
        // (mtime) >= (recorded_at) is racy; strictly older is trusted.
        assert!(!is_racy((100, 0), (101, 0)));
        assert!(!is_racy((100, 999_999_999), (101, 0)));
        assert!(is_racy((101, 0), (101, 0)));
        assert!(is_racy((101, 1), (101, 0)));
        assert!(is_racy((102, 0), (101, 0)));
    }

    #[test]
    #[cfg(unix)]
    fn miss_then_hit_roundtrip() -> io::Result<()> {
        let dir = init_repo()?;
        let repo = open_repo(&dir);
        stale_stat_cache(&dir);

        // No entry yet: miss, with a fingerprint to store under.
        let fingerprint = match lookup_for_test(&repo) {
            LookupOutcome::Miss { fingerprint } => fingerprint,
            other => panic!("expected miss, got {other:?}"),
        };

        // Store and look up again: hit with identical counts.
        let expected = dummy_status();
        store(&git_dir_of(&repo), &fingerprint, &expected);
        match lookup_for_test(&repo) {
            LookupOutcome::Hit { status, .. } => {
                assert_eq!(status.counts, expected.counts);
                assert_eq!(status.ahead, expected.ahead);
                assert_eq!(status.behind, expected.behind);
            }
            other => panic!("expected hit, got {other:?}"),
        }

        // A cache file was actually written, outside the repository.
        let cache_path = cache_file_path_for_test(&git_dir_of(&repo)).expect("cache path");
        assert!(cache_path.is_file(), "cache file missing: {cache_path:?}");
        assert!(
            !cache_path.starts_with(dir.path()),
            "cache must not live inside the repository"
        );
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn miss_on_content_change() -> io::Result<()> {
        let dir = init_repo()?;
        let repo = open_repo(&dir);
        stale_stat_cache(&dir);

        let fingerprint = match lookup_for_test(&repo) {
            LookupOutcome::Miss { fingerprint } => fingerprint,
            other => panic!("expected miss, got {other:?}"),
        };
        store(&git_dir_of(&repo), &fingerprint, &dummy_status());

        // Sanity: hit before the change.
        assert!(matches!(lookup_for_test(&repo), LookupOutcome::Hit { .. }));

        // Modify a tracked file: must miss.
        fs::write(dir.path().join("src").join("a.txt"), "changed\n")?;
        assert!(matches!(lookup_for_test(&repo), LookupOutcome::Miss { .. }));
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn miss_on_untracked_file() -> io::Result<()> {
        let dir = init_repo()?;
        let repo = open_repo(&dir);
        stale_stat_cache(&dir);

        let fingerprint = match lookup_for_test(&repo) {
            LookupOutcome::Miss { fingerprint } => fingerprint,
            other => panic!("expected miss, got {other:?}"),
        };
        store(&git_dir_of(&repo), &fingerprint, &dummy_status());
        assert!(matches!(lookup_for_test(&repo), LookupOutcome::Hit { .. }));

        fs::write(dir.path().join("new-untracked.txt"), "new\n")?;
        assert!(matches!(lookup_for_test(&repo), LookupOutcome::Miss { .. }));
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn miss_on_stage() -> io::Result<()> {
        let dir = init_repo()?;
        let repo = open_repo(&dir);
        stale_stat_cache(&dir);

        let fingerprint = match lookup_for_test(&repo) {
            LookupOutcome::Miss { fingerprint } => fingerprint,
            other => panic!("expected miss, got {other:?}"),
        };
        store(&git_dir_of(&repo), &fingerprint, &dummy_status());
        assert!(matches!(lookup_for_test(&repo), LookupOutcome::Hit { .. }));

        // Staging rewrites the index: must miss (index mtime + blob change).
        fs::write(dir.path().join("src").join("a.txt"), "staged change\n")?;
        git(dir.path(), &["add", "src/a.txt"]);
        assert!(matches!(lookup_for_test(&repo), LookupOutcome::Miss { .. }));
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn miss_on_head_change() -> io::Result<()> {
        let dir = init_repo()?;
        let repo = open_repo(&dir);
        stale_stat_cache(&dir);

        let fingerprint = match lookup_for_test(&repo) {
            LookupOutcome::Miss { fingerprint } => fingerprint,
            other => panic!("expected miss, got {other:?}"),
        };
        store(&git_dir_of(&repo), &fingerprint, &dummy_status());
        assert!(matches!(lookup_for_test(&repo), LookupOutcome::Hit { .. }));

        // A new commit moves HEAD: must miss. (Committing warms the index
        // stat cache, so re-stale it to exercise the fingerprint path.)
        fs::write(dir.path().join("src").join("a.txt"), "v2\n")?;
        git(dir.path(), &["commit", "-qm", "v2", "src/a.txt"]);
        stale_stat_cache(&dir);
        assert!(matches!(lookup_for_test(&repo), LookupOutcome::Miss { .. }));
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn warm_repo_skips_cache() -> io::Result<()> {
        let dir = init_repo()?;
        let repo = open_repo(&dir);
        // Fresh repo: index stat cache is warm, so the persistent cache is
        // bypassed entirely (today's behavior, no overhead).
        assert!(matches!(lookup_for_test(&repo), LookupOutcome::Skip));
        // And nothing was written.
        let cache_path = cache_file_path_for_test(&git_dir_of(&repo)).expect("cache path");
        assert!(!cache_path.exists());
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn never_writes_index_or_lock() -> io::Result<()> {
        let dir = init_repo()?;
        let repo = open_repo(&dir);
        stale_stat_cache(&dir);
        let index_path = git_dir_of(&repo).join("index");
        let mtime_before = fs::metadata(&index_path)?.modified()?;

        let fingerprint = match lookup_for_test(&repo) {
            LookupOutcome::Miss { fingerprint } => fingerprint,
            other => panic!("expected miss, got {other:?}"),
        };
        store(&git_dir_of(&repo), &fingerprint, &dummy_status());
        let _ = lookup_for_test(&repo);

        // Lock safety: the index must be untouched and no lock file created.
        let mtime_after = fs::metadata(&index_path)?.modified()?;
        assert_eq!(
            mtime_before, mtime_after,
            "persistent cache must never rewrite the index"
        );
        assert!(
            !git_dir_of(&repo).join("index.lock").exists(),
            "persistent cache must never create index.lock"
        );
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn corrupt_entry_is_treated_as_miss() -> io::Result<()> {
        let dir = init_repo()?;
        let repo = open_repo(&dir);
        stale_stat_cache(&dir);

        // Poison the cache file.
        if let Some(path) = cache_file_path_for_test(&git_dir_of(&repo)) {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, b"not valid json{{{")?;
        }
        assert!(matches!(lookup_for_test(&repo), LookupOutcome::Miss { .. }));
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn submodule_disables_cache() -> io::Result<()> {
        let sub = init_repo()?;
        let dir = init_repo()?;
        // `protocol.file.allow=always` is needed for local submodule clones.
        git(
            dir.path(),
            &[
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                sub.path().to_str().unwrap(),
                "vendor/sub",
            ],
        );
        git(dir.path(), &["commit", "-qm", "add submodule"]);
        let repo = open_repo(&dir);
        stale_stat_cache(&dir);
        assert!(matches!(lookup_for_test(&repo), LookupOutcome::Skip));
        Ok(())
    }
}
