//! Runs on disk.
//!
//! A [`Store`] is a directory with one subdirectory for each run, named by the
//! run's [`RunId`]. A run's directory holds its `journal`: every change to the
//! run, in the format of [`crate::journal`].
//!
//! # What a commit guarantees
//!
//! [`Store::commit`] and [`Store::create_run`] return `Ok` only after the
//! record is written and `fsync` has returned. So a change that was
//! acknowledged is in the journal when the server is killed and started again.
//! The events of one commit are one record: after a restart the run has all of
//! them or none.
//!
//! If the write fails, the store takes the record out again and returns the
//! error. If that fails too, it returns [`StoreError::OutcomeUnknown`]: the
//! commit may or may not be in the run.
//!
//! # Refused submissions
//!
//! A submission past the run's cap on unchecked submissions for its batch
//! (see [`crate::state::RunConfig::max_unchecked_per_batch`]) is not kept in
//! the journal: [`Store::commit_or_refuse`] counts it instead, in the file
//! `refused` beside the journal. The file holds a count for each iteration
//! and kind, and the journal's id, so it does not grow with the refusals.
//! It is written whole to `refused.new`, synced, and renamed over
//! `refused`, under the run's lock, so a crash leaves the old count or the
//! new one. A count is on disk before the refusal is answered. A `refused`
//! file with another journal's id, or one that does not read, counts as none
//! (the second is reported on standard error, and covers a file that cannot
//! be read at all), and the next refusal writes
//! over it. The journal does not depend on the file.
//!
//! # More than one server
//!
//! Several stores, in one process or in several, can use the same run. Each
//! operation takes an exclusive lock on the run's directory (`flock`), reads
//! the records the others added since it last looked, and only then acts. A
//! store keeps each run's state in memory, so reading a run's state parses
//! each record once. It reads a journal one record at a time and applies each
//! record before it reads the next. So a store that reads a run for the first
//! time holds about one record in memory, not the journal. The exception is a
//! torn tail: the scan holds each of its lines whole, and a damaged tail can be
//! one line of any length. The lock needs a local file system.
//!
//! An operation waits for the lock for a limited time, [`DEFAULT_LOCK_WAIT`]
//! unless the store was opened with another, and then fails with
//! [`StoreError::Busy`]. If the run's directory was removed while it waited,
//! it fails with [`StoreError::Replaced`] or [`StoreError::NoSuchRun`].
//!
//! # Torn tails
//!
//! A write that did not finish leaves a torn tail after the last good record
//! (see [`crate::journal`]). Reading ignores it. The next commit that has
//! events to write copies it to a
//! file `torn-<offset>-<n>` beside the journal and writes its record in the
//! tail's place. It says so in a log record of kind
//! [`LogKind::Recovery`], which it puts first
//! in that commit, and on standard error. Damage to the end of a journal looks
//! the same, so an operator who sees such a record after a clean stop should
//! treat it as lost commits.
//!
//! If that commit fails, the store puts the tail back, and the next commit
//! starts again. It uses the file it already made. Of the run's records, the
//! store deletes one thing only: a `torn-*` file that it could not finish
//! writing. (It also removes claim files, which hold nothing: see "Claims on
//! checks".)
//!
//! A `torn-*` file can be there with no log record. Treat the file as the same
//! sign. The commit that made the file did not finish:
//!
//! * the commit failed, or the server stopped, with the tail still in the
//!   journal or put back there. The next commit that writes reports it;
//! * the server stopped after it cut the tail off and before its record was
//!   on disk;
//! * the commit failed and the store could not put the whole tail back;
//! * the commit failed with [`StoreError::OutcomeUnknown`].
//!
//! In the last three cases no later commit reports the tail.
//!
//! # Claims on checks
//!
//! Several servers check a run's pending submissions (see
//! [`crate::proposer::check_pending`]). So that each submission is checked by
//! one of them, a checker first claims the submission's batch with
//! [`Store::try_claim_check`]: an exclusive, non-blocking `flock` on the file
//! `checks/<batch>.lock` in the run's directory. A checker that cannot take a
//! claim leaves the batch to its holder. The claim is held while the programs
//! run, without the run's lock.
//!
//! The operating system releases a claim when its holder's file is closed,
//! also when the holder is killed, so a claim cannot outlive its process. A
//! holder that is done removes the file before it releases the lock
//! ([`CheckClaim::release`]), if the file at the path is still the one it
//! locked: after the run was removed and made again, the path can name a
//! file another checker holds. A claim is only taken on the file that is at
//! the path once the lock is held, so a checker that opened the removed file
//! takes no claim on it. A file left by a holder that stopped is taken by the
//! next checker of its batch, or removed by [`Store::remove_settled_claims`]
//! once the batch is checked. The files hold nothing, and the journal does
//! not depend on them.
//!
//! Claims need a local file system, as the run's lock does. On NFS, Linux
//! emulates `flock` with POSIX locks, which belong to a process, so two
//! checkers in one process would both get a claim.
//!
//! A claim does not tie a check to the run: a checker that reads a pending
//! submission with [`Store::read_journal`] commits its checks with
//! [`Store::commit_to`], which writes nothing if the run was removed and made
//! again in the meantime.
//!
//! # A journal that is replaced
//!
//! Each journal has a random id in its header. Before a store uses what it
//! has in memory, it checks that id and the last record it read. If the id is
//! another, the run was removed and made again, and the store reads the new
//! journal from the start. If the id is the same and the journal is shorter or
//! its last record has changed, the operation fails with
//! [`StoreError::Corrupt`], and the next one reads the journal from the start.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::hash::{BuildHasher, Hasher, RandomState};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::ops::ControlFlow;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::journal::{self, Corrupt, HeaderError, Record, ScanError};
pub use crate::state::{Event, LogRecord, RunConfig, RunId, RunState};
use crate::state::{LogKind, Rejected};

const JOURNAL: &str = "journal";
const JOURNAL_NEW: &str = "journal.new";
/// The directory of the claims on checks, in a run's directory.
const CHECKS: &str = "checks";
/// How many times a claim is tried when its file is removed meanwhile.
const CLAIM_TRIES: usize = 8;
const REFUSED: &str = "refused";
const REFUSED_NEW: &str = "refused.new";

/// How many submissions were refused in one iteration because their batch
/// had as many unchecked submissions of their kind as the run keeps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefusedCounts {
    /// Late or repeat submissions from the batch's worker. A late one is
    /// the first from its worker, and a batch keeps at least one of this
    /// kind, so a late one is never refused.
    pub repeats: u64,
    /// Foreign submissions.
    pub foreign: u64,
}

/// The submissions of a run that were refused and counted rather than kept
/// (see the module docs), as its `refused` file holds them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Refusals {
    /// The id of the journal the counts are for, in hex.
    journal: String,
    /// The counts of each iteration that has any.
    iterations: BTreeMap<u64, RefusedCounts>,
}

impl Refusals {
    /// The counts of `iteration`.
    pub fn iteration(&self, iteration: u64) -> RefusedCounts {
        self.iterations.get(&iteration).copied().unwrap_or_default()
    }

    /// The counts of every iteration, added up.
    pub fn total(&self) -> RefusedCounts {
        self.iterations
            .values()
            .fold(RefusedCounts::default(), |a, c| RefusedCounts {
                repeats: a.repeats.saturating_add(c.repeats),
                foreign: a.foreign.saturating_add(c.foreign),
            })
    }
}

/// A run's `refused` file, for [`Store::commit_then_with_refusals`] to read
/// while the run is locked.
#[derive(Debug)]
pub struct RefusalsFile {
    dir: PathBuf,
    journal: JournalId,
}

impl RefusalsFile {
    /// The refusals the file holds now: none if it cannot be read (see
    /// the module docs).
    pub fn read(&self) -> Refusals {
        read_refusals(&self.dir, self.journal)
    }
}

/// What a [`Store::commit_or_refuse`] does to a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Change {
    /// Commit these events, as [`Store::commit`] does.
    Events(Vec<Event>),
    /// Count a submission of this kind as refused in this iteration,
    /// and write nothing to the journal.
    Refuse {
        /// The run's iteration when it came in.
        iteration: u64,
        /// Its kind.
        kind: RefusalKind,
    },
}

/// The kind of a refused submission, for its count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefusalKind {
    /// A late or repeat submission from the batch's worker.
    Repeat,
    /// A foreign submission.
    Foreign,
}

/// How much of a journal a scan reads from the file at a time.
const READ_BUFFER: usize = 1 << 16;

/// How long an operation waits for a run's lock by default.
pub const DEFAULT_LOCK_WAIT: Duration = Duration::from_secs(30);

/// Why a store operation failed.
#[derive(Debug)]
pub enum StoreError {
    /// The store has no run of this name.
    NoSuchRun(RunId),
    /// The store already has a run of this name.
    RunExists(RunId),
    /// The events cannot be applied to the run. Nothing was written.
    Rejected(Rejected),
    /// Another store held the run's lock for the whole wait. Nothing was
    /// written.
    Busy(RunId),
    /// The run's directory was removed, or removed and made again, while the
    /// operation waited for its lock. No run was changed.
    Replaced(RunId),
    /// The run's journal does not read back. The store does not change it.
    Corrupt {
        /// The run.
        run: RunId,
        /// What is wrong.
        error: Corrupt,
    },
    /// The run's journal is in a format this version does not read.
    UnsupportedFormat {
        /// The run.
        run: RunId,
        /// The journal's header line.
        header: String,
    },
    /// A file operation failed. If it was part of a commit, the commit is not
    /// in the run.
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The error.
        error: io::Error,
    },
    /// A write failed at a point where the store cannot tell what is on disk:
    /// a commit's write failed and so did taking the record out again, or a
    /// new run's directory could not be synced. The change may or may not be
    /// in the store: read the run to find out.
    OutcomeUnknown {
        /// The journal.
        path: PathBuf,
        /// The error of the write.
        error: io::Error,
    },
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::NoSuchRun(run) => write!(f, "there is no run {:?}", run.as_str()),
            StoreError::RunExists(run) => write!(f, "there is already a run {:?}", run.as_str()),
            StoreError::Rejected(why) => write!(f, "{why}"),
            StoreError::Busy(run) => write!(
                f,
                "run {:?} is locked by another server or call. Nothing was changed",
                run.as_str()
            ),
            StoreError::Replaced(run) => write!(
                f,
                "the directory of run {:?} was removed or replaced while the call waited for \
                 it. No run was changed. Try again",
                run.as_str()
            ),
            StoreError::Corrupt { run, error } => {
                write!(
                    f,
                    "the journal of run {:?} is damaged: {error}",
                    run.as_str()
                )
            }
            StoreError::UnsupportedFormat { run, header } => write!(
                f,
                "the journal of run {:?} has the format {header:?}, which this version does not \
                 read",
                run.as_str()
            ),
            StoreError::Io { path, error } => write!(f, "{}: {error}", path.display()),
            StoreError::OutcomeUnknown { path, error } => write!(
                f,
                "{}: {error}. The change may or may not be in the run",
                path.display()
            ),
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StoreError::Rejected(why) => Some(why),
            StoreError::Corrupt { error, .. } => Some(error),
            StoreError::Io { error, .. } | StoreError::OutcomeUnknown { error, .. } => Some(error),
            StoreError::NoSuchRun(_)
            | StoreError::RunExists(_)
            | StoreError::Busy(_)
            | StoreError::Replaced(_)
            | StoreError::UnsupportedFormat { .. } => None,
        }
    }
}

/// Makes an [`io::Error`] on `path` a [`StoreError`].
fn io_at(path: &Path) -> impl FnOnce(io::Error) -> StoreError + '_ {
    |error| StoreError::Io {
        path: path.to_owned(),
        error,
    }
}

/// The file operations of an append, so that a test can make them fail.
pub(crate) trait Durable {
    /// Sets the file's length.
    fn truncate(&mut self, len: u64) -> io::Result<()>;
    /// Writes all of `bytes` at offset `at`.
    fn write_at(&mut self, at: u64, bytes: &[u8]) -> io::Result<()>;
    /// Waits until the file's content is on disk.
    fn sync(&mut self) -> io::Result<()>;
}

impl Durable for File {
    fn truncate(&mut self, len: u64) -> io::Result<()> {
        self.set_len(len)
    }

    fn write_at(&mut self, at: u64, bytes: &[u8]) -> io::Result<()> {
        self.seek(SeekFrom::Start(at))?;
        self.write_all(bytes)
    }

    fn sync(&mut self) -> io::Result<()> {
        self.sync_data()
    }
}

/// Why an append failed.
#[derive(Debug)]
pub(crate) enum AppendError {
    /// The record is not in the file.
    NotWritten(io::Error),
    /// The record may be in the file.
    Unknown(io::Error),
}

/// Writes `record` at byte `at` of the journal, the end of its last good
/// record, and waits until it is on disk. Anything after `at` goes first.
///
/// If a step fails, the file is cut back to `at`, so that a record the caller
/// is told failed is not read as committed later.
pub(crate) fn append<F: Durable>(file: &mut F, at: u64, record: &[u8]) -> Result<(), AppendError> {
    let written = file
        .truncate(at)
        .and_then(|()| file.write_at(at, record))
        .and_then(|()| file.sync());
    let Err(error) = written else {
        return Ok(());
    };
    match file.truncate(at).and_then(|()| file.sync()) {
        Ok(()) => Err(AppendError::NotWritten(error)),
        Err(_) => Err(AppendError::Unknown(error)),
    }
}

/// Writes `tail` back at byte `at` of the journal, its end, after an append
/// that failed cut it off. If it cannot write all of it, it takes the part
/// out again: a part would be set aside later as a tail of its own.
pub(crate) fn put_back<F: Durable>(file: &mut F, at: u64, tail: &[u8]) {
    if file.write_at(at, tail).and_then(|()| file.sync()).is_err() {
        let _ = file.truncate(at).and_then(|()| file.sync());
    }
}

/// What a store remembers of a run between operations.
#[derive(Debug)]
struct Cached {
    state: RunState,
    /// The id in the journal's header.
    id: u64,
    /// The length of the journal's good records.
    len: u64,
    /// The number of the journal's next record.
    next_seq: u64,
    /// Where the last good record starts, and the CRC-32 of its line.
    last: (u64, u32),
}

/// A run's files, open and locked. Dropping it releases the lock.
#[derive(Debug)]
struct Locked {
    journal: File,
    path: PathBuf,
    dir_path: PathBuf,
    /// The run's directory. The lock is on it.
    dir: File,
}

/// A claim on checking one batch's submission. See the module docs.
/// Dropping it releases the claim and leaves its file, for the next checker.
#[derive(Debug)]
pub struct CheckClaim {
    /// The open claim file. The lock is on it.
    file: File,
    path: PathBuf,
    batch: u64,
}

impl CheckClaim {
    /// The batch it claims.
    pub fn batch(&self) -> u64 {
        self.batch
    }

    /// Removes the claim's file, then releases the claim. Call it when the
    /// batch needs no more checking. A file that cannot be removed is left
    /// for [`Store::remove_settled_claims`].
    pub fn release(self) {
        // Removed while the lock is held: see `Store::try_claim_check`. Only
        // the file this claim locked is removed. If the run's directory was
        // removed and made again, the path can name a file of the new run
        // that another checker holds.
        if still_at(&self.file, &self.path).ok() == Some(Some(true)) {
            let _ = fs::remove_file(&self.path);
        }
        drop(self.file);
    }
}

/// One journal of a run. A run that is removed and made again has a journal
/// with another id. See [`Store::read_journal`] and [`Store::commit_to`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JournalId(u64);

/// A directory of runs. See the module docs.
#[derive(Debug)]
pub struct Store {
    root: PathBuf,
    lock_wait: Duration,
    /// A run's entry is taken out while an operation holds the run's lock,
    /// and put back if the operation leaves it true to the journal.
    cache: Mutex<HashMap<RunId, Cached>>,
    /// The steps of the next append that fail, for tests of the error paths.
    #[cfg(test)]
    faults: Mutex<Vec<usize>>,
}

impl Store {
    /// The store in directory `root`, which is made if it does not exist.
    pub fn open(root: &Path) -> Result<Store, StoreError> {
        Store::open_with_lock_wait(root, DEFAULT_LOCK_WAIT)
    }

    /// As [`Store::open`], with operations that wait `lock_wait` for a run's
    /// lock.
    pub fn open_with_lock_wait(root: &Path, lock_wait: Duration) -> Result<Store, StoreError> {
        fs::create_dir_all(root).map_err(io_at(root))?;
        Ok(Store {
            root: root.to_owned(),
            lock_wait,
            cache: Mutex::new(HashMap::new()),
            #[cfg(test)]
            faults: Mutex::new(Vec::new()),
        })
    }

    /// The store's directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Makes a run at iteration 0 and returns its state.
    pub fn create_run(
        &self,
        run: &RunId,
        config: RunConfig,
        tasks: Vec<String>,
    ) -> Result<RunState, StoreError> {
        let created = Event::Created { config, tasks };
        let state = RunState::replay([&created]).map_err(StoreError::Rejected)?;
        let dir_path = self.root.join(run.as_str());
        match fs::create_dir(&dir_path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(io_at(&dir_path)(e)),
        }
        // Also when the directory was there: an earlier attempt may have
        // stopped before this.
        sync_dir(&self.root)?;
        let dir = File::open(&dir_path).map_err(io_at(&dir_path))?;
        self.wait_for_lock(run, &dir, &dir_path)?;
        if still_at(&dir, &dir_path).map_err(io_at(&dir_path))? != Some(true) {
            return Err(StoreError::Replaced(run.clone()));
        }

        let path = dir_path.join(JOURNAL);
        if path.try_exists().map_err(io_at(&path))? {
            return Err(StoreError::RunExists(run.clone()));
        }
        // The journal appears under its name only when it is complete, so a
        // run that exists always has its first record.
        let new_path = dir_path.join(JOURNAL_NEW);
        let mut file = File::create(&new_path).map_err(io_at(&new_path))?;
        let id = RandomState::new().build_hasher().finish();
        let header = journal::header(id);
        let first = journal::encode(0, &[created]);
        file.write_all(header.as_bytes())
            .and_then(|()| file.write_all(first.as_bytes()))
            .and_then(|()| file.sync_all())
            .map_err(io_at(&new_path))?;
        fs::rename(&new_path, &path).map_err(io_at(&path))?;
        // From here the run exists, whatever the sync says.
        dir.sync_all().map_err(|error| StoreError::OutcomeUnknown {
            path: dir_path.clone(),
            error,
        })?;
        self.put(
            run,
            Cached {
                state: state.clone(),
                id,
                len: (header.len() + first.len()) as u64,
                next_seq: 1,
                last: (header.len() as u64, journal::crc32(first.as_bytes())),
            },
        );
        Ok(state)
    }

    /// The file of the claim on checking `batch` of `run`.
    fn claim_path(&self, run: &RunId, batch: u64) -> PathBuf {
        self.root
            .join(run.as_str())
            .join(CHECKS)
            .join(format!("{batch}.lock"))
    }

    /// Claims the check of `batch`'s submission in `run` for the caller,
    /// without waiting. `None` if another checker holds the claim. See the
    /// module docs.
    pub fn try_claim_check(
        &self,
        run: &RunId,
        batch: u64,
    ) -> Result<Option<CheckClaim>, StoreError> {
        let path = self.claim_path(run, batch);
        let checks = path.parent().unwrap_or(&self.root);
        // Not `create_dir_all`: a run's directory that is gone is not made.
        match fs::create_dir(checks) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(StoreError::NoSuchRun(run.clone()))
            }
            Err(e) => return Err(io_at(checks)(e)),
        }
        for _ in 0..CLAIM_TRIES {
            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)
                .map_err(io_at(&path))?;
            match file.try_lock() {
                Ok(()) => {}
                Err(TryLockError::WouldBlock) => return Ok(None),
                Err(TryLockError::Error(e)) => return Err(io_at(&path)(e)),
            }
            // The holder before may have removed the file after this call
            // opened it. Then the lock is on a file no other checker sees.
            if still_at(&file, &path).map_err(io_at(&path))? == Some(true) {
                return Ok(Some(CheckClaim { file, path, batch }));
            }
        }
        // Each try found its file removed by a holder that was done.
        Ok(None)
    }

    /// The batches of `run` that have a claim file, held or not, in order.
    pub fn check_claims(&self, run: &RunId) -> Result<Vec<u64>, StoreError> {
        let dir = self.root.join(run.as_str()).join(CHECKS);
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io_at(&dir)(e)),
        };
        let mut batches = Vec::new();
        for entry in entries {
            let entry = entry.map_err(io_at(&dir))?;
            let name = entry.file_name();
            let batch = name
                .to_str()
                .and_then(|n| n.strip_suffix(".lock"))
                .and_then(|n| n.parse::<u64>().ok());
            batches.extend(batch);
        }
        batches.sort_unstable();
        Ok(batches)
    }

    /// Removes the claim files of `batches` of `run` that no checker holds.
    /// Call it only for batches listed by [`Store::check_claims`] whose
    /// submission was checked when the run was read after that. A file is
    /// only made for a batch whose submission waits to be checked, and a
    /// batch's submission is checked once, so no checker wants these claims.
    pub fn remove_settled_claims(&self, run: &RunId, batches: &[u64]) -> Result<(), StoreError> {
        for &batch in batches {
            if let Some(claim) = self.try_claim_check(run, batch)? {
                claim.release();
            }
        }
        Ok(())
    }

    /// The ids of the store's runs, in order.
    pub fn runs(&self) -> Result<Vec<RunId>, StoreError> {
        let mut runs = Vec::new();
        for entry in fs::read_dir(&self.root).map_err(io_at(&self.root))? {
            let entry = entry.map_err(io_at(&self.root))?;
            let Some(run) = entry.file_name().to_str().and_then(|n| RunId::new(n).ok()) else {
                continue;
            };
            let path = entry.path().join(JOURNAL);
            match fs::metadata(&path) {
                Ok(meta) if meta.is_file() => runs.push(run),
                Ok(_) => {}
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
                    ) => {}
                Err(e) => return Err(io_at(&path)(e)),
            }
        }
        runs.sort();
        Ok(runs)
    }

    /// What `read` makes of the run's state now.
    pub fn read<T, F>(&self, run: &RunId, read: F) -> Result<T, StoreError>
    where
        F: FnOnce(&RunState) -> T,
    {
        self.read_journal(run, read).map(|(result, _)| result)
    }

    /// As [`Store::read`], and the journal that was read. Pass it to
    /// [`Store::commit_to`] to write only to that journal.
    pub fn read_journal<T, F>(&self, run: &RunId, read: F) -> Result<(T, JournalId), StoreError>
    where
        F: FnOnce(&RunState) -> T,
    {
        let mut locked = self.lock_run(run)?;
        let (cached, _) = self.catch_up(run, &mut locked)?;
        let result = read(&cached.state);
        let journal = JournalId(cached.id);
        self.put(run, cached);
        Ok((result, journal))
    }

    /// A copy of the run's state now.
    pub fn state(&self, run: &RunId) -> Result<RunState, StoreError> {
        self.read(run, RunState::clone)
    }

    /// Changes the run by the events `change` returns, all of them or none,
    /// and returns the value `change` returns with them.
    ///
    /// `change` sees the run's state now. No other store changes the run
    /// between that and the write, so `change` can decide by what it sees. If
    /// it returns an error, or no events, nothing is written. `Ok` means the
    /// events are on disk: see the module docs.
    ///
    /// `change` must not use this run through any store: the run is locked,
    /// so that call would wait and then fail with [`StoreError::Busy`].
    pub fn commit<T, E, F>(&self, run: &RunId, change: F) -> Result<T, E>
    where
        F: FnOnce(&RunState) -> Result<(T, Vec<Event>), E>,
        E: From<StoreError>,
    {
        self.commit_then(run, change, |value, _| value)
    }

    /// As [`Store::commit`], and then returns what `then` makes of the value
    /// and the run's state after the events, while the run is still locked.
    pub fn commit_then<T, U, E, F, G>(&self, run: &RunId, change: F, then: G) -> Result<U, E>
    where
        F: FnOnce(&RunState) -> Result<(T, Vec<Event>), E>,
        G: FnOnce(T, &RunState) -> U,
        E: From<StoreError>,
    {
        self.change_then(
            run,
            |state, _| change(state).map(|(value, events)| (value, Change::Events(events))),
            |value, state, _| then(value, state),
        )
    }

    /// As [`Store::commit`], if the run's journal is still `journal`.
    /// Otherwise the run was removed and made again since `journal` was read:
    /// `change` is not called, nothing is written, and the result is `None`.
    pub fn commit_to<T, E, F>(
        &self,
        run: &RunId,
        journal: JournalId,
        change: F,
    ) -> Result<Option<T>, E>
    where
        F: FnOnce(&RunState) -> Result<(T, Vec<Event>), E>,
        E: From<StoreError>,
    {
        self.change_then(
            run,
            |state, now| {
                if now == journal {
                    change(state).map(|(value, events)| (Some(value), Change::Events(events)))
                } else {
                    Ok((None, Change::Events(Vec::new())))
                }
            },
            |value, _, _| value,
        )
    }

    /// As [`Store::commit_then`], where `then` can also read the run's
    /// refusals (see [`Store::commit_or_refuse`]) while the run is locked.
    pub fn commit_then_with_refusals<T, U, E, F, G>(
        &self,
        run: &RunId,
        change: F,
        then: G,
    ) -> Result<U, E>
    where
        F: FnOnce(&RunState) -> Result<(T, Vec<Event>), E>,
        G: FnOnce(T, &RunState, &RefusalsFile) -> U,
        E: From<StoreError>,
    {
        self.change_then(
            run,
            |state, _| change(state).map(|(value, events)| (value, Change::Events(events))),
            then,
        )
    }

    /// As [`Store::commit`], where `change` can also count a submission as
    /// refused rather than commit events. A refusal is in the run's
    /// `refused` file, not the journal, so a proposer that keeps sending
    /// does not grow the journal (see the module docs). `Ok` means the
    /// count is on disk.
    pub fn commit_or_refuse<T, E, F>(&self, run: &RunId, change: F) -> Result<T, E>
    where
        F: FnOnce(&RunState) -> Result<(T, Change), E>,
        E: From<StoreError>,
    {
        self.change_then(run, |state, _| change(state), |value, _, _| value)
    }

    /// The refusals counted for the run (see [`Store::commit_or_refuse`]).
    pub fn refusals(&self, run: &RunId) -> Result<Refusals, StoreError> {
        self.read_with_refusals(run, |_, refusals| refusals.clone())
    }

    /// What `read` makes of the run's state and its refusals now.
    pub fn read_with_refusals<T, F>(&self, run: &RunId, read: F) -> Result<T, StoreError>
    where
        F: FnOnce(&RunState, &Refusals) -> T,
    {
        let mut locked = self.lock_run(run)?;
        let (cached, _) = self.catch_up(run, &mut locked)?;
        let refusals = read_refusals(&locked.dir_path, JournalId(cached.id));
        let result = read(&cached.state, &refusals);
        self.put(run, cached);
        Ok(result)
    }

    /// [`Store::commit_then`], [`Store::commit_to`] and
    /// [`Store::commit_or_refuse`], with `change` told which journal it sees.
    fn change_then<T, U, E, F, G>(&self, run: &RunId, change: F, then: G) -> Result<U, E>
    where
        F: FnOnce(&RunState, JournalId) -> Result<(T, Change), E>,
        G: FnOnce(T, &RunState, &RefusalsFile) -> U,
        E: From<StoreError>,
    {
        let mut locked = self.lock_run(run)?;
        let (mut cached, torn_len) = self.catch_up(run, &mut locked)?;
        let file = RefusalsFile {
            dir: locked.dir_path.clone(),
            journal: JournalId(cached.id),
        };
        let then = |value: T, state: &RunState| then(value, state, &file);
        let (value, mut events) = match change(&cached.state, JournalId(cached.id)) {
            Ok((value, Change::Events(events))) => (value, events),
            Ok((value, Change::Refuse { iteration, kind })) => {
                let counted = count_refusal(&locked, JournalId(cached.id), iteration, kind);
                let result = counted.map(|()| then(value, &cached.state));
                self.put(run, cached);
                return result.map_err(E::from);
            }
            Err(error) => {
                self.put(run, cached);
                return Err(error);
            }
        };
        if events.is_empty() {
            let result = then(value, &cached.state);
            self.put(run, cached);
            return Ok(result);
        }
        if torn_len == 0 {
            // The usual commit applies the events to the state in place, not
            // to a copy, so it does not cost a copy of the whole run.
            for (i, event) in events.iter().enumerate() {
                if let Err(why) = cached.state.apply(event) {
                    // A rejected event leaves the state as it was (see
                    // `RunState::apply`). So after the first event the state
                    // is unchanged and is kept; after a later one it holds
                    // events that are not in the journal, and it is dropped:
                    // the next operation reads the journal again.
                    if i == 0 {
                        self.put(run, cached);
                    }
                    return Err(StoreError::Rejected(why).into());
                }
            }
            let record = journal::encode(cached.next_seq, &events);
            match self.append_record(&mut locked.journal, cached.len, record.as_bytes()) {
                Ok(()) => {}
                // The state holds events that are not in the journal, so it
                // is dropped, and the next operation reads the journal again.
                Err(AppendError::NotWritten(error)) => {
                    return Err(io_at(&locked.path)(error).into());
                }
                Err(AppendError::Unknown(error)) => {
                    return Err(StoreError::OutcomeUnknown {
                        path: locked.path,
                        error,
                    }
                    .into());
                }
            }
            return Ok(self.written(run, cached, &record, value, then));
        }
        // A commit over a torn tail is rare: it applies the events to a copy,
        // so that a rejected event leaves no file set aside, and then again
        // after the tail's report.
        let mut state = cached.state.clone();
        for event in &events {
            if let Err(why) = state.apply(event) {
                self.put(run, cached);
                return Err(StoreError::Rejected(why).into());
            }
        }
        // Only a commit that writes needs the tail's bytes. A tail that an
        // unfinished write left is at most one record long; damage to the
        // end of the journal can leave a longer one.
        let torn = if torn_len == 0 {
            Vec::new()
        } else {
            match read_at(&mut locked.journal, cached.len, torn_len) {
                Ok(torn) => torn,
                Err(error) => {
                    self.put(run, cached);
                    return Err(io_at(&locked.path)(error).into());
                }
            }
        };
        if !torn.is_empty() {
            let report = set_aside(&locked, cached.len, &torn, cached.state.iteration())?;
            // The report goes first, in the iteration the run is in now. It
            // always applies, and nothing later in the commit depends on it.
            state = cached.state.clone();
            events.insert(0, report);
            for event in &events {
                state.apply(event).map_err(StoreError::Rejected)?;
            }
        }
        let record = journal::encode(cached.next_seq, &events);
        match self.append_record(&mut locked.journal, cached.len, record.as_bytes()) {
            Ok(()) => {}
            Err(AppendError::NotWritten(error)) => {
                // The append cut the tail off. It goes back, so that the
                // commit that does replace it reports it. If this fails, the
                // file set aside is the only sign of the tail.
                if !torn.is_empty() {
                    put_back(&mut locked.journal, cached.len, &torn);
                }
                self.put(run, cached);
                return Err(io_at(&locked.path)(error).into());
            }
            // The state is not put back: the next operation reads the file.
            Err(AppendError::Unknown(error)) => {
                return Err(StoreError::OutcomeUnknown {
                    path: locked.path,
                    error,
                }
                .into());
            }
        }
        cached.state = state;
        Ok(self.written(run, cached, &record, value, then))
    }

    /// Notes in `cached`, whose state already has the record's events, that
    /// `record` is the journal's last record, keeps it, and returns what
    /// `then` makes of `value` and the state.
    fn written<T, U, G>(
        &self,
        run: &RunId,
        mut cached: Cached,
        record: &str,
        value: T,
        then: G,
    ) -> U
    where
        G: FnOnce(T, &RunState) -> U,
    {
        cached.last = (cached.len, journal::crc32(record.as_bytes()));
        cached.len += record.len() as u64;
        cached.next_seq += 1;
        let result = then(value, &cached.state);
        self.put(run, cached);
        result
    }

    /// Appends a record to the journal. See [`append`].
    fn append_record(&self, file: &mut File, at: u64, record: &[u8]) -> Result<(), AppendError> {
        #[cfg(test)]
        {
            let fail = std::mem::take(&mut *self.faults.lock().unwrap());
            if !fail.is_empty() {
                return append(&mut tests::Faulty::new(file, fail), at, record);
            }
        }
        append(file, at, record)
    }

    /// Changes the run by `events`, all of them or none. See [`Store::commit`].
    pub fn append(&self, run: &RunId, events: Vec<Event>) -> Result<(), StoreError> {
        self.commit(run, |_| Ok(((), events)))
    }

    /// Every log record of the run, oldest first. This reads the whole
    /// journal each time, and returns all the log's text at once: see
    /// [`Store::for_each_log`] for a reader that does not.
    pub fn log(&self, run: &RunId) -> Result<Vec<LogRecord>, StoreError> {
        let mut log = Vec::new();
        self.for_each_log(run, |record| {
            log.push(record);
            ControlFlow::Continue(())
        })?;
        Ok(log)
    }

    /// Calls `each` on the run's log records, oldest first, until it returns
    /// [`ControlFlow::Break`]. See [`Store::for_each_event`].
    pub fn for_each_log<F>(&self, run: &RunId, mut each: F) -> Result<(), StoreError>
    where
        F: FnMut(LogRecord) -> ControlFlow<()>,
    {
        self.for_each_event(run, |event| match event {
            Event::Logged(log) => each(log),
            _ => ControlFlow::Continue(()),
        })
    }

    /// Calls `each` on the run's events, oldest first, until it returns
    /// [`ControlFlow::Break`]. The events include the payloads handed out and
    /// the submissions received, with their text; a
    /// [`PromptTable`](crate::prompt::PromptTable) reads each payload back
    /// from them. It reads the whole journal each time, one record at a time,
    /// and holds only the record it is on.
    ///
    /// `each` is called while the run is locked, so it must not use this run
    /// through any store: that call would wait and then fail with
    /// [`StoreError::Busy`].
    ///
    /// If the journal is damaged after some records, `each` sees those
    /// records and then the call returns [`StoreError::Corrupt`].
    pub fn for_each_event<F>(&self, run: &RunId, mut each: F) -> Result<(), StoreError>
    where
        F: FnMut(Event) -> ControlFlow<()>,
    {
        let mut locked = self.lock_run(run)?;
        let scanned = scan_journal(&mut locked, run, None, |record| {
            for event in record.events {
                if each(event).is_break() {
                    return Err(Halt::Break);
                }
            }
            Ok(())
        });
        match scanned {
            Ok(_) | Err(Halt::Break) => Ok(()),
            Err(Halt::Error(error)) => Err(error),
        }
    }

    /// Puts a run's state back for the next operation.
    fn put(&self, run: &RunId, cached: Cached) {
        self.cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(run.clone(), cached);
    }

    /// Takes the lock on `dir`, a run's directory, or gives up after the
    /// store's wait.
    fn wait_for_lock(&self, run: &RunId, dir: &File, dir_path: &Path) -> Result<(), StoreError> {
        // No deadline if the wait is too long to add to the time now.
        let deadline = Instant::now().checked_add(self.lock_wait);
        let mut pause = Duration::from_micros(200);
        loop {
            match dir.try_lock() {
                Ok(()) => return Ok(()),
                Err(TryLockError::Error(e)) => return Err(io_at(dir_path)(e)),
                Err(TryLockError::WouldBlock) => {}
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(StoreError::Busy(run.clone()));
            }
            std::thread::sleep(pause);
            pause = (pause * 2).min(Duration::from_millis(10));
        }
    }

    /// Opens and locks the run's files.
    fn lock_run(&self, run: &RunId) -> Result<Locked, StoreError> {
        let dir_path = self.root.join(run.as_str());
        let missing = |e: &io::Error| {
            matches!(
                e.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
            )
        };
        let dir = match File::open(&dir_path) {
            Ok(dir) => dir,
            Err(e) if missing(&e) => return Err(StoreError::NoSuchRun(run.clone())),
            Err(e) => return Err(io_at(&dir_path)(e)),
        };
        self.wait_for_lock(run, &dir, &dir_path)?;
        match still_at(&dir, &dir_path).map_err(io_at(&dir_path))? {
            Some(true) => {}
            Some(false) => return Err(StoreError::Replaced(run.clone())),
            None => return Err(StoreError::NoSuchRun(run.clone())),
        }
        let path = dir_path.join(JOURNAL);
        match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(journal) => Ok(Locked {
                journal,
                path,
                dir_path,
                dir,
            }),
            Err(e) if missing(&e) => Err(StoreError::NoSuchRun(run.clone())),
            Err(e) => Err(io_at(&path)(e)),
        }
    }

    /// The state of `run` as the journal on disk has it, and the length of
    /// the journal's torn tail, which starts where the good records end. On
    /// an error the store forgets the run, and the next operation reads its
    /// whole journal.
    fn catch_up(&self, run: &RunId, locked: &mut Locked) -> Result<(Cached, u64), StoreError> {
        let cached = self
            .cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(run);
        let cached = match cached {
            Some(cached) => still_true(locked, run, cached)?,
            None => None,
        };
        let resume = cached.as_ref().map(|c| Resume {
            id: c.id,
            len: c.len,
            next_seq: c.next_seq,
        });
        let (mut state, mut last) = match cached {
            Some(cached) => (Some(cached.state), Some(cached.last)),
            None => (None, None),
        };
        let rejected = |offset: u64, why: Rejected| {
            Halt::Error(StoreError::Corrupt {
                run: run.clone(),
                error: Corrupt {
                    offset,
                    reason: match why {
                        Rejected::NotCreated => "a first record that does not create the run",
                        _ => "a record that cannot be applied to the run",
                    },
                },
            })
        };
        // Each record is applied as it is read and then dropped. The state
        // keeps counts of the log's records, not their text.
        let scanned = scan_journal(locked, run, resume, |record: Record| {
            match &mut state {
                Some(state) => {
                    for event in &record.events {
                        state.apply(event).map_err(|w| rejected(record.offset, w))?;
                    }
                }
                None => {
                    let first =
                        RunState::replay(&record.events).map_err(|w| rejected(record.offset, w))?;
                    state = Some(first);
                }
            }
            Ok(())
        })
        .map_err(Halt::into_error)?;
        let Some(state) = state else {
            return Err(StoreError::Corrupt {
                run: run.clone(),
                error: Corrupt {
                    offset: scanned.base,
                    reason: "a journal with no records",
                },
            });
        };
        if let Some((at, len)) = scanned.end.last {
            let line = read_at(&mut locked.journal, at, len).map_err(io_at(&locked.path))?;
            last = Some((at, journal::crc32(&line)));
        }
        let Some(last) = last else {
            unreachable!("a run with a state has a record")
        };
        let cached = Cached {
            state,
            id: scanned.id,
            len: scanned.base + scanned.end.valid_len,
            next_seq: scanned.first_seq + scanned.end.records,
            last,
        };
        Ok((cached, scanned.end.torn_len))
    }
}

/// Why a scan of a journal stopped before its end.
enum Halt {
    /// The operation fails.
    Error(StoreError),
    /// The caller has read all it wants.
    Break,
}

impl Halt {
    /// The error, for a caller that never breaks.
    fn into_error(self) -> StoreError {
        match self {
            Halt::Error(error) => error,
            Halt::Break => unreachable!("only for_each_log breaks"),
        }
    }
}

/// Checks that the journal on disk is still the one `cached` was read from.
/// Returns `None` if it is another run's journal.
fn still_true(
    locked: &mut Locked,
    run: &RunId,
    cached: Cached,
) -> Result<Option<Cached>, StoreError> {
    let corrupt = |offset, reason| StoreError::Corrupt {
        run: run.clone(),
        error: Corrupt { offset, reason },
    };
    let file = &mut locked.journal;
    let len = file.metadata().map_err(io_at(&locked.path))?.len();
    let mut start = vec![0; journal::MAX_HEADER_LEN.min(len as usize)];
    file.seek(SeekFrom::Start(0))
        .and_then(|_| file.read_exact(&mut start))
        .map_err(io_at(&locked.path))?;
    if journal::parse_header(&start).map(|(id, _)| id) != Ok(cached.id) {
        return Ok(None);
    }
    if len < cached.len {
        return Err(corrupt(len, "a journal that is shorter than it was"));
    }
    let (at, crc) = cached.last;
    let line = read_at(file, at, cached.len - at).map_err(io_at(&locked.path))?;
    if journal::crc32(&line) != crc {
        return Err(corrupt(at, "a record that changed after it was read"));
    }
    Ok(Some(cached))
}

/// Reads `len` bytes of `file` from byte `at`.
fn read_at(file: &mut File, at: u64, len: u64) -> io::Result<Vec<u8>> {
    let len = usize::try_from(len).map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
    let mut bytes = vec![0; len];
    file.seek(SeekFrom::Start(at))?;
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// Where a scan of a journal starts when the store has read part of it.
#[derive(Clone, Copy, Debug)]
struct Resume {
    /// The id in the journal's header.
    id: u64,
    /// The length of the good records the store has read.
    len: u64,
    /// The number of the next record.
    next_seq: u64,
}

/// What [`scan_journal`] read.
#[derive(Debug)]
struct Scanned {
    /// The id in the journal's header.
    id: u64,
    /// Where the scan started in the file.
    base: u64,
    /// The number of the first record the scan read.
    first_seq: u64,
    end: journal::End,
}

/// Reads the journal's records, those after `resume` or all of them if there
/// is no `resume`, and hands each one to `each` as it is read.
fn scan_journal<F>(
    locked: &mut Locked,
    run: &RunId,
    resume: Option<Resume>,
    each: F,
) -> Result<Scanned, Halt>
where
    F: FnMut(Record) -> Result<(), Halt>,
{
    let io_error = |error| Halt::Error(io_at(&locked.path)(error));
    let file = &mut locked.journal;
    let (id, base, first_seq) = match resume {
        Some(resume) => (resume.id, resume.len, resume.next_seq),
        None => {
            let mut start = Vec::with_capacity(journal::MAX_HEADER_LEN);
            file.seek(SeekFrom::Start(0))
                .and_then(|_| {
                    Read::by_ref(file)
                        .take(journal::MAX_HEADER_LEN as u64)
                        .read_to_end(&mut start)
                })
                .map_err(io_error)?;
            match journal::parse_header(&start) {
                Ok((id, len)) => (id, len as u64, 0),
                Err(HeaderError::Unsupported(header)) => {
                    return Err(Halt::Error(StoreError::UnsupportedFormat {
                        run: run.clone(),
                        header,
                    }))
                }
                Err(HeaderError::NotAJournal) => {
                    return Err(Halt::Error(StoreError::Corrupt {
                        run: run.clone(),
                        error: Corrupt {
                            offset: 0,
                            reason: "a file that does not start with a journal header",
                        },
                    }))
                }
            }
        }
    };
    file.seek(SeekFrom::Start(base)).map_err(io_error)?;
    let reader = BufReader::with_capacity(READ_BUFFER, file);
    let end = journal::scan_records(reader, base, first_seq, each).map_err(|e| match e {
        ScanError::Corrupt(error) => Halt::Error(StoreError::Corrupt {
            run: run.clone(),
            error,
        }),
        ScanError::Read(error) => io_error(error),
        ScanError::Stopped(halt) => halt,
    })?;
    Ok(Scanned {
        id,
        base,
        first_seq,
        end,
    })
}

/// Whether `dir`, an open directory, is still the one at `path`. `None` when
/// nothing is at `path`.
///
/// A directory can be removed, or removed and made again, while an operation
/// waits for its lock. Then the lock is on a directory that is gone.
fn still_at(dir: &File, path: &Path) -> io::Result<Option<bool>> {
    let held = dir.metadata()?;
    match fs::metadata(path) {
        Ok(named) => Ok(Some((held.dev(), held.ino()) == (named.dev(), named.ino()))),
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
            ) =>
        {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

/// Makes a new file at `path` and has `fill` write it. Fails if a file is
/// there. If `fill` fails, the new file is removed.
fn write_new(path: &Path, fill: impl FnOnce(&mut File) -> io::Result<()>) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    // This call made the file, so it is this call's to take away.
    fill(&mut file).inspect_err(|_| {
        let _ = fs::remove_file(path);
    })
}

/// Copies the journal's torn tail, which starts at byte `at`, to a file
/// beside the journal, unless a file `torn-<at>-<n>` already holds the same
/// bytes, and says so on standard error. Returns the log record
/// that reports it, for `iteration`.
fn set_aside(locked: &Locked, at: u64, torn: &[u8], iteration: u64) -> Result<Event, StoreError> {
    for n in 0_u32.. {
        let name = format!("torn-{at}-{n}");
        let path = locked.dir_path.join(&name);
        match fs::symlink_metadata(&path) {
            Ok(there) => {
                // A commit that then failed may have set these bytes aside
                // before. Anything else under the name is left alone: only a
                // plain file of the same length is read.
                let same = there.is_file()
                    && there.len() == torn.len() as u64
                    && fs::read(&path).is_ok_and(|old| old == torn);
                if !same {
                    continue;
                }
                File::open(&path)
                    .and_then(|file| file.sync_all())
                    .map_err(io_at(&path))?;
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                write_new(&path, |file| {
                    file.write_all(torn).and_then(|()| file.sync_all())
                })
                .map_err(io_at(&path))?;
            }
            Err(e) => return Err(io_at(&path)(e)),
        }
        locked.dir.sync_all().map_err(io_at(&locked.dir_path))?;
        let text = format!(
            "{} bytes after the last good record, at byte {at} of the journal, were not a \
             record. They are now in the file {name}",
            torn.len()
        );
        // Standard error may be closed. The log record is the report that
        // counts.
        let _ = writeln!(io::stderr(), "dc-mcp: {}: {text}", locked.path.display());
        return Ok(Event::Logged(LogRecord {
            kind: LogKind::Recovery,
            iteration,
            task: None,
            text,
        }));
    }
    unreachable!("the loop ends by returning")
}

/// The id of `journal` as the `refused` file writes it.
fn journal_id(journal: JournalId) -> String {
    format!("{:016x}", journal.0)
}

/// The refusals in the `refused` file of the run in `dir_path`, whose
/// journal is `journal`. None if there is no file or it is for another
/// journal. None too if it cannot be read or does not parse, which is
/// reported on standard error; the next refusal writes over it. So a damaged
/// file never stops a run.
fn read_refusals(dir_path: &Path, journal: JournalId) -> Refusals {
    let path = dir_path.join(REFUSED);
    let unreadable = |e: &dyn fmt::Display| {
        // Standard error may be closed; the counts start again from none,
        // and the next refusal writes over the file.
        let _ = writeln!(
            io::stderr(),
            "dc-mcp: {}: the refusal counts do not read ({e}); they count as none",
            path.display()
        );
        Refusals::default()
    };
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Refusals::default(),
        Err(e) => return unreadable(&e),
    };
    // The journal's id first, with any other fields, so that a file of
    // another journal counts as none whatever else it holds.
    #[derive(Deserialize)]
    struct Owner {
        journal: String,
    }
    let owner: Owner = match serde_json::from_slice(&bytes) {
        Ok(owner) => owner,
        Err(e) => return unreadable(&e),
    };
    if owner.journal != journal_id(journal) {
        return Refusals::default();
    }
    match serde_json::from_slice(&bytes) {
        Ok(refusals) => refusals,
        Err(e) => unreadable(&e),
    }
}

/// Adds one refusal of a submission of `kind` in `iteration` to the
/// `refused` file of the locked run, whose journal is `journal`. The new
/// count is on disk when it returns `Ok`; on an error the file holds the
/// old count or the new one.
fn count_refusal(
    locked: &Locked,
    journal: JournalId,
    iteration: u64,
    kind: RefusalKind,
) -> Result<(), StoreError> {
    let mut refusals = read_refusals(&locked.dir_path, journal);
    refusals.journal = journal_id(journal);
    let counts = refusals.iterations.entry(iteration).or_default();
    match kind {
        RefusalKind::Foreign => counts.foreign = counts.foreign.saturating_add(1),
        RefusalKind::Repeat => counts.repeats = counts.repeats.saturating_add(1),
    }
    // Serialising these types cannot fail: every map key is a number.
    let mut text = serde_json::to_vec(&refusals).expect("refusals serialise to JSON");
    text.push(b'\n');
    let new_path = locked.dir_path.join(REFUSED_NEW);
    File::create(&new_path)
        .and_then(|mut file| file.write_all(&text).and_then(|()| file.sync_all()))
        .map_err(io_at(&new_path))?;
    let path = locked.dir_path.join(REFUSED);
    fs::rename(&new_path, &path).map_err(io_at(&path))?;
    locked.dir.sync_all().map_err(io_at(&locked.dir_path))
}

/// Waits until the entries of directory `dir` are on disk.
fn sync_dir(dir: &Path) -> Result<(), StoreError> {
    File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(io_at(dir))
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
