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
//! # More than one server
//!
//! Several stores, in one process or in several, can use the same run. Each
//! operation takes an exclusive lock on the run's directory (`flock`), reads
//! the records the others added since it last looked, and only then acts. A
//! store keeps each run's state in memory, so reading a run's state parses
//! each record once. The lock needs a local file system.
//!
//! An operation waits for the lock for a limited time, [`DEFAULT_LOCK_WAIT`]
//! unless the store was opened with another, and then fails with
//! [`StoreError::Busy`].
//!
//! # Torn tails
//!
//! A write that did not finish leaves a torn tail after the last good record
//! (see [`crate::journal`]). Reading ignores it. The next commit moves it to a
//! file `torn-<offset>-<n>` beside the journal, says so on standard error, and
//! writes its record in the tail's place. The store deletes nothing. Damage to
//! the end of a journal looks the same, so an operator who sees such a file
//! after a clean stop should treat it as lost commits.

use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use crate::journal::{self, Corrupt, HEADER};
use crate::state::Rejected;
pub use crate::state::{Event, LogRecord, RunConfig, RunId, RunState};

const JOURNAL: &str = "journal";
const JOURNAL_NEW: &str = "journal.new";
/// What every version of the journal header starts with.
const HEADER_PREFIX: &str = "dc-mcp journal ";

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
    /// A commit's write failed, and so did taking the record out again. The
    /// commit may or may not be in the run: read the run to find out.
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
                "run {:?} is locked by another server. Nothing was changed",
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

/// What a store remembers of a run between operations.
#[derive(Debug)]
struct Cached {
    state: RunState,
    /// The length of the journal's good records.
    len: u64,
    /// The number of the journal's next record.
    next_seq: u64,
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

/// A directory of runs. See the module docs.
#[derive(Debug)]
pub struct Store {
    root: PathBuf,
    lock_wait: Duration,
    /// A run's entry is taken out while an operation holds the run's lock,
    /// and put back if the operation leaves it true to the journal.
    cache: Mutex<HashMap<RunId, Cached>>,
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

        let path = dir_path.join(JOURNAL);
        if path.try_exists().map_err(io_at(&path))? {
            return Err(StoreError::RunExists(run.clone()));
        }
        // The journal appears under its name only when it is complete, so a
        // run that exists always has its first record.
        let new_path = dir_path.join(JOURNAL_NEW);
        let mut file = File::create(&new_path).map_err(io_at(&new_path))?;
        let first = journal::encode(0, &[created]);
        file.write_all(HEADER.as_bytes())
            .and_then(|()| file.write_all(first.as_bytes()))
            .and_then(|()| file.sync_all())
            .map_err(io_at(&new_path))?;
        fs::rename(&new_path, &path).map_err(io_at(&path))?;
        dir.sync_all().map_err(io_at(&dir_path))?;
        self.put(
            run,
            Cached {
                state: state.clone(),
                len: (HEADER.len() + first.len()) as u64,
                next_seq: 1,
            },
        );
        Ok(state)
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
        let mut locked = self.lock_run(run)?;
        let (cached, _) = self.catch_up(run, &mut locked)?;
        let result = read(&cached.state);
        self.put(run, cached);
        Ok(result)
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
        let mut locked = self.lock_run(run)?;
        let (mut cached, torn) = self.catch_up(run, &mut locked)?;
        let (value, events) = match change(&cached.state) {
            Ok(changed) => changed,
            Err(error) => {
                self.put(run, cached);
                return Err(error);
            }
        };
        if events.is_empty() {
            self.put(run, cached);
            return Ok(value);
        }
        let mut state = cached.state.clone();
        for event in &events {
            if let Err(why) = state.apply(event) {
                self.put(run, cached);
                return Err(StoreError::Rejected(why).into());
            }
        }
        if !torn.is_empty() {
            set_aside(&locked, cached.len, &torn)?;
        }
        let record = journal::encode(cached.next_seq, &events);
        match append(&mut locked.journal, cached.len, record.as_bytes()) {
            Ok(()) => {}
            Err(AppendError::NotWritten(error)) => {
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
        cached.len += record.len() as u64;
        cached.next_seq += 1;
        self.put(run, cached);
        Ok(value)
    }

    /// Changes the run by `events`, all of them or none. See [`Store::commit`].
    pub fn append(&self, run: &RunId, events: Vec<Event>) -> Result<(), StoreError> {
        self.commit(run, |_| Ok(((), events)))
    }

    /// Every log record of the run, oldest first. This reads the whole
    /// journal each time.
    pub fn log(&self, run: &RunId) -> Result<Vec<LogRecord>, StoreError> {
        let mut locked = self.lock_run(run)?;
        let (scan, _) = read_from(&mut locked, run, 0, 0)?;
        Ok(scan
            .records
            .into_iter()
            .flatten()
            .filter_map(|event| match event {
                Event::Logged(record) => Some(record),
                _ => None,
            })
            .collect())
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
        let deadline = Instant::now() + self.lock_wait;
        let mut pause = Duration::from_micros(200);
        loop {
            match dir.try_lock() {
                Ok(()) => return Ok(()),
                Err(TryLockError::Error(e)) => return Err(io_at(dir_path)(e)),
                Err(TryLockError::WouldBlock) => {}
            }
            if Instant::now() >= deadline {
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

    /// The state of `run` as the journal on disk has it, and the journal's
    /// torn tail. On an error the store forgets the run, and the next
    /// operation reads its whole journal.
    fn catch_up(&self, run: &RunId, locked: &mut Locked) -> Result<(Cached, Vec<u8>), StoreError> {
        let cached = self
            .cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(run);
        let (from, first_seq) = cached.as_ref().map_or((0, 0), |c| (c.len, c.next_seq));
        let (scan, torn) = read_from(locked, run, from, first_seq)?;
        let rejected = |index: usize, why: Rejected| StoreError::Corrupt {
            run: run.clone(),
            error: Corrupt {
                offset: scan.offsets[index],
                reason: match why {
                    Rejected::NotCreated => "a first record that does not create the run",
                    _ => "a record that cannot be applied to the run",
                },
            },
        };
        let mut records = scan.records.iter().enumerate();
        let mut state = match cached {
            Some(cached) => cached.state,
            None => match records.next() {
                Some((index, first)) => RunState::replay(first).map_err(|w| rejected(index, w))?,
                None => {
                    return Err(StoreError::Corrupt {
                        run: run.clone(),
                        error: Corrupt {
                            offset: HEADER.len() as u64,
                            reason: "a journal with no records",
                        },
                    })
                }
            },
        };
        for (index, events) in records {
            for event in events {
                state.apply(event).map_err(|w| rejected(index, w))?;
            }
        }
        let header = if from == 0 { HEADER.len() as u64 } else { 0 };
        let cached = Cached {
            state,
            len: from + header + scan.valid_len as u64,
            next_seq: first_seq + scan.records.len() as u64,
        };
        Ok((cached, torn))
    }
}

/// Reads the journal from byte `from`, where record `first_seq` starts or the
/// header does if `from` is 0. Returns the records and the torn tail.
fn read_from(
    locked: &mut Locked,
    run: &RunId,
    from: u64,
    first_seq: u64,
) -> Result<(journal::Scan, Vec<u8>), StoreError> {
    let corrupt = |offset, reason| StoreError::Corrupt {
        run: run.clone(),
        error: Corrupt { offset, reason },
    };
    let file = &mut locked.journal;
    let len = file.metadata().map_err(io_at(&locked.path))?.len();
    if len < from {
        return Err(corrupt(len, "a journal that is shorter than it was"));
    }
    let mut bytes = Vec::new();
    file.seek(SeekFrom::Start(from))
        .and_then(|_| file.read_to_end(&mut bytes))
        .map_err(io_at(&locked.path))?;
    let mut body = &bytes[..];
    let mut base = from;
    if from == 0 {
        body = match body.strip_prefix(HEADER.as_bytes()) {
            Some(body) => body,
            None if body.starts_with(HEADER_PREFIX.as_bytes()) => {
                let line = body.split(|b| *b == b'\n').next().unwrap_or(body);
                let shown = &line[..line.len().min(64)];
                return Err(StoreError::UnsupportedFormat {
                    run: run.clone(),
                    header: String::from_utf8_lossy(shown).into_owned(),
                });
            }
            None => {
                return Err(corrupt(
                    0,
                    "a file that does not start with the journal header",
                ))
            }
        };
        base = HEADER.len() as u64;
    }
    let scan = journal::scan(body, base, first_seq).map_err(|error| StoreError::Corrupt {
        run: run.clone(),
        error,
    })?;
    let torn = body[scan.valid_len..].to_vec();
    Ok((scan, torn))
}

/// Copies the journal's torn tail, which starts at byte `at`, to a new file
/// beside the journal, and says so on standard error.
fn set_aside(locked: &Locked, at: u64, torn: &[u8]) -> Result<(), StoreError> {
    for n in 0_u32.. {
        let path = locked.dir_path.join(format!("torn-{at}-{n}"));
        let mut file = match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(io_at(&path)(e)),
        };
        file.write_all(torn)
            .and_then(|()| file.sync_all())
            .map_err(io_at(&path))?;
        locked.dir.sync_all().map_err(io_at(&locked.dir_path))?;
        eprintln!(
            "dc-mcp: {}: {} bytes after the last good record at byte {at} were not a record. \
             They are now in {}",
            locked.path.display(),
            torn.len(),
            path.display()
        );
        return Ok(());
    }
    unreachable!("the loop ends by returning")
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
