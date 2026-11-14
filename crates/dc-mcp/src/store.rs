//! Runs on disk.
//!
//! A [`Store`] is a directory with one subdirectory for each run, named by the
//! run's [`RunId`]. A run's directory holds two files:
//!
//! * `journal`: every change to the run, in the format of [`crate::journal`].
//! * `lock`: an empty file. A store holds an exclusive lock on it while it
//!   reads or writes the journal.
//!
//! # What a commit guarantees
//!
//! [`Store::commit`] and [`Store::create_run`] return `Ok` only after the
//! record is written and `fsync` has returned. So a change that was
//! acknowledged is in the journal when the server is killed and started again.
//! The events of one commit are one record: after a restart the run has all of
//! them or none.
//!
//! # More than one server
//!
//! Several stores, in one process or in several, can use the same run. Each
//! takes the lock, reads the records the others added since it last looked,
//! and only then acts. A store keeps each run's state in memory, so it reads
//! each record once.

use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use crate::journal::{self, Corrupt, HEADER};
use crate::state::Rejected;
pub use crate::state::{Event, LogRecord, RunConfig, RunId, RunState};

const JOURNAL: &str = "journal";
const JOURNAL_NEW: &str = "journal.new";
const LOCK: &str = "lock";

/// Why a store operation failed.
#[derive(Debug)]
pub enum StoreError {
    /// The store has no run of this name.
    NoSuchRun(RunId),
    /// The store already has a run of this name.
    RunExists(RunId),
    /// The events cannot be applied to the run. Nothing was written.
    Rejected(Rejected),
    /// The run's journal does not read back. The store does not change it.
    Corrupt {
        /// The run.
        run: RunId,
        /// What is wrong.
        error: Corrupt,
    },
    /// A file operation failed.
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The error.
        error: io::Error,
    },
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::NoSuchRun(run) => write!(f, "there is no run {run:?}", run = run.as_str()),
            StoreError::RunExists(run) => {
                write!(f, "there is already a run {run:?}", run = run.as_str())
            }
            StoreError::Rejected(why) => write!(f, "{why}"),
            StoreError::Corrupt { run, error } => write!(
                f,
                "the journal of run {run:?} is damaged: {error}",
                run = run.as_str()
            ),
            StoreError::Io { path, error } => write!(f, "{}: {error}", path.display()),
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StoreError::Rejected(why) => Some(why),
            StoreError::Corrupt { error, .. } => Some(error),
            StoreError::Io { error, .. } => Some(error),
            StoreError::NoSuchRun(_) | StoreError::RunExists(_) => None,
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

/// What a store remembers of a run between operations.
#[derive(Debug)]
struct Cached {
    state: RunState,
    /// The length of the journal the state was read from.
    len: u64,
    /// The number of the journal's next record.
    next_seq: u64,
}

/// A run's files, open and locked. Dropping it releases the lock.
#[derive(Debug)]
struct Locked {
    journal: File,
    path: PathBuf,
    _lock: File,
}

/// A directory of runs. See the module docs.
#[derive(Debug)]
pub struct Store {
    root: PathBuf,
    /// Held for the whole of each operation, so the threads of one process
    /// take turns.
    cache: Mutex<HashMap<RunId, Cached>>,
}

impl Store {
    /// The store in directory `root`, which is made if it does not exist.
    pub fn open(root: &Path) -> Result<Store, StoreError> {
        fs::create_dir_all(root).map_err(io_at(root))?;
        Ok(Store {
            root: root.to_owned(),
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Makes a run at iteration 0 and returns its state.
    pub fn create_run(
        &self,
        run: &RunId,
        config: RunConfig,
        tasks: Vec<String>,
    ) -> Result<RunState, StoreError> {
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        let state =
            RunState::create(config.clone(), tasks.clone()).map_err(StoreError::Rejected)?;
        let dir = self.root.join(run.as_str());
        match fs::create_dir(&dir) {
            Ok(()) => sync_dir(&self.root)?,
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(io_at(&dir)(e)),
        }
        let lock_path = dir.join(LOCK);
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .map_err(io_at(&lock_path))?;
        lock.lock().map_err(io_at(&lock_path))?;

        let path = dir.join(JOURNAL);
        if path.try_exists().map_err(io_at(&path))? {
            return Err(StoreError::RunExists(run.clone()));
        }
        // The journal appears under its name only when it is complete, so a
        // run that exists always has its first record.
        let new_path = dir.join(JOURNAL_NEW);
        let mut file = File::create(&new_path).map_err(io_at(&new_path))?;
        let first = journal::encode(0, &[Event::Created { config, tasks }]);
        file.write_all(HEADER.as_bytes())
            .and_then(|()| file.write_all(first.as_bytes()))
            .and_then(|()| file.sync_all())
            .map_err(io_at(&new_path))?;
        fs::rename(&new_path, &path).map_err(io_at(&path))?;
        sync_dir(&dir)?;
        cache.insert(
            run.clone(),
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

    /// The run's state now.
    pub fn state(&self, run: &RunId) -> Result<RunState, StoreError> {
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        let mut locked = self.lock_run(run)?;
        Ok(catch_up(&mut cache, run, &mut locked)?.state.clone())
    }

    /// Changes the run by the events `change` returns, all of them or none.
    ///
    /// `change` sees the run's state now. No other store changes the run
    /// between that and the write. `Ok` means the events are on disk: see the
    /// module docs.
    pub fn commit<F>(&self, run: &RunId, change: F) -> Result<(), StoreError>
    where
        F: FnOnce(&RunState) -> Vec<Event>,
    {
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        let mut locked = self.lock_run(run)?;
        let cached = catch_up(&mut cache, run, &mut locked)?;
        let events = change(&cached.state);
        if events.is_empty() {
            return Ok(());
        }
        let mut state = cached.state.clone();
        for event in &events {
            state.apply(event).map_err(StoreError::Rejected)?;
        }
        let record = journal::encode(cached.next_seq, &events);
        let written = append(&mut locked, cached.len, record.as_bytes());
        if written.is_err() {
            // The file may hold part of the record. Read it again next time.
            cache.remove(run);
            return written;
        }
        cached.state = state;
        cached.len += record.len() as u64;
        cached.next_seq += 1;
        Ok(())
    }

    /// Every log record of the run, oldest first.
    pub fn log(&self, run: &RunId) -> Result<Vec<LogRecord>, StoreError> {
        let _cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
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

    /// Opens and locks the run's files.
    fn lock_run(&self, run: &RunId) -> Result<Locked, StoreError> {
        let dir = self.root.join(run.as_str());
        let missing = |e: &io::Error| e.kind() == io::ErrorKind::NotFound;
        let lock_path = dir.join(LOCK);
        let lock = match File::open(&lock_path) {
            Ok(lock) => lock,
            Err(e) if missing(&e) => return Err(StoreError::NoSuchRun(run.clone())),
            Err(e) => return Err(io_at(&lock_path)(e)),
        };
        lock.lock().map_err(io_at(&lock_path))?;
        let path = dir.join(JOURNAL);
        match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(journal) => Ok(Locked {
                journal,
                path,
                _lock: lock,
            }),
            Err(e) if missing(&e) => Err(StoreError::NoSuchRun(run.clone())),
            Err(e) => Err(io_at(&path)(e)),
        }
    }
}

/// Reads the journal from byte `from`, where record `first_seq` starts or the
/// header does if `from` is 0. Returns the records and the journal's length.
fn read_from(
    locked: &mut Locked,
    run: &RunId,
    from: u64,
    first_seq: u64,
) -> Result<(journal::Scan, u64), StoreError> {
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
        body = body
            .strip_prefix(HEADER.as_bytes())
            .ok_or_else(|| corrupt(0, "a file that does not start with the journal header"))?;
        base = HEADER.len() as u64;
    }
    let scan = journal::scan(body, base, first_seq).map_err(|error| StoreError::Corrupt {
        run: run.clone(),
        error,
    })?;
    Ok((scan, len))
}

/// Brings the cached state of `run` up to the journal on disk.
fn catch_up<'a>(
    cache: &'a mut HashMap<RunId, Cached>,
    run: &RunId,
    locked: &mut Locked,
) -> Result<&'a mut Cached, StoreError> {
    let (from, first_seq) = cache.get(run).map_or((0, 0), |c| (c.len, c.next_seq));
    let fresh = read_from(locked, run, from, first_seq).and_then(|(scan, _)| {
        let rejected = |why: Rejected| StoreError::Corrupt {
            run: run.clone(),
            error: Corrupt {
                offset: from,
                reason: match why {
                    Rejected::NotCreated => "a journal whose first record does not create the run",
                    _ => "a record that cannot be applied to the run",
                },
            },
        };
        let (mut state, count) = match cache.remove(run) {
            Some(cached) => (cached.state, cached.next_seq),
            None => {
                let first = scan.records.first().map_or(&[][..], Vec::as_slice);
                (RunState::replay(first).map_err(rejected)?, 0)
            }
        };
        let skip = usize::from(count == 0);
        for event in scan.records.iter().skip(skip).flatten() {
            state.apply(event).map_err(rejected)?;
        }
        let header = if from == 0 { HEADER.len() as u64 } else { 0 };
        Ok(Cached {
            state,
            len: from + header + scan.valid_len as u64,
            next_seq: first_seq + scan.records.len() as u64,
        })
    })?;
    Ok(cache.entry(run.clone()).or_insert(fresh))
}

/// Writes `record` at byte `at` of the journal, the end of its last complete
/// record, and waits until it is on disk. Any torn tail after `at` is dropped
/// first.
fn append(locked: &mut Locked, at: u64, record: &[u8]) -> Result<(), StoreError> {
    let file = &mut locked.journal;
    file.set_len(at)
        .and_then(|()| file.seek(SeekFrom::Start(at)))
        .and_then(|_| file.write_all(record))
        .and_then(|()| file.sync_data())
        .map_err(io_at(&locked.path))
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
