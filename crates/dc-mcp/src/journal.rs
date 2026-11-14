//! The format of a run's journal: the file that holds every change to the run.
//!
//! A journal is a header line (see [`header`]) and then one line for each
//! record.
//! A record is the events of one commit, which take effect together. Its line
//! is
//!
//! ```text
//! <number> <checksum> <events>
//! ```
//!
//! * `<number>` counts records from 0, in decimal.
//! * `<events>` is the events as a JSON array. JSON escapes line breaks, so a
//!   record never contains one.
//! * `<checksum>` is the CRC-32 of `<number> <events>`, as eight hexadecimal
//!   digits.
//!
//! # Torn tails and corruption
//!
//! A write that the server or the machine did not finish can leave anything
//! after the last complete record: part of a line, or a whole line with wrong
//! bytes. [`scan_records`] calls that a torn tail when no record with a right
//! checksum follows it. The commit was never acknowledged, so the store sets
//! the tail aside and goes on (see [`crate::store`]).
//!
//! Everything else that does not read back is corruption, and [`scan_records`] returns
//! an error:
//!
//! * a line that fails its checksum with a good record on a later line;
//! * a line with a right checksum whose number is out of order or whose events
//!   do not parse.
//!
//! # What the format cannot detect
//!
//! [`scan_records`] reads line by line. If the line break before the last record is
//! lost, that record is part of a bad last line, and so part of a torn tail.
//!
//! Damage to the end of a journal looks like a torn tail. If the last records
//! are cut off or damaged after they were acknowledged, [`scan_records`] reads a
//! shorter run and reports no error. Nothing in the file says how long it
//! should be.

use std::convert::Infallible;
use std::fmt;
use std::io::{self, BufRead};

use crate::state::Event;

/// What the header line of every journal starts with, in every format version.
pub const HEADER_PREFIX: &str = "dc-mcp journal ";

/// The format version this code reads and writes.
pub const FORMAT_VERSION: &str = "1";

/// The longest header line, with its line break, in bytes.
pub const MAX_HEADER_LEN: usize = 64;

/// The header line of a journal: the format version, and an id that is
/// chosen at random when the journal is made. Two journals with different ids
/// are different runs, even under the same name.
pub fn header(id: u64) -> String {
    format!("{HEADER_PREFIX}{FORMAT_VERSION} {id:016x}\n")
}

/// Why the start of a file is not a header this code reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeaderError {
    /// The file does not start with a journal header.
    NotAJournal,
    /// The header is of another format version. This is the header line.
    Unsupported(String),
}

/// Reads the header at the start of `bytes`: the journal's id, and the
/// header's length in bytes.
pub fn parse_header(bytes: &[u8]) -> Result<(u64, usize), HeaderError> {
    let start = &bytes[..bytes.len().min(MAX_HEADER_LEN)];
    let line = start
        .iter()
        .position(|b| *b == b'\n')
        .and_then(|end| std::str::from_utf8(&start[..end]).ok())
        .and_then(|line| line.strip_prefix(HEADER_PREFIX).map(|rest| (line, rest)));
    let Some((line, rest)) = line else {
        return Err(HeaderError::NotAJournal);
    };
    let unsupported = || HeaderError::Unsupported(line.to_owned());
    let (version, id) = rest.split_once(' ').ok_or_else(unsupported)?;
    let lower_hex = |b: u8| b.is_ascii_digit() || (b'a'..=b'f').contains(&b);
    if version != FORMAT_VERSION || id.len() != 16 || !id.bytes().all(lower_hex) {
        return Err(unsupported());
    }
    let id = u64::from_str_radix(id, 16).map_err(|_| unsupported())?;
    Ok((id, line.len() + 1))
}

/// A journal does not read back. See the module docs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Corrupt {
    /// Where the bad line starts, in bytes from the start of the file.
    pub offset: u64,
    /// What is wrong with it.
    pub reason: &'static str,
}

impl fmt::Display for Corrupt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at byte {}", self.reason, self.offset)
    }
}

impl std::error::Error for Corrupt {}

/// The CRC-32 table for the IEEE polynomial, one entry for each byte value.
const CRC_TABLE: [u32; 256] = {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// The CRC-32 of `bytes` (the IEEE polynomial, as in zlib and PNG).
pub fn crc32(bytes: &[u8]) -> u32 {
    !crc32_update(u32::MAX, bytes)
}

/// Runs the CRC-32 register `crc`, not yet inverted, over `bytes`. So the
/// CRC-32 of `a` then `b` is `!crc32_update(crc32_update(u32::MAX, a), b)`.
fn crc32_update(mut crc: u32, bytes: &[u8]) -> u32 {
    for byte in bytes {
        crc = CRC_TABLE[usize::from((crc as u8) ^ *byte)] ^ (crc >> 8);
    }
    crc
}

/// The line for record `seq`, with its line break.
pub fn encode(seq: u64, events: &[Event]) -> String {
    // Serialising these types cannot fail: every map key is a string and no
    // `Serialize` implementation returns an error.
    let json = serde_json::to_string(events).expect("events serialise to JSON");
    let body = format!("{seq} {json}");
    format!("{seq} {:08x} {json}\n", crc32(body.as_bytes()))
}

/// One good record, as [`scan_records`] reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    /// Where the record starts, in bytes from the start of the file.
    pub offset: u64,
    /// The record's events, at least one.
    pub events: Vec<Event>,
}

/// What [`scan_records`] read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct End {
    /// How many good records there are.
    pub records: u64,
    /// The length of the good records, in bytes.
    pub valid_len: u64,
    /// The length of the torn tail after the good records, in bytes. It is 0
    /// when there is no torn tail.
    pub torn_len: u64,
    /// Where the last good record starts, in bytes from the start of the
    /// file, and the length of its line with the line break.
    pub last: Option<(u64, u64)>,
}

/// Why [`scan_records`] stopped before the end.
#[derive(Debug)]
pub enum ScanError<E> {
    /// The records do not read back. See the module docs.
    Corrupt(Corrupt),
    /// Reading failed.
    Read(io::Error),
    /// The caller's function returned this error.
    Stopped(E),
}

/// Reads the records from `reader`, which starts at a record boundary, and
/// hands each good record to `each` as soon as it is read, before the next
/// one is read.
///
/// So the scan holds one line in memory at a time, and the caller decides
/// what to keep. If `each` returns an error, the scan stops and returns it.
/// A corrupt line after records that `each` was given still makes the scan an
/// error: the caller must not act on those records until the scan returns
/// `Ok`.
///
/// `base` is the offset of `reader` in the file, for offsets and error
/// messages, and `first_seq` is the number the first record must have.
pub fn scan_records<R, E, F>(
    mut reader: R,
    base: u64,
    first_seq: u64,
    mut each: F,
) -> Result<End, ScanError<E>>
where
    R: BufRead,
    F: FnMut(Record) -> Result<(), E>,
{
    let mut line = Vec::new();
    let mut start = 0_u64;
    let mut seq = first_seq;
    let mut last = None;
    // The length of the first line that is not a good record, if there is one.
    let first_bad = loop {
        line.clear();
        let len = reader
            .read_until(b'\n', &mut line)
            .map_err(ScanError::Read)?;
        if len == 0 {
            break None;
        }
        let Some(body) = line.strip_suffix(b"\n") else {
            break Some(len as u64);
        };
        let offset = base + start;
        let corrupt = |reason| ScanError::Corrupt(Corrupt { offset, reason });
        let Some(json) = verified(body, seq).map_err(corrupt)? else {
            break Some(len as u64);
        };
        let events: Vec<Event> = serde_json::from_str(json)
            .map_err(|_| corrupt("a record whose events do not parse"))?;
        if events.is_empty() {
            return Err(corrupt("a record with no events"));
        }
        each(Record { offset, events }).map_err(ScanError::Stopped)?;
        last = Some((offset, len as u64));
        start += len as u64;
        seq += 1;
    };
    let Some(mut torn_len) = first_bad else {
        return Ok(End {
            records: seq - first_seq,
            valid_len: start,
            torn_len: 0,
            last,
        });
    };
    // What is left is a torn tail, unless a good record follows it. The
    // first bad line has no right checksum, so only the lines after it are
    // checked.
    loop {
        line.clear();
        let len = reader
            .read_until(b'\n', &mut line)
            .map_err(ScanError::Read)?;
        if len == 0 {
            break;
        }
        if line.strip_suffix(b"\n").is_some_and(checksum_holds) {
            return Err(ScanError::Corrupt(Corrupt {
                offset: base + start,
                reason: "a damaged record with a good record after it",
            }));
        }
        torn_len += len as u64;
    }
    Ok(End {
        records: seq - first_seq,
        valid_len: start,
        torn_len,
        last,
    })
}

/// What [`scan`] read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scan {
    /// The events of each good record, in order.
    pub records: Vec<Vec<Event>>,
    /// Where each good record starts, in bytes from the start of the file.
    pub offsets: Vec<u64>,
    /// The length of the good records, in bytes.
    pub valid_len: usize,
    /// Whether a torn tail follows the good records.
    pub torn: bool,
}

/// Reads all the records in `bytes`, which start at a record boundary, as
/// [`scan_records`] does, and collects them. For a journal on disk use
/// [`scan_records`], which does not hold every record at once.
pub fn scan(bytes: &[u8], base: u64, first_seq: u64) -> Result<Scan, Corrupt> {
    let mut records = Vec::new();
    let mut offsets = Vec::new();
    let collect = |record: Record| {
        offsets.push(record.offset);
        records.push(record.events);
        Ok::<(), Infallible>(())
    };
    let end = match scan_records(bytes, base, first_seq, collect) {
        Ok(end) => end,
        Err(ScanError::Corrupt(corrupt)) => return Err(corrupt),
        Err(ScanError::Read(error)) => unreachable!("reading a slice does not fail: {error}"),
        Err(ScanError::Stopped(never)) => match never {},
    };
    Ok(Scan {
        records,
        offsets,
        valid_len: end.valid_len as usize,
        torn: end.torn_len > 0,
    })
}

/// The three fields of a line whose checksum is right.
fn fields(line: &[u8]) -> Option<(&str, &str)> {
    let line = std::str::from_utf8(line).ok()?;
    let mut parts = line.splitn(3, ' ');
    let (number, checksum, json) = (parts.next()?, parts.next()?, parts.next()?);
    let lower_hex = |b: u8| b.is_ascii_digit() || (b'a'..=b'f').contains(&b);
    if checksum.len() != 8 || !checksum.bytes().all(lower_hex) {
        return None;
    }
    let checksum = u32::from_str_radix(checksum, 16).ok()?;
    // The checksum covers `<number> <events>`: the line without its checksum.
    let crc = [number.as_bytes(), b" ", json.as_bytes()]
        .into_iter()
        .fold(u32::MAX, crc32_update);
    (!crc == checksum).then_some((number, json))
}

/// Whether `line`, without its line break, has a right checksum.
fn checksum_holds(line: &[u8]) -> bool {
    fields(line).is_some()
}

/// The events of record `seq` as JSON text, or `None` if the line's checksum
/// is wrong. A right checksum with another number is an error.
fn verified(line: &[u8], seq: u64) -> Result<Option<&str>, &'static str> {
    let Some((number, json)) = fields(line) else {
        return Ok(None);
    };
    if number != seq.to_string() {
        return Err("a record out of order");
    }
    Ok(Some(json))
}

#[cfg(test)]
#[path = "journal_tests.rs"]
mod tests;
