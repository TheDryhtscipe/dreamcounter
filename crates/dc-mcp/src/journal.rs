//! The format of a run's journal: the file that holds every change to the run.
//!
//! A journal is a header line, [`HEADER`], and then one line for each record.
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
//! bytes. [`scan`] calls that a torn tail when no record with a right
//! checksum follows it. The commit was never acknowledged, so the store sets
//! the tail aside and goes on (see [`crate::store`]).
//!
//! Everything else that does not read back is corruption, and [`scan`] returns
//! an error:
//!
//! * a line that fails its checksum, or part of a line, with a good record
//!   after it;
//! * a line with a right checksum whose number is out of order or whose events
//!   do not parse.
//!
//! # What the format cannot detect
//!
//! Damage to the end of a journal looks like a torn tail. If the last records
//! are cut off or damaged after they were acknowledged, [`scan`] reads a
//! shorter run and reports no error. Nothing in the file says how long it
//! should be.

use std::fmt;

use crate::state::Event;

/// The first line of every journal. The number is the format version.
pub const HEADER: &str = "dc-mcp journal 1\n";

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
    let mut crc = u32::MAX;
    for byte in bytes {
        crc = CRC_TABLE[usize::from((crc as u8) ^ *byte)] ^ (crc >> 8);
    }
    !crc
}

/// The line for record `seq`, with its line break.
pub fn encode(seq: u64, events: &[Event]) -> String {
    // Serialising these types cannot fail: every map key is a string and no
    // `Serialize` implementation returns an error.
    let json = serde_json::to_string(events).expect("events serialise to JSON");
    let body = format!("{seq} {json}");
    format!("{seq} {:08x} {json}\n", crc32(body.as_bytes()))
}

/// Reads the records in `bytes`, which start at a record boundary.
///
/// `base` is the offset of `bytes` in the file, for error messages, and
/// `first_seq` is the number the first record must have.
pub fn scan(bytes: &[u8], base: u64, first_seq: u64) -> Result<Scan, Corrupt> {
    let mut records = Vec::new();
    let mut offsets = Vec::new();
    let mut start = 0;
    let mut seq = first_seq;
    while let Some(len) = bytes[start..].iter().position(|b| *b == b'\n') {
        let corrupt = |reason| Corrupt {
            offset: base + start as u64,
            reason,
        };
        let Some(json) = verified(&bytes[start..start + len], seq).map_err(corrupt)? else {
            break;
        };
        let events: Vec<Event> = serde_json::from_str(json)
            .map_err(|_| corrupt("a record whose events do not parse"))?;
        if events.is_empty() {
            return Err(corrupt("a record with no events"));
        }
        records.push(events);
        offsets.push(base + start as u64);
        start += len + 1;
        seq += 1;
    }
    // What is left is a torn tail, unless a good record follows it.
    let mut rest = &bytes[start..];
    while let Some(len) = rest.iter().position(|b| *b == b'\n') {
        if checksum_holds(&rest[..len]) {
            return Err(Corrupt {
                offset: base + start as u64,
                reason: "a damaged record with a good record after it",
            });
        }
        rest = &rest[len + 1..];
    }
    Ok(Scan {
        records,
        offsets,
        valid_len: start,
        torn: start < bytes.len(),
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
    (crc32(format!("{number} {json}").as_bytes()) == checksum).then_some((number, json))
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
