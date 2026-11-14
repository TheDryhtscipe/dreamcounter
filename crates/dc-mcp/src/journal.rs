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
//! The store appends a record with one write. If the server dies during the
//! write, the file can end with part of a line. [`scan`] reports that as a torn
//! tail. The commit was never acknowledged, so the store drops the part line.
//!
//! Anything else that does not read back is corruption, and [`scan`] returns
//! an error: a complete line with a wrong checksum, a number out of order, or
//! events that do not parse. The store then refuses to use the run.

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
    /// The events of each complete record, in order.
    pub records: Vec<Vec<Event>>,
    /// The length of the complete records, in bytes.
    pub valid_len: usize,
    /// Whether part of a line follows the complete records.
    pub torn: bool,
}

/// The CRC-32 of `bytes` (the IEEE polynomial, as in zlib and PNG).
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let low = crc & 1;
            crc >>= 1;
            if low == 1 {
                crc ^= 0xEDB8_8320;
            }
        }
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
    let mut start = 0;
    let mut seq = first_seq;
    while let Some(len) = bytes[start..].iter().position(|b| *b == b'\n') {
        let events = decode(&bytes[start..start + len], seq).map_err(|reason| Corrupt {
            offset: base + start as u64,
            reason,
        })?;
        records.push(events);
        start += len + 1;
        seq += 1;
    }
    Ok(Scan {
        records,
        valid_len: start,
        torn: start < bytes.len(),
    })
}

/// The events of one line, without its line break.
fn decode(line: &[u8], seq: u64) -> Result<Vec<Event>, &'static str> {
    let line = std::str::from_utf8(line).map_err(|_| "a record that is not UTF-8")?;
    let mut parts = line.splitn(3, ' ');
    let (Some(number), Some(checksum), Some(json)) = (parts.next(), parts.next(), parts.next())
    else {
        return Err("a record with missing fields");
    };
    let canonical = |text: &str, radix| {
        text.bytes()
            .all(|b| char::from(b).is_digit(radix) && !b.is_ascii_uppercase())
    };
    if !canonical(number, 10) || checksum.len() != 8 || !canonical(checksum, 16) {
        return Err("a record with a malformed number or checksum");
    }
    let checksum =
        u32::from_str_radix(checksum, 16).map_err(|_| "a record with a malformed checksum")?;
    if crc32(format!("{number} {json}").as_bytes()) != checksum {
        return Err("a record with a wrong checksum");
    }
    if number.parse() != Ok(seq) || number != seq.to_string() {
        return Err("a record out of order");
    }
    let events: Vec<Event> =
        serde_json::from_str(json).map_err(|_| "a record whose events do not parse")?;
    if events.is_empty() {
        return Err("a record with no events");
    }
    Ok(events)
}

#[cfg(test)]
#[path = "journal_tests.rs"]
mod tests;
