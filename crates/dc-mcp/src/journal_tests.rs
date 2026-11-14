use super::*;
use crate::state::{Arm, Event, LogKind, LogRecord, RunConfig};

fn created() -> Event {
    Event::Created {
        config: RunConfig {
            arm: Arm::Blind,
            seed: 1,
            proposals_per_task: 2,
        },
        tasks: vec!["a".into()],
    }
}

fn note(text: &str) -> Event {
    Event::Logged(LogRecord {
        kind: LogKind::Note,
        iteration: 0,
        task: None,
        text: text.into(),
    })
}

/// A journal body of three records, and the offset where each one ends.
fn three() -> (Vec<u8>, Vec<usize>) {
    let mut bytes = Vec::new();
    let mut ends = Vec::new();
    for (seq, events) in [
        vec![created()],
        vec![note("line\nbreak \u{1F980} \"quoted\"")],
        vec![note("x"), Event::IterationFinished { iteration: 0 }],
    ]
    .into_iter()
    .enumerate()
    {
        bytes.extend_from_slice(encode(seq as u64, &events).as_bytes());
        ends.push(bytes.len());
    }
    (bytes, ends)
}

#[test]
fn crc32_matches_the_standard_check_value() {
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    assert_eq!(crc32(b""), 0);
}

#[test]
fn a_record_is_one_line() {
    let line = encode(5, &[note("a\nb\r\nc")]);
    assert!(line.ends_with('\n'));
    assert_eq!(line.matches('\n').count(), 1);
    assert!(line.starts_with("5 "));
}

#[test]
fn scan_reads_back_what_encode_wrote() {
    let (bytes, ends) = three();
    let scan = scan(&bytes, 0, 0).unwrap();
    assert_eq!(scan.valid_len, bytes.len());
    assert!(!scan.torn);
    assert_eq!(scan.records.len(), 3);
    assert_eq!(scan.records[0], vec![created()]);
    assert_eq!(
        scan.records[2],
        vec![note("x"), Event::IterationFinished { iteration: 0 }]
    );

    // A scan can start at any record boundary, given that record's number.
    let tail = super::scan(&bytes[ends[0]..], ends[0] as u64, 1).unwrap();
    assert_eq!(tail.records.len(), 2);
    assert_eq!(tail.valid_len, bytes.len() - ends[0]);
}

#[test]
fn a_cut_inside_the_last_record_is_a_torn_tail() {
    let (bytes, ends) = three();
    for cut in ends[1] + 1..ends[2] {
        let scan = scan(&bytes[..cut], 0, 0).unwrap();
        assert!(scan.torn, "cut at {cut}");
        assert_eq!(scan.valid_len, ends[1], "cut at {cut}");
        assert_eq!(scan.records.len(), 2, "cut at {cut}");
    }
    let whole = scan(&bytes[..ends[1]], 0, 0).unwrap();
    assert!(!whole.torn);
    assert_eq!(whole.records.len(), 2);
}

#[test]
fn a_changed_byte_is_corruption_unless_only_the_tail_is_damaged() {
    let (bytes, ends) = three();
    let starts = [0, ends[0], ends[1]];
    for i in 0..bytes.len() {
        let mut bad = bytes.clone();
        bad[i] ^= 0x01;
        // A changed line break joins a record to the next one.
        let record = starts.iter().rposition(|s| *s <= i).unwrap();
        let last_damaged = if bytes[i] == b'\n' {
            record + 1
        } else {
            record
        };
        match scan(&bad, 0, 0) {
            Err(Corrupt { offset, .. }) => {
                assert!(
                    last_damaged < 2,
                    "byte {i}: damage to the tail alone is an error"
                );
                assert_eq!(offset, starts[record] as u64, "byte {i}");
            }
            Ok(scan) => {
                assert!(last_damaged >= 2, "byte {i} changed before a good record");
                assert!(scan.torn, "byte {i}");
                assert_eq!(scan.valid_len, starts[record], "byte {i}");
                assert_eq!(scan.records.len(), record, "byte {i}");
            }
        }
    }
}

#[test]
fn bytes_after_the_last_record_are_a_torn_tail_unless_a_good_record_follows() {
    let (bytes, ends) = three();
    for junk in [
        &b"\0\0\0\0"[..],
        b"\n",
        b"junk\n",
        b"\n\n7 00000000 []\n",
        b"9 zz",
    ] {
        let mut with = bytes.clone();
        with.extend_from_slice(junk);
        let scan = scan(&with, 0, 0).unwrap();
        assert!(scan.torn, "{junk:?}");
        assert_eq!(scan.valid_len, bytes.len(), "{junk:?}");
        assert_eq!(scan.records.len(), 3, "{junk:?}");
        assert_eq!(scan.offsets, [0, ends[0] as u64, ends[1] as u64]);
    }
    // Junk, then a record with a right checksum: the junk is not a tail.
    let mut with = bytes[..ends[1]].to_vec();
    with.extend_from_slice(b"junk\n");
    with.extend_from_slice(&bytes[ends[1]..]);
    assert_eq!(scan(&with, 0, 0).unwrap_err().offset, ends[1] as u64);
}

#[test]
fn a_record_with_a_right_checksum_and_bad_events_is_corruption() {
    for json in ["[", "[{\"nonsense\":1}]", "{}"] {
        let body = format!("0 {json}");
        let line = format!("0 {:08x} {json}\n", crc32(body.as_bytes()));
        assert_eq!(scan(line.as_bytes(), 0, 0).unwrap_err().offset, 0, "{json}");
    }
}

#[test]
fn a_missing_or_repeated_record_is_corruption() {
    let (bytes, ends) = three();
    let mut missing = bytes[..ends[0]].to_vec();
    missing.extend_from_slice(&bytes[ends[1]..]);
    assert_eq!(scan(&missing, 0, 0).unwrap_err().offset, ends[0] as u64);

    let mut repeated = bytes[..ends[1]].to_vec();
    repeated.extend_from_slice(&bytes[ends[0]..]);
    assert_eq!(scan(&repeated, 0, 0).unwrap_err().offset, ends[1] as u64);

    // The same bytes are fine when the scan is told the right first number.
    assert!(scan(&bytes[ends[0]..], 0, 0).is_err());
    assert!(scan(&bytes[ends[0]..], 0, 1).is_ok());
}

#[test]
fn a_record_with_no_events_is_corruption() {
    let line = encode(0, &[]);
    assert!(scan(line.as_bytes(), 0, 0).is_err());
}

#[test]
fn offsets_in_errors_include_the_base() {
    let (bytes, ends) = three();
    let mut bad = bytes[ends[0]..].to_vec();
    bad[0] ^= 0x01;
    assert_eq!(scan(&bad, 1000, 1).unwrap_err().offset, 1000);
}

/// Every record `scan_records` hands over, read through a buffer of
/// `capacity` bytes, and what it returns.
fn scan_chunked(
    bytes: &[u8],
    capacity: usize,
) -> Result<(Vec<Record>, End), ScanError<Infallible>> {
    let mut seen = Vec::new();
    let reader = std::io::BufReader::with_capacity(capacity, bytes);
    let end = scan_records(reader, 0, 0, |record| {
        seen.push(record);
        Ok::<(), Infallible>(())
    })?;
    Ok((seen, end))
}

#[test]
fn small_reads_give_the_same_scan_as_one_read() {
    let (bytes, ends) = three();
    let mut cases = vec![bytes.clone()];
    for cut in 0..bytes.len() {
        cases.push(bytes[..cut].to_vec());
    }
    for i in 0..bytes.len() {
        let mut bad = bytes.clone();
        bad[i] ^= 0x01;
        cases.push(bad);
    }
    let mut junk = bytes[..ends[1]].to_vec();
    junk.extend_from_slice(b"junk\n");
    junk.extend_from_slice(&bytes[ends[1]..]);
    cases.push(junk);
    for case in cases {
        let whole = scan(&case, 0, 0);
        for capacity in [1, 2, 7, 64] {
            match (&whole, scan_chunked(&case, capacity)) {
                (Ok(whole), Ok((seen, end))) => {
                    let records: Vec<Vec<Event>> = seen.iter().map(|r| r.events.clone()).collect();
                    let offsets: Vec<u64> = seen.iter().map(|r| r.offset).collect();
                    assert_eq!(records, whole.records, "{case:?} {capacity}");
                    assert_eq!(offsets, whole.offsets, "{case:?} {capacity}");
                    assert_eq!(end.valid_len as usize, whole.valid_len);
                    assert_eq!(end.torn_len > 0, whole.torn);
                }
                (Err(whole), Err(ScanError::Corrupt(chunked))) => {
                    assert_eq!(&chunked, whole, "{case:?} {capacity}")
                }
                (whole, chunked) => {
                    panic!("{case:?} {capacity}: {whole:?} against {chunked:?}")
                }
            }
        }
    }
}

#[test]
fn the_end_gives_the_last_record_and_the_tail() {
    let (bytes, ends) = three();
    let (_, end) = scan_chunked(&bytes, 64).unwrap();
    let last_len = (ends[2] - ends[1]) as u64;
    assert_eq!(
        end,
        End {
            records: 3,
            valid_len: bytes.len() as u64,
            torn_len: 0,
            last: Some((ends[1] as u64, last_len)),
        }
    );

    let mut torn = bytes.clone();
    torn.extend_from_slice(b"junk\n\n9 zz");
    let (_, end) = scan_chunked(&torn, 3).unwrap();
    assert_eq!(end.torn_len, b"junk\n\n9 zz".len() as u64);
    assert_eq!(end.valid_len, bytes.len() as u64);
    assert_eq!(end.last, Some((ends[1] as u64, last_len)));

    // A torn tail in place of the first record leaves no last record.
    let (seen, end) = scan_chunked(&bytes[..ends[0] - 1], 3).unwrap();
    assert!(seen.is_empty());
    assert_eq!(end.records, 0);
    assert_eq!(end.last, None);
    assert_eq!(end.torn_len, (ends[0] - 1) as u64);
}

#[test]
fn an_error_from_the_caller_stops_the_scan() {
    let (bytes, ends) = three();
    let mut seen = Vec::new();
    let result = scan_records(&bytes[..], 0, 0, |record| {
        seen.push(record.offset);
        if seen.len() == 2 {
            Err("enough")
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(ScanError::Stopped("enough"))));
    assert_eq!(seen, [0, ends[0] as u64]);
}

#[test]
fn records_before_a_corrupt_line_are_handed_over_and_the_scan_fails() {
    let (bytes, ends) = three();
    let mut bad = bytes.clone();
    bad[ends[1] + 2] ^= 0x01;
    bad.extend_from_slice(encode(3, &[note("after")]).as_bytes());
    let mut seen = Vec::new();
    let result = scan_records(&bad[..], 0, 0, |record| {
        seen.push(record.offset);
        Ok::<(), Infallible>(())
    });
    match result {
        Err(ScanError::Corrupt(corrupt)) => assert_eq!(corrupt.offset, ends[1] as u64),
        other => panic!("{other:?}"),
    }
    assert_eq!(seen, [0, ends[0] as u64]);
}

#[test]
fn a_header_reads_back_and_other_starts_are_told_apart() {
    for id in [0, 1, 0xdead_beef, u64::MAX] {
        let line = header(id);
        assert!(line.len() <= MAX_HEADER_LEN);
        assert_eq!(parse_header(line.as_bytes()), Ok((id, line.len())));
        let mut more = line.clone().into_bytes();
        more.extend_from_slice(b"0 00000000 []\n");
        assert_eq!(parse_header(&more), Ok((id, line.len())));
    }
    let unsupported = |text: &str| {
        let line = text.trim_end_matches('\n').to_owned();
        assert_eq!(
            parse_header(text.as_bytes()),
            Err(HeaderError::Unsupported(line)),
            "{text:?}"
        );
    };
    unsupported("dc-mcp journal 2 0000000000000001\n");
    unsupported("dc-mcp journal 1\n");
    unsupported("dc-mcp journal 1 1\n");
    unsupported("dc-mcp journal 1 000000000000000G\n");
    unsupported("dc-mcp journal 1 000000000000000A\n");
    unsupported("dc-mcp journal 1 0000000000000001 extra\n");
    for not in [
        "",
        "dc-mcp journal 1 0000000000000001",
        "junk\n",
        "\n",
        &format!("dc-mcp journal 1 {}\n", "0".repeat(80)),
    ] {
        assert_eq!(
            parse_header(not.as_bytes()),
            Err(HeaderError::NotAJournal),
            "{not:?}"
        );
    }
}
