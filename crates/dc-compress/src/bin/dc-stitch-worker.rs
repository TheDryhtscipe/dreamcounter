//! Runs Stitch for `dc_compress::compress_isolated`: reads one request on
//! standard input and writes Stitch's steps on standard output. See
//! `dc_compress::worker`.

fn main() -> std::process::ExitCode {
    dc_compress::worker::serve()
}
