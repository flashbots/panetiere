//! Splice a client-side sweep into a host-measured one.
//!
//! The client runs in a TDX VM, so its timings have to be taken there and joined
//! to a host run of everything else:
//!
//! ```text
//! panetiere_merge <client.csv> <host.csv> <out.csv>
//! ```
//!
//! `--recompute` re-derives the composed/projected columns of one CSV from its
//! measured/derived ones (after a model or net-sim change):
//!
//! ```text
//! panetiere_merge --recompute <in.csv> <out.csv>
//! ```

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag, input, out] if flag == "--recompute" => {
            panetiere::scaling_bench::recompute(input, out)
        }
        [client, host, out] => panetiere::scaling_bench::merge(client, host, out),
        _ => {
            eprintln!("usage: panetiere_merge <client.csv> <host.csv> <out.csv>");
            eprintln!("       panetiere_merge --recompute <in.csv> <out.csv>");
            std::process::exit(2);
        }
    }
}
