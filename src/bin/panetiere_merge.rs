//! Splice a client-side sweep into a host-measured one.
//!
//! The client runs in a TDX VM, so its timings have to be taken there and joined
//! to a host run of everything else:
//!
//! ```text
//! panetiere_merge <client.csv> <host.csv> <out.csv>
//! ```

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [client, host, out] = args.as_slice() else {
        eprintln!("usage: panetiere_merge <client.csv> <host.csv> <out.csv>");
        std::process::exit(2);
    };
    panetiere::scaling_bench::merge(client, host, out);
}
