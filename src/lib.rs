pub mod bulletin;
pub mod channel;
pub mod codec;
pub mod cs;
pub mod digest;
pub mod kahe;
pub mod mse;
pub mod pke;
pub mod prony;
pub mod protocol;
pub mod rs;
pub mod sig;
pub mod sss;

pub use chipmunk_code::path::Path;
pub use chipmunk_code::{HVCHash, HVCPoly, LinearHash, Polynomial, Tree};

/// Draw `n` independent 256-bit seeds from `rng`, serially and in order, so a
/// caller-seeded round is bit-identical regardless of how the seeds are later
/// consumed by parallel workers (`ChaCha20Rng::from_seed(seed)` per worker).
pub(crate) fn fork_seeds<R: rand::Rng>(rng: &mut R, n: usize) -> Vec<[u8; 32]> {
    (0..n).map(|_| rng.gen()).collect()
}
