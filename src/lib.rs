// Index arithmetic over parallel coefficient/limb arrays reads better than
// iterator chains in the ring kernels.
#![allow(clippy::needless_range_loop)]

pub mod bulletin;
pub mod channel;
pub mod codec;
pub mod cs;
#[doc(hidden)]
pub mod hvc_stream;
#[doc(hidden)]
pub mod hvc_sum;
pub mod kahe;
pub mod mse;
pub mod pke;
pub mod prony;
pub mod protocol;
pub mod rings;
pub mod rs;
pub mod scaling_bench;
pub mod share_commitment;
pub mod sig;
pub mod sss;

pub use chipmunk_code::path::Path;
pub use chipmunk_code::{HVCHash, HVCPoly, Tree};
pub use rings::{
    pointwise_dot_cs, pointwise_dot_kahe, pointwise_dot_rs, CsNTTPoly, CsPoly, KaheNTTPoly,
    KahePoly, RsNTTPoly, CS_MODULUS, CS_MODULUS_OVER_TWO, KAHE_MODULUS, KAHE_MODULUS_OVER_TWO, N,
};

/// Draw `n` independent 256-bit seeds from `rng`, serially and in order
pub(crate) fn fork_seeds<R: rand::Rng>(rng: &mut R, n: usize) -> Vec<[u8; 32]> {
    (0..n).map(|_| rng.gen()).collect()
}
