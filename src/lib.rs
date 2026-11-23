// Index arithmetic over parallel coefficient/limb arrays reads better than
// iterator chains in the ring kernels.
#![allow(clippy::needless_range_loop)]

pub mod bulletin;
pub mod channel;
mod chunking;
pub mod cs;
#[doc(hidden)]
pub mod hvc_stream;
#[doc(hidden)]
pub mod hvc_sum;
mod iblt_shape;
pub mod kahe;
pub mod mse;
pub mod pke;
pub mod prony;
pub mod protocol;
#[cfg(test)]
mod reference_params;
pub mod rings;
pub mod sss;

pub use chipmunk_code::path::Path;
pub use chipmunk_code::{HVCHash, HVCPoly, Tree};
pub use rings::{
    pointwise_dot_cs, pointwise_dot_kahe, CsNTTPoly, CsPoly, KaheNTTPoly, KahePoly, CS_MODULUS,
    CS_MODULUS_OVER_TWO, KAHE_MODULUS, KAHE_MODULUS_OVER_TWO, N,
};

/// Draw `n` independent 256-bit seeds from `rng`, serially and in order
pub(crate) fn fork_seeds<R: rand::Rng>(rng: &mut R, n: usize) -> Vec<[u8; 32]> {
    (0..n).map(|_| rng.gen()).collect()
}
