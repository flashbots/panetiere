// Index arithmetic over parallel coefficient/limb arrays reads better than
// iterator chains in the ring kernels.
#![allow(clippy::needless_range_loop)]

pub mod bulletin;
pub mod channel;
mod chunking;
pub(crate) mod cs;
#[cfg(test)]
mod e2e;
#[doc(hidden)]
pub mod hvc_stream;
#[doc(hidden)]
pub mod hvc_sum;
pub(crate) mod iblt;
mod iblt_shape;
pub(crate) mod kahe;
pub(crate) mod ng;
pub mod pke;
pub mod protocol;
#[cfg(test)]
mod reference_params;
pub(crate) mod rings;
pub(crate) mod sss;

pub use rings::{KahePoly, N};

pub(crate) use rings::{
    pointwise_dot_cs, pointwise_dot_kahe, CsNTTPoly, CsPoly, KaheNTTPoly, CS_MODULUS,
    CS_MODULUS_OVER_TWO, KAHE_MODULUS, KAHE_MODULUS_OVER_TWO,
};

/// Draw `n` independent 256-bit seeds from `rng`, serially and in order
pub(crate) fn fork_seeds<R: rand::Rng>(rng: &mut R, n: usize) -> Vec<[u8; 32]> {
    (0..n).map(|_| rng.gen()).collect()
}
