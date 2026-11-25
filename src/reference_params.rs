//! checks the crate's constants against the paper's reference parameter set,
//! `scripts/params/reference.json`, which ci checks against every condition
//! of the paper's theorems.

use chipmunk_code::{HVC_MODULUS, HVC_WIDTH, ZETA};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde_json::Value;

use crate::cs::{beta_agg_hvc, HidingMerkleCommitment, BETA_CS, KAPPA_CS, MU_CS, R_BOUND};
use crate::kahe::{SIGMA_E_DEFAULT, SIGMA_S_DEFAULT, T_MODULUS_DEFAULT};
use crate::mse::{BITS_PER_SYMBOL, K_LIMBS};
use crate::protocol::ProtocolParams;
use crate::rings::{CS_MODULUS, KAHE_MODULUS, MU_MAX, RHO_MAX};
use crate::N;

pub(crate) fn reference() -> Value {
    serde_json::from_str(include_str!("../scripts/params/reference.json"))
        .expect("reference parameter set is valid json")
}

/// the integer at `pointer` (e.g. `/kahe/t`) of the reference set.
pub(crate) fn int(reference: &Value, pointer: &str) -> u64 {
    reference
        .pointer(pointer)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("reference parameter set has an integer at {pointer}"))
}

#[test]
fn constants_match_reference() {
    let r = reference();

    assert_eq!(int(&r, "/setting/rho"), u64::from(RHO_MAX));
    assert_eq!(int(&r, "/setting/n"), N as u64);

    assert_eq!(int(&r, "/kahe/t"), T_MODULUS_DEFAULT);
    assert_eq!(int(&r, "/kahe/q"), KAHE_MODULUS as u64);
    assert_eq!(int(&r, "/kahe/mu"), MU_MAX as u64);
    let sigma = r.pointer("/kahe/sigma").and_then(Value::as_f64);
    assert_eq!(sigma, Some(SIGMA_S_DEFAULT));
    assert_eq!(sigma, Some(SIGMA_E_DEFAULT));
    assert_eq!(int(&r, "/kahe/mse/rho"), u64::from(RHO_MAX));
    assert_eq!(int(&r, "/kahe/mse/chunk_bits"), BITS_PER_SYMBOL as u64);
    // the IBLT key takes ω symbols; the sketch's keys are still single
    // symbols (issue #27)
    assert_eq!(int(&r, "/kahe/mse/omega"), K_LIMBS as u64);

    assert_eq!(int(&r, "/leaf/p"), CS_MODULUS as u64);
    assert_eq!(int(&r, "/leaf/beta"), u64::from(BETA_CS));
    assert_eq!(int(&r, "/leaf/beta_agg"), u64::from(R_BOUND));
    assert_eq!(int(&r, "/leaf/kappa"), KAPPA_CS as u64);
    assert_eq!(int(&r, "/leaf/mu"), MU_CS as u64);

    assert_eq!(int(&r, "/chipmunk/q"), HVC_MODULUS as u64);
    assert_eq!(int(&r, "/chipmunk/eta"), u64::from(ZETA));
    // the code decomposes both chipmunk and leaf elements into `HVC_WIDTH`
    // digits
    assert_eq!(int(&r, "/chipmunk/kappa"), HVC_WIDTH as u64);
    assert_eq!(int(&r, "/chipmunk/kappa_prime"), HVC_WIDTH as u64);
    assert_eq!(int(&r, "/chipmunk/beta_agg"), u64::from(beta_agg_hvc()));

    let gamma = int(&r, "/setting/gamma") as usize;
    let cs = HidingMerkleCommitment::setup_with_dims(
        &mut ChaCha20Rng::from_seed([0; 32]),
        gamma,
        MU_CS,
        KAPPA_CS,
    );
    assert_eq!(int(&r, "/chipmunk/xi"), cs.xi() as u64);
    assert_eq!(int(&r, "/chipmunk/tau"), cs.total_path_len() as u64);
}

/// The protocol parameters report the reference set's `t`, `q` and `σ`.
#[test]
fn protocol_params_report_the_reference_values() {
    let r = reference();
    let pp = ProtocolParams::setup(&mut ChaCha20Rng::from_seed([0; 32]), 8);
    assert_eq!(pp.plaintext_modulus(), int(&r, "/kahe/t"));
    assert_eq!(pp.ciphertext_modulus(), int(&r, "/kahe/q"));
    let sigma = r.pointer("/kahe/sigma").and_then(Value::as_f64);
    assert_eq!(Some(pp.key_sigma()), sigma);
    assert_eq!(Some(pp.error_sigma()), sigma);
}
