use chipmunk_code::{HVCPoly, ALPHA_H, HVC_WIDTH, TWO_ZETA_PLUS_ONE};
use negacyclic_rings::arithmetic;
use negacyclic_rings::decomposition;
use negacyclic_rings::params::{find_psi32, generate_ring32};
use negacyclic_rings::{ntt32, Residues, Rns};
use rand::Rng;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub const N: usize = chipmunk_code::N;
pub const CS_MODULUS: i32 = 139_301;
pub const CS_MODULUS_OVER_TWO: i32 = (CS_MODULUS - 1) / 2;

pub const KAHE_MODULUS: i64 = 280_513_608_622_081;
pub const KAHE_MODULUS_OVER_TWO: i64 = (KAHE_MODULUS - 1) / 2;

pub(crate) const KAHE_RNS_MODULI: [u32; 2] = [16_760_833, 16_736_257];
const CS_AUX_RNS_MODULI: [u32; 2] = [1_073_692_673, 1_073_668_097];

fn kahe_rns() -> &'static Rns<N, 2> {
    static RING: OnceLock<Box<Rns<N, 2>>> = OnceLock::new();
    RING.get_or_init(|| {
        std::thread::Builder::new()
            .name("kahe-rns-init".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                Box::new(Rns::new(
                    KAHE_RNS_MODULI.map(|q| generate_ring32(q, find_psi32::<N>(q))),
                ))
            })
            .expect("spawn KAHE RNS initialization")
            .join()
            .expect("KAHE RNS initialization panicked")
    })
}

fn cs_aux_rns() -> &'static Rns<N, 2> {
    static RING: OnceLock<Box<Rns<N, 2>>> = OnceLock::new();
    RING.get_or_init(|| {
        std::thread::Builder::new()
            .name("cs-rns-init".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                Box::new(Rns::new(
                    CS_AUX_RNS_MODULI.map(|q| generate_ring32(q, find_psi32::<N>(q))),
                ))
            })
            .expect("spawn CS RNS initialization")
            .join()
            .expect("CS RNS initialization panicked")
    })
}

fn boxed_residues() -> Box<Residues<N, 2>> {
    let mut value = Box::<Residues<N, 2>>::new_uninit();
    unsafe {
        value.as_mut_ptr().write_bytes(0, 1);
        value.assume_init()
    }
}

pub(crate) fn boxed_i64_coeffs() -> Box<[i64; N]> {
    let mut value = Box::<[i64; N]>::new_uninit();
    unsafe {
        value.as_mut_ptr().write_bytes(0, 1);
        value.assume_init()
    }
}

#[inline]
fn lift64(a: i64, modulus: i64) -> i64 {
    a.rem_euclid(modulus)
}

#[inline]
pub(crate) fn center_canonical_i64(value: i64, modulus: i64) -> i64 {
    debug_assert!((0..modulus).contains(&value));
    value - (modulus & -((value > modulus / 2) as i64))
}

#[inline]
pub(crate) fn center_i64(value: i64, modulus: i64) -> i64 {
    center_canonical_i64(value.rem_euclid(modulus), modulus)
}

#[inline]
pub(crate) fn center_i32(value: i32, modulus: i32) -> i32 {
    let reduced = value.rem_euclid(modulus);
    reduced - (modulus & -((reduced > modulus / 2) as i32))
}

#[inline]
pub(crate) fn center_half_open_i64(value: i64, modulus: i64) -> i64 {
    center_canonical_half_open_i64(value.rem_euclid(modulus), modulus)
}

#[inline]
pub(crate) fn center_canonical_half_open_i64(value: i64, modulus: i64) -> i64 {
    debug_assert!((0..modulus).contains(&value));
    value - (modulus & -((value >= modulus / 2) as i64))
}

#[inline]
fn normalize64(value: i64, modulus: i64) -> i64 {
    center_i64(value, modulus)
}

#[cfg(test)]
#[inline]
fn reduce_i64_rns2<const Q0: u32, const Q1: u32>(value: i64) -> [u32; 2] {
    #[inline]
    fn reduce<const Q: u32>(value: i64) -> u32 {
        let remainder = value % Q as i64;
        (remainder + ((remainder >> 63) & Q as i64)) as u32
    }
    [reduce::<Q0>(value), reduce::<Q1>(value)]
}

#[cfg(test)]
#[inline]
fn reduce_small_i32_rns2<const Q0: u32, const Q1: u32>(value: i32) -> [u32; 2] {
    debug_assert!(value.unsigned_abs() < Q0);
    debug_assert!(value.unsigned_abs() < Q1);
    let negative = 0u32.wrapping_sub((value < 0) as u32);
    let positive = value as u32;
    let magnitude = value.unsigned_abs();
    [
        (positive & !negative) | ((Q0 - magnitude) & negative),
        (positive & !negative) | ((Q1 - magnitude) & negative),
    ]
}

#[cfg(test)]
#[inline]
fn lift_centered_rns2<const Q0: u32, const Q1: u32>(
    prefix_inverse: u32,
    residues: [u32; 2],
) -> i64 {
    const { assert!(Q0 < 2 * Q1) };
    let reduce_r0 = 0u32.wrapping_sub((residues[0] >= Q1) as u32);
    let r0_mod_q1 = residues[0] - (Q1 & reduce_r0);
    let add_q1 = 0u32.wrapping_sub((residues[1] < r0_mod_q1) as u32);
    let delta = residues[1]
        .wrapping_sub(r0_mod_q1)
        .wrapping_add(Q1 & add_q1);
    let digit = (delta as u64 * prefix_inverse as u64) % Q1 as u64;
    let value = residues[0] as u64 + Q0 as u64 * digit;
    let product = Q0 as u64 * Q1 as u64;
    value as i64 - (value > product / 2) as i64 * product as i64
}

#[derive(Debug, Clone, Copy)]
pub struct KahePoly {
    coeffs: [i64; N],
}

impl Default for KahePoly {
    fn default() -> Self {
        Self { coeffs: [0; N] }
    }
}

impl PartialEq for KahePoly {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs
            .iter()
            .zip(&other.coeffs)
            .all(|(&a, &b)| lift64(a, KAHE_MODULUS) == lift64(b, KAHE_MODULUS))
    }
}

impl Eq for KahePoly {}

impl std::ops::Add for KahePoly {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl std::ops::AddAssign for KahePoly {
    fn add_assign(&mut self, other: Self) {
        for (a, b) in self.coeffs.iter_mut().zip(other.coeffs) {
            *a = (*a + b) % KAHE_MODULUS;
        }
    }
}

impl std::ops::Sub for KahePoly {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        self -= other;
        self
    }
}

impl std::ops::SubAssign for KahePoly {
    fn sub_assign(&mut self, other: Self) {
        for (a, b) in self.coeffs.iter_mut().zip(other.coeffs) {
            *a = (*a - b) % KAHE_MODULUS;
        }
    }
}

impl std::ops::Mul for KahePoly {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        KahePoly::from(KaheNTTPoly::from(self) * KaheNTTPoly::from(other))
    }
}

impl KahePoly {
    pub fn from_coeffs(coeffs: [i64; N]) -> Self {
        Self { coeffs }
    }

    pub fn from_signed_coeffs(coeffs: &[i64; N]) -> Self {
        debug_assert!(coeffs
            .iter()
            .all(|&c| (-KAHE_MODULUS_OVER_TWO..=KAHE_MODULUS_OVER_TWO).contains(&c)));
        Self { coeffs: *coeffs }
    }

    pub fn coeffs(&self) -> &[i64; N] {
        &self.coeffs
    }

    pub fn rand_poly<R: Rng>(rng: &mut R) -> Self {
        let q = KAHE_MODULUS as u64;
        let threshold = (((1u128 << 64) / q as u128) * q as u128) as u64;
        let coeffs = core::array::from_fn(|_| {
            let mut value = rng.next_u64();
            while value >= threshold {
                value = rng.next_u64();
            }
            (value % q) as i64 - KAHE_MODULUS_OVER_TWO
        });
        Self { coeffs }
    }

    pub fn schoolbook(a: &Self, b: &Self) -> Self {
        let q = KAHE_MODULUS as i128;
        let mut buf = vec![0i128; N * 2];
        for i in 0..N {
            for j in 0..N {
                buf[i + j] = (buf[i + j] + a.coeffs[i] as i128 * b.coeffs[j] as i128) % q;
            }
        }
        Self {
            coeffs: core::array::from_fn(|i| (buf[i] - buf[i + N]).rem_euclid(q) as i64),
        }
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for &coefficient in &self.coeffs {
            hasher.update(lift64(coefficient, KAHE_MODULUS).to_le_bytes());
        }
        hasher.finalize().into()
    }

    pub fn infinity_norm(&self) -> u64 {
        self.coeffs
            .iter()
            .map(|x| x.unsigned_abs())
            .max()
            .unwrap_or(0)
    }

    pub fn lift(&mut self) {
        for coefficient in &mut self.coeffs {
            *coefficient = lift64(*coefficient, KAHE_MODULUS);
        }
    }

    pub fn lift_inplace(&mut self) {
        self.lift();
    }

    pub fn normalize(&mut self) {
        for coefficient in &mut self.coeffs {
            *coefficient = normalize64(*coefficient, KAHE_MODULUS);
        }
    }
}

type KaheNTTCoeffs = Residues<N, 2>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KaheNTTPoly {
    coeffs: KaheNTTCoeffs,
}

impl Default for KaheNTTPoly {
    fn default() -> Self {
        Self {
            coeffs: [[0u32; N]; 2],
        }
    }
}

impl From<&KahePoly> for KaheNTTPoly {
    fn from(poly: &KahePoly) -> Self {
        let ring = kahe_rns();
        let mut coeffs = boxed_residues();
        ring.reduce_i64_into(&poly.coeffs, &mut coeffs);
        ring.forward(&mut coeffs);
        Self { coeffs: *coeffs }
    }
}

impl From<KahePoly> for KaheNTTPoly {
    fn from(poly: KahePoly) -> Self {
        Self::from(&poly)
    }
}

impl From<&KaheNTTPoly> for KahePoly {
    fn from(poly: &KaheNTTPoly) -> Self {
        let ring = kahe_rns();
        let mut residues = boxed_residues();
        residues[0].copy_from_slice(&poly.coeffs[0]);
        residues[1].copy_from_slice(&poly.coeffs[1]);
        ring.inverse(&mut residues);
        let mut output = Self::default();
        ring.lift_centered_i64_into(&residues, &mut output.coeffs);
        output
    }
}

impl From<KaheNTTPoly> for KahePoly {
    fn from(poly: KaheNTTPoly) -> Self {
        Self::from(&poly)
    }
}

impl std::ops::Add for KaheNTTPoly {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl std::ops::AddAssign for KaheNTTPoly {
    fn add_assign(&mut self, other: Self) {
        kahe_rns().add_assign(&mut self.coeffs, &other.coeffs);
    }
}

impl std::ops::Sub for KaheNTTPoly {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        kahe_rns().sub_assign(&mut self.coeffs, &other.coeffs);
        self
    }
}

impl std::ops::Mul for KaheNTTPoly {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let coeffs = kahe_rns().pointwise_mul(&self.coeffs, &other.coeffs);
        Self { coeffs }
    }
}

impl KaheNTTPoly {
    pub fn rand_ntt_poly<R: Rng>(rng: &mut R) -> Self {
        let mut result = Self::default();
        result.fill_random(rng);
        result
    }

    pub(crate) fn fill_random<R: Rng>(&mut self, rng: &mut R) {
        let ring = kahe_rns();
        for limb in 0..2 {
            for coefficient in &mut self.coeffs[limb] {
                let mut value = rng.next_u32();
                while value >= ring.sample_threshold[limb] {
                    value = rng.next_u32();
                }
                *coefficient = value % ring.ch[limb].q;
            }
        }
    }

    pub(crate) fn residues(&self) -> &Residues<N, 2> {
        &self.coeffs
    }

    pub fn ntt_slots(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        (0..N).map(|i| kahe_rns().lift_coeff([self.coeffs[0][i], self.coeffs[1][i]]) as u64)
    }

    pub fn from_ntt_slots(slots: &[u64]) -> Option<Self> {
        if slots.len() != N || slots.iter().any(|&x| x >= KAHE_MODULUS as u64) {
            return None;
        }
        let mut coeffs = boxed_residues();
        for (i, &value) in slots.iter().enumerate() {
            coeffs[0][i] = (value % KAHE_RNS_MODULI[0] as u64) as u32;
            coeffs[1][i] = (value % KAHE_RNS_MODULI[1] as u64) as u32;
        }
        Some(Self { coeffs: *coeffs })
    }

    pub(crate) fn from_residues(coeffs: Residues<N, 2>) -> Option<Self> {
        coeffs
            .iter()
            .zip(KAHE_RNS_MODULI)
            .all(|(channel, q)| channel.iter().all(|&x| x < q))
            .then_some(Self { coeffs })
    }

    pub fn from_kahe(poly: &KahePoly) -> Self {
        Self::from(poly)
    }

    pub(crate) fn set_from_coeffs(&mut self, coeffs: &[i64; N]) {
        let ring = kahe_rns();
        ring.reduce_i64_into(coeffs, &mut self.coeffs);
        ring.forward(&mut self.coeffs);
    }

    pub(crate) fn add_product_assign(&mut self, lhs: &Self, rhs: &Self) {
        let ring = kahe_rns();
        for limb in 0..2 {
            for i in 0..N {
                let product =
                    ntt32::mul_mod(lhs.coeffs[limb][i], rhs.coeffs[limb][i], &ring.ch[limb]);
                self.coeffs[limb][i] =
                    ntt32::add_mod(self.coeffs[limb][i], product, ring.ch[limb].q);
            }
        }
    }

    pub fn to_centered_coeffs(&self) -> [i64; N] {
        KahePoly::from(self).coeffs
    }
}

pub fn pointwise_dot_kahe(a: &[KaheNTTPoly], b: &[KaheNTTPoly]) -> KaheNTTPoly {
    debug_assert_eq!(a.len(), b.len());
    let mut result = KaheNTTPoly::default();
    let a: Vec<_> = a.iter().map(|poly| &poly.coeffs).collect();
    let b: Vec<_> = b.iter().map(|poly| &poly.coeffs).collect();
    kahe_rns().pointwise_mac(&mut result.coeffs, &a, &b);
    result
}

#[derive(Debug, Clone, Copy)]
pub struct CsPoly {
    coeffs: [i32; N],
}

impl Default for CsPoly {
    fn default() -> Self {
        Self { coeffs: [0; N] }
    }
}

impl PartialEq for CsPoly {
    fn eq(&self, other: &Self) -> bool {
        arithmetic::eq_i32(&self.coeffs, &other.coeffs, CS_MODULUS)
    }
}

impl Eq for CsPoly {}

impl std::fmt::Display for CsPoly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        arithmetic::fmt_i32(&self.coeffs, f)
    }
}

impl std::ops::Add for CsPoly {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl std::ops::AddAssign for CsPoly {
    fn add_assign(&mut self, other: Self) {
        arithmetic::add_assign_i32(&mut self.coeffs, &other.coeffs, CS_MODULUS);
    }
}

impl std::ops::Sub for CsPoly {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        self -= other;
        self
    }
}

impl std::ops::SubAssign for CsPoly {
    fn sub_assign(&mut self, other: Self) {
        arithmetic::sub_assign_i32(&mut self.coeffs, &other.coeffs, CS_MODULUS);
    }
}

impl std::ops::Mul for CsPoly {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        CsPoly::from(CsNTTPoly::from(self) * CsNTTPoly::from(other))
    }
}

impl CsPoly {
    pub fn from_coeffs(coeffs: [i32; N]) -> Self {
        Self { coeffs }
    }

    pub fn from_signed_coeffs(coeffs: &[i32; N]) -> Self {
        debug_assert!(coeffs
            .iter()
            .all(|&c| (-CS_MODULUS_OVER_TWO..=CS_MODULUS_OVER_TWO).contains(&c)));
        Self { coeffs: *coeffs }
    }

    pub fn coeffs(&self) -> &[i32; N] {
        &self.coeffs
    }

    pub fn schoolbook(a: &Self, b: &Self) -> Self {
        Self::from_coeffs(arithmetic::schoolbook_i32(&a.coeffs, &b.coeffs, CS_MODULUS))
    }

    pub fn rand_poly<R: Rng>(rng: &mut R) -> Self {
        let threshold = (((1u64 << 32) / CS_MODULUS as u64) * CS_MODULUS as u64) as u32;
        Self::from_coeffs(arithmetic::rand_poly_i32(
            rng,
            CS_MODULUS,
            CS_MODULUS_OVER_TWO,
            threshold,
        ))
    }

    pub fn rand_balanced_ternary<R: Rng>(rng: &mut R, half_weight: usize) -> Self {
        Self::from_coeffs(arithmetic::rand_balanced_ternary_i32(rng, half_weight))
    }

    pub fn rand_binary<R: Rng>(rng: &mut R) -> Self {
        Self::from_coeffs(arithmetic::rand_binary_i32(rng))
    }

    pub fn rand_ternary<R: Rng>(rng: &mut R, weight: usize) -> Self {
        Self::from_coeffs(arithmetic::rand_ternary_i32(rng, weight))
    }

    pub fn rand_mod_p<R: Rng>(rng: &mut R, p: u32) -> Self {
        Self::from_coeffs(arithmetic::rand_mod_p_i32(rng, p))
    }

    pub fn from_hash_message(msg: &[u8]) -> Self {
        Self::from_coeffs(arithmetic::from_hash_message_i32(msg, ALPHA_H))
    }

    pub fn digest(&self) -> [u8; 32] {
        arithmetic::digest_i32(&self.coeffs)
    }

    pub fn is_ternary(&self) -> bool {
        arithmetic::is_ternary_i32(&self.coeffs)
    }

    pub fn infinity_norm(&self) -> u32 {
        arithmetic::infinity_norm_i32(&self.coeffs)
    }

    pub fn lift(&mut self) {
        arithmetic::lift_assign_i32(&mut self.coeffs, CS_MODULUS);
    }

    pub fn lift_inplace(&mut self) {
        self.lift();
    }

    pub fn normalize(&mut self) {
        arithmetic::normalize_assign_i32(&mut self.coeffs, CS_MODULUS);
    }

    pub fn decompose_r_to_hvc(&self) -> [HVCPoly; HVC_WIDTH] {
        decomposition::decompose_i32::<N, HVC_WIDTH>(
            &self.coeffs,
            Some(CS_MODULUS),
            TWO_ZETA_PLUS_ONE as i32,
            true,
        )
        .map(HVCPoly::from_coeffs)
    }

    pub fn project_r_from_hvc(decomposed: &[HVCPoly]) -> Self {
        assert_eq!(decomposed.len(), HVC_WIDTH);
        let digits: [[i32; N]; HVC_WIDTH] = core::array::from_fn(|i| *decomposed[i].coeffs());
        Self::from_coeffs(decomposition::recompose_i32(
            &digits,
            TWO_ZETA_PLUS_ONE as i32,
            None,
            Some(CS_MODULUS),
        ))
    }
}

type CsNTTCoeffs = Residues<N, 2>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsNTTPoly {
    coeffs: CsNTTCoeffs,
}

impl Default for CsNTTPoly {
    fn default() -> Self {
        Self {
            coeffs: [[0u32; N]; 2],
        }
    }
}

impl From<&CsPoly> for CsNTTPoly {
    fn from(poly: &CsPoly) -> Self {
        let ring = cs_aux_rns();
        let mut coeffs = boxed_residues();
        let mut centered = poly.coeffs;
        arithmetic::normalize_assign_i32(&mut centered, CS_MODULUS);
        ring.reduce_centered_i32_into(&centered, &mut coeffs);
        ring.forward(&mut coeffs);
        Self { coeffs: *coeffs }
    }
}

impl From<CsPoly> for CsNTTPoly {
    fn from(poly: CsPoly) -> Self {
        Self::from(&poly)
    }
}

impl From<&CsNTTPoly> for CsPoly {
    fn from(poly: &CsNTTPoly) -> Self {
        let exact: [i64; N] = {
            let ring = cs_aux_rns();
            let mut coeffs = boxed_residues();
            coeffs[0].copy_from_slice(&poly.coeffs[0]);
            coeffs[1].copy_from_slice(&poly.coeffs[1]);
            ring.inverse(&mut coeffs);
            let mut lifted = boxed_i64_coeffs();
            ring.lift_centered_i64_into(&coeffs, &mut lifted);
            *lifted
        };
        let mut output = Self::default();
        for (coefficient, value) in output.coeffs.iter_mut().zip(exact) {
            *coefficient =
                arithmetic::normalize_i32(value.rem_euclid(CS_MODULUS as i64) as i32, CS_MODULUS);
        }
        output
    }
}

impl From<CsNTTPoly> for CsPoly {
    fn from(poly: CsNTTPoly) -> Self {
        Self::from(&poly)
    }
}

impl std::ops::Add for CsNTTPoly {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl std::ops::AddAssign for CsNTTPoly {
    fn add_assign(&mut self, other: Self) {
        cs_aux_rns().add_assign(&mut self.coeffs, &other.coeffs);
    }
}

impl std::ops::Sub for CsNTTPoly {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        cs_aux_rns().sub_assign(&mut self.coeffs, &other.coeffs);
        self
    }
}

impl std::ops::Mul for CsNTTPoly {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let coeffs = cs_aux_rns().pointwise_mul(&self.coeffs, &other.coeffs);
        Self { coeffs }
    }
}

pub fn pointwise_dot_cs(a: &[CsNTTPoly], b: &[CsNTTPoly]) -> CsNTTPoly {
    debug_assert_eq!(a.len(), b.len());
    let mut result = CsNTTPoly::default();
    let a: Vec<_> = a.iter().map(|poly| &poly.coeffs).collect();
    let b: Vec<_> = b.iter().map(|poly| &poly.coeffs).collect();
    cs_aux_rns().pointwise_mac(&mut result.coeffs, &a, &b);
    result
}
pub type RsNTTPoly = KaheNTTPoly;

pub fn pointwise_dot_rs(a: &[RsNTTPoly], b: &[RsNTTPoly]) -> RsNTTPoly {
    pointwise_dot_kahe(a, b)
}

#[cfg(test)]
mod rns_conversion_tests {
    use super::*;

    #[test]
    fn specialized_kahe_conversion_matches_generic_rns() {
        let ring = kahe_rns();
        let values = [
            -KAHE_MODULUS / 2,
            -1,
            0,
            1,
            KAHE_MODULUS / 2,
            KAHE_MODULUS - 1,
        ];
        for value in values {
            let residues = reduce_i64_rns2::<{ KAHE_RNS_MODULI[0] }, { KAHE_RNS_MODULI[1] }>(value);
            assert_eq!(residues, ring.reduce_coeff(value as i128));
            assert_eq!(
                lift_centered_rns2::<{ KAHE_RNS_MODULI[0] }, { KAHE_RNS_MODULI[1] }>(
                    ring.prefix_inverses[1],
                    residues
                ) as i128,
                ring.lift_centered(residues)
            );
        }
    }

    #[test]
    fn specialized_cs_conversion_matches_generic_rns() {
        let ring = cs_aux_rns();
        for value in [-CS_MODULUS_OVER_TWO, -1, 0, 1, CS_MODULUS_OVER_TWO] {
            let residues =
                reduce_small_i32_rns2::<{ CS_AUX_RNS_MODULI[0] }, { CS_AUX_RNS_MODULI[1] }>(value);
            assert_eq!(residues, ring.reduce_coeff(value as i128));
            assert_eq!(
                lift_centered_rns2::<{ CS_AUX_RNS_MODULI[0] }, { CS_AUX_RNS_MODULI[1] }>(
                    ring.prefix_inverses[1],
                    residues
                ) as i128,
                ring.lift_centered(residues)
            );
        }
    }

    #[test]
    fn array_conversions_match_generic_rns() {
        let kahe = kahe_rns();
        let kahe_values = [
            -KAHE_MODULUS / 2,
            -KAHE_MODULUS / 3,
            -1,
            0,
            1,
            KAHE_MODULUS / 3,
            KAHE_MODULUS / 2,
            KAHE_MODULUS - 1,
        ];
        let input = core::array::from_fn(|i| kahe_values[i % kahe_values.len()]);
        let mut residues = [[0u32; N]; 2];
        kahe.reduce_i64_into(&input, &mut residues);
        let mut lifted = [0i64; N];
        kahe.lift_centered_i64_into(&residues, &mut lifted);
        for i in 0..N {
            let expected_residues = kahe.reduce_coeff(input[i] as i128);
            assert_eq!([residues[0][i], residues[1][i]], expected_residues);
            assert_eq!(lifted[i] as i128, kahe.lift_centered(expected_residues));
        }

        let cs = cs_aux_rns();
        let cs_values = [-CS_MODULUS_OVER_TWO, -1, 0, 1, CS_MODULUS_OVER_TWO];
        let input = core::array::from_fn(|i| cs_values[i % cs_values.len()]);
        let mut residues = [[0u32; N]; 2];
        cs.reduce_centered_i32_into(&input, &mut residues);
        let mut lifted = [0i64; N];
        cs.lift_centered_i64_into(&residues, &mut lifted);
        for i in 0..N {
            let expected_residues = cs.reduce_coeff(input[i] as i128);
            assert_eq!([residues[0][i], residues[1][i]], expected_residues);
            assert_eq!(lifted[i] as i128, cs.lift_centered(expected_residues));
        }
    }
}
