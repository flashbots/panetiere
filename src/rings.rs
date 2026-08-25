use chipmunk_code::{HVCPoly, ALPHA_H, HVC_WIDTH, TWO_ZETA_PLUS_ONE};
use negacyclic_rings::arithmetic;
use negacyclic_rings::decomposition;
use negacyclic_rings::ntt64::{self, Ring64};
#[cfg(feature = "rns")]
use negacyclic_rings::params::{find_psi32, generate_ring32};
use negacyclic_rings::params::{find_psi64, generate_ring64};
#[cfg(feature = "rns")]
use negacyclic_rings::{Residues, Rns};
use rand::Rng;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub const N: usize = chipmunk_code::N;
pub const CS_MODULUS: i32 = 139_301;
pub const CS_MODULUS_OVER_TWO: i32 = (CS_MODULUS - 1) / 2;

#[cfg(not(feature = "rns"))]
pub const KAHE_MODULUS: i64 = 347_280_875_347_969;
#[cfg(feature = "rns")]
pub const KAHE_MODULUS: i64 = 280_513_608_622_081;
pub const KAHE_MODULUS_OVER_TWO: i64 = (KAHE_MODULUS - 1) / 2;

pub const DGT_MODULUS: u64 = 2_305_843_009_213_616_129;

#[cfg(not(feature = "rns"))]
const CS_AUX_MODULUS: u64 = 288_230_376_151_748_609;
#[cfg(feature = "rns")]
const KAHE_RNS_MODULI: [u32; 2] = [16_760_833, 16_736_257];
#[cfg(feature = "rns")]
const CS_AUX_RNS_MODULI: [u32; 2] = [1_073_692_673, 1_073_668_097];

fn ring64(cell: &'static OnceLock<Ring64<N>>, modulus: u64) -> &'static Ring64<N> {
    cell.get_or_init(|| generate_ring64(modulus, find_psi64::<N>(modulus)))
}

#[cfg(not(feature = "rns"))]
fn kahe_ring64() -> &'static Ring64<N> {
    static RING: OnceLock<Ring64<N>> = OnceLock::new();
    ring64(&RING, KAHE_MODULUS as u64)
}

#[cfg(not(feature = "rns"))]
fn cs_aux_ring64() -> &'static Ring64<N> {
    static RING: OnceLock<Ring64<N>> = OnceLock::new();
    ring64(&RING, CS_AUX_MODULUS)
}

fn dgt_ring64() -> &'static Ring64<N> {
    static RING: OnceLock<Ring64<N>> = OnceLock::new();
    ring64(&RING, DGT_MODULUS)
}

#[cfg(feature = "rns")]
fn kahe_rns() -> &'static Rns<N, 2> {
    static RING: OnceLock<Rns<N, 2>> = OnceLock::new();
    RING.get_or_init(|| Rns::new(KAHE_RNS_MODULI.map(|q| generate_ring32(q, find_psi32::<N>(q)))))
}

#[cfg(feature = "rns")]
fn cs_aux_rns() -> &'static Rns<N, 2> {
    static RING: OnceLock<Rns<N, 2>> = OnceLock::new();
    RING.get_or_init(|| Rns::new(CS_AUX_RNS_MODULI.map(|q| generate_ring32(q, find_psi32::<N>(q)))))
}

#[inline]
fn lift64(a: i64, modulus: i64) -> i64 {
    a.rem_euclid(modulus)
}

#[inline]
fn normalize64(mut a: i64, modulus: i64) -> i64 {
    a %= modulus;
    if a > modulus / 2 {
        a -= modulus;
    }
    if a < -modulus / 2 {
        a += modulus;
    }
    a
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

#[cfg(not(feature = "rns"))]
type KaheNTTCoeffs = [u64; N];
#[cfg(feature = "rns")]
type KaheNTTCoeffs = Residues<N, 2>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KaheNTTPoly {
    coeffs: KaheNTTCoeffs,
}

impl Default for KaheNTTPoly {
    fn default() -> Self {
        #[cfg(not(feature = "rns"))]
        let coeffs = [0u64; N];
        #[cfg(feature = "rns")]
        let coeffs = [[0u32; N]; 2];
        Self { coeffs }
    }
}

impl From<&KahePoly> for KaheNTTPoly {
    fn from(poly: &KahePoly) -> Self {
        #[cfg(not(feature = "rns"))]
        {
            let ring = kahe_ring64();
            let mut coeffs = poly.coeffs.map(|x| lift64(x, KAHE_MODULUS) as u64);
            ntt64::ntt(ring, &mut coeffs);
            Self { coeffs }
        }
        #[cfg(feature = "rns")]
        {
            let ring = kahe_rns();
            let mut coeffs = [[0u32; N]; 2];
            for (i, &x) in poly.coeffs.iter().enumerate() {
                let residues = ring.reduce_coeff(x as i128);
                coeffs[0][i] = residues[0];
                coeffs[1][i] = residues[1];
            }
            ring.forward(&mut coeffs);
            Self { coeffs }
        }
    }
}

impl From<KahePoly> for KaheNTTPoly {
    fn from(poly: KahePoly) -> Self {
        Self::from(&poly)
    }
}

impl From<&KaheNTTPoly> for KahePoly {
    fn from(poly: &KaheNTTPoly) -> Self {
        #[cfg(not(feature = "rns"))]
        {
            let mut coeffs = poly.coeffs;
            ntt64::inv_ntt(kahe_ring64(), &mut coeffs);
            Self {
                coeffs: coeffs.map(|x| normalize64(x as i64, KAHE_MODULUS)),
            }
        }
        #[cfg(feature = "rns")]
        {
            let ring = kahe_rns();
            let mut residues = poly.coeffs;
            ring.inverse(&mut residues);
            Self {
                coeffs: core::array::from_fn(|i| {
                    ring.lift_centered([residues[0][i], residues[1][i]]) as i64
                }),
            }
        }
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
        #[cfg(not(feature = "rns"))]
        ntt64::add_assign(kahe_ring64(), &mut self.coeffs, &other.coeffs);
        #[cfg(feature = "rns")]
        kahe_rns().add_assign(&mut self.coeffs, &other.coeffs);
    }
}

impl std::ops::Sub for KaheNTTPoly {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        #[cfg(not(feature = "rns"))]
        ntt64::sub_assign(kahe_ring64(), &mut self.coeffs, &other.coeffs);
        #[cfg(feature = "rns")]
        kahe_rns().sub_assign(&mut self.coeffs, &other.coeffs);
        self
    }
}

impl std::ops::Mul for KaheNTTPoly {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        #[cfg(not(feature = "rns"))]
        let coeffs = ntt64::pointwise_mul(kahe_ring64(), &self.coeffs, &other.coeffs);
        #[cfg(feature = "rns")]
        let coeffs = kahe_rns().pointwise_mul(&self.coeffs, &other.coeffs);
        Self { coeffs }
    }
}

impl KaheNTTPoly {
    pub fn rand_ntt_poly<R: Rng>(rng: &mut R) -> Self {
        #[cfg(not(feature = "rns"))]
        {
            let q = KAHE_MODULUS as u64;
            let threshold = (((1u128 << 64) / q as u128) * q as u128) as u64;
            Self {
                coeffs: core::array::from_fn(|_| {
                    let mut value = rng.next_u64();
                    while value >= threshold {
                        value = rng.next_u64();
                    }
                    value % q
                }),
            }
        }
        #[cfg(feature = "rns")]
        {
            Self {
                coeffs: kahe_rns().rand(rng),
            }
        }
    }
}

pub fn pointwise_dot_kahe(a: &[KaheNTTPoly], b: &[KaheNTTPoly]) -> KaheNTTPoly {
    debug_assert_eq!(a.len(), b.len());
    let mut result = KaheNTTPoly::default();
    #[cfg(not(feature = "rns"))]
    {
        let a: Vec<_> = a.iter().map(|poly| &poly.coeffs).collect();
        let b: Vec<_> = b.iter().map(|poly| &poly.coeffs).collect();
        ntt64::pointwise_mac(kahe_ring64(), &mut result.coeffs, &a, &b);
    }
    #[cfg(feature = "rns")]
    {
        let a: Vec<_> = a.iter().map(|poly| &poly.coeffs).collect();
        let b: Vec<_> = b.iter().map(|poly| &poly.coeffs).collect();
        kahe_rns().pointwise_mac(&mut result.coeffs, &a, &b);
    }
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

#[cfg(not(feature = "rns"))]
type CsNTTCoeffs = [u64; N];
#[cfg(feature = "rns")]
type CsNTTCoeffs = Residues<N, 2>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsNTTPoly {
    coeffs: CsNTTCoeffs,
}

impl Default for CsNTTPoly {
    fn default() -> Self {
        #[cfg(not(feature = "rns"))]
        let coeffs = [0u64; N];
        #[cfg(feature = "rns")]
        let coeffs = [[0u32; N]; 2];
        Self { coeffs }
    }
}

impl From<&CsPoly> for CsNTTPoly {
    fn from(poly: &CsPoly) -> Self {
        #[cfg(not(feature = "rns"))]
        {
            let mut coeffs = [0u64; N];
            for (output, &input) in coeffs.iter_mut().zip(&poly.coeffs) {
                let centered = arithmetic::normalize_i32(input, CS_MODULUS) as i64;
                *output = if centered < 0 {
                    (CS_AUX_MODULUS as i64 + centered) as u64
                } else {
                    centered as u64
                };
            }
            ntt64::ntt(cs_aux_ring64(), &mut coeffs);
            Self { coeffs }
        }
        #[cfg(feature = "rns")]
        {
            let ring = cs_aux_rns();
            let mut coeffs = [[0u32; N]; 2];
            for (i, &input) in poly.coeffs.iter().enumerate() {
                let residues =
                    ring.reduce_coeff(arithmetic::normalize_i32(input, CS_MODULUS) as i128);
                coeffs[0][i] = residues[0];
                coeffs[1][i] = residues[1];
            }
            ring.forward(&mut coeffs);
            Self { coeffs }
        }
    }
}

impl From<CsPoly> for CsNTTPoly {
    fn from(poly: CsPoly) -> Self {
        Self::from(&poly)
    }
}

impl From<&CsNTTPoly> for CsPoly {
    fn from(poly: &CsNTTPoly) -> Self {
        #[cfg(not(feature = "rns"))]
        let exact: [i128; N] = {
            let mut coeffs = poly.coeffs;
            ntt64::inv_ntt(cs_aux_ring64(), &mut coeffs);
            coeffs.map(|value| {
                if value > CS_AUX_MODULUS / 2 {
                    value as i128 - CS_AUX_MODULUS as i128
                } else {
                    value as i128
                }
            })
        };
        #[cfg(feature = "rns")]
        let exact: [i128; N] = {
            let ring = cs_aux_rns();
            let mut coeffs = poly.coeffs;
            ring.inverse(&mut coeffs);
            core::array::from_fn(|i| ring.lift_centered([coeffs[0][i], coeffs[1][i]]))
        };
        Self {
            coeffs: exact
                .map(|value| value.rem_euclid(CS_MODULUS as i128) as i32)
                .map(|value| arithmetic::normalize_i32(value, CS_MODULUS)),
        }
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
        #[cfg(not(feature = "rns"))]
        ntt64::add_assign(cs_aux_ring64(), &mut self.coeffs, &other.coeffs);
        #[cfg(feature = "rns")]
        cs_aux_rns().add_assign(&mut self.coeffs, &other.coeffs);
    }
}

impl std::ops::Sub for CsNTTPoly {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        #[cfg(not(feature = "rns"))]
        ntt64::sub_assign(cs_aux_ring64(), &mut self.coeffs, &other.coeffs);
        #[cfg(feature = "rns")]
        cs_aux_rns().sub_assign(&mut self.coeffs, &other.coeffs);
        self
    }
}

impl std::ops::Mul for CsNTTPoly {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        #[cfg(not(feature = "rns"))]
        let coeffs = ntt64::pointwise_mul(cs_aux_ring64(), &self.coeffs, &other.coeffs);
        #[cfg(feature = "rns")]
        let coeffs = cs_aux_rns().pointwise_mul(&self.coeffs, &other.coeffs);
        Self { coeffs }
    }
}

pub fn pointwise_dot_cs(a: &[CsNTTPoly], b: &[CsNTTPoly]) -> CsNTTPoly {
    debug_assert_eq!(a.len(), b.len());
    let mut result = CsNTTPoly::default();
    #[cfg(not(feature = "rns"))]
    {
        let a: Vec<_> = a.iter().map(|poly| &poly.coeffs).collect();
        let b: Vec<_> = b.iter().map(|poly| &poly.coeffs).collect();
        ntt64::pointwise_mac(cs_aux_ring64(), &mut result.coeffs, &a, &b);
    }
    #[cfg(feature = "rns")]
    {
        let a: Vec<_> = a.iter().map(|poly| &poly.coeffs).collect();
        let b: Vec<_> = b.iter().map(|poly| &poly.coeffs).collect();
        cs_aux_rns().pointwise_mac(&mut result.coeffs, &a, &b);
    }
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DgtNTTPoly {
    coeffs: [u64; N],
}

impl Default for DgtNTTPoly {
    fn default() -> Self {
        Self { coeffs: [0; N] }
    }
}

impl DgtNTTPoly {
    pub fn rand_ntt_poly<R: Rng>(rng: &mut R) -> Self {
        let threshold = (((1u128 << 64) / DGT_MODULUS as u128) * DGT_MODULUS as u128) as u64;
        Self {
            coeffs: core::array::from_fn(|_| {
                let mut value = rng.next_u64();
                while value >= threshold {
                    value = rng.next_u64();
                }
                value % DGT_MODULUS
            }),
        }
    }

    pub fn from_kahe(poly: &KahePoly) -> Self {
        let mut coeffs = [0u64; N];
        for (output, &input) in coeffs.iter_mut().zip(poly.coeffs()) {
            let centered = normalize64(input, KAHE_MODULUS);
            *output = if centered < 0 {
                (DGT_MODULUS as i64 + centered) as u64
            } else {
                centered as u64
            };
        }
        ntt64::ntt(dgt_ring64(), &mut coeffs);
        Self { coeffs }
    }

    pub fn to_centered_coeffs(&self) -> [i64; N] {
        let mut coeffs = self.coeffs;
        ntt64::inv_ntt(dgt_ring64(), &mut coeffs);
        coeffs.map(|value| {
            if value > DGT_MODULUS / 2 {
                value as i64 - DGT_MODULUS as i64
            } else {
                value as i64
            }
        })
    }

    pub fn coeffs(&self) -> &[u64; N] {
        &self.coeffs
    }

    pub fn from_raw(coeffs: &[u64; N]) -> Self {
        Self {
            coeffs: coeffs.map(|value| value % DGT_MODULUS),
        }
    }
}

impl std::ops::Add for DgtNTTPoly {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl std::ops::AddAssign for DgtNTTPoly {
    fn add_assign(&mut self, other: Self) {
        ntt64::add_assign(dgt_ring64(), &mut self.coeffs, &other.coeffs);
    }
}

impl std::ops::Sub for DgtNTTPoly {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        ntt64::sub_assign(dgt_ring64(), &mut self.coeffs, &other.coeffs);
        self
    }
}

pub fn pointwise_dot_dgt(a: &[DgtNTTPoly], b: &[DgtNTTPoly]) -> DgtNTTPoly {
    debug_assert_eq!(a.len(), b.len());
    let mut result = DgtNTTPoly::default();
    let a: Vec<_> = a.iter().map(|poly| &poly.coeffs).collect();
    let b: Vec<_> = b.iter().map(|poly| &poly.coeffs).collect();
    ntt64::pointwise_mac(dgt_ring64(), &mut result.coeffs, &a, &b);
    result
}
