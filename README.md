# Flashnet

Flashnet is an anonymous broadcast protocol intended to allow (TEE) clients to send messages through `t`-of-`n` (non-TEE) servers in a way that preserves anonymity of clients.
This repository is an early proof of concept built using lattice-based key-additive homomorphic encryption (BDLOP-style §4 KAHE) and the §5.3 hiding-vector commitment over the chipmunk Ring-SIS Merkle hash.
Everything apart from this section is AI-generated. Do not use anywhere near production data.


## Quick start

```sh
RAYON_NUM_THREADS=8 cargo test -j 8                                    # all unit + integration tests
RAYON_NUM_THREADS=8 cargo bench -j 8 --bench scaling                   # progressive scaling bench (~30s)
RAYON_NUM_THREADS=8 cargo run -j 8 --release --example demo            # slot-mode broadcast demo
```

`bench.sh` and `scripts/run_demo.sh` are thin wrappers around the above. The crate path-deps a sister crate `chipmunk_code` for the lattice primitives (Ring-SIS hash, dynamic-height Merkle tree, NTT polynomial multiplication; the `fast-ntt` feature enables Barrett + AVX2).

## Repository layout

```
src/
  lib.rs                    re-exports
  kahe.rs                   matrix-form KAHE (RingOtp pad-expansion + Kahe scheme with short keys / agg keys)
  sss.rs                    AdditiveSharing (n-of-n) + ShamirSharing (t-of-n) over R_q
  cs.rs                     BDLOP §5.3 hiding vector commitment over a chipmunk Merkle tree
  bulletin.rs               in-memory typed broadcast store
  consensus.rs              canonical-client-set selector + FixedSet stub
  codec.rs                  bytes <-> HVCPoly (header-prefixed and fixed-buffer flavors)
  mse.rs                    additive multi-set encoding (paper §3 Fig. 1) + pack/unpack to HVCPoly
  protocol/{mod,client,server,verify,message}.rs   ProtocolParams + round drivers + public verifier
tests/
  end_to_end.rs             core protocol integration tests (slot-mode, t-of-n, tamper rejection)
  mse_e2e.rs                MSE carried through the protocol end-to-end
benches/
  protocol.rs               criterion micro-benches per stage at small parameters
  scaling.rs                (S, N) × (μ_kahe, κ_kahe, β) bench, payload extrapolated to 1 MB
examples/
  demo.rs                   slot-mode broadcast over byte messages
  profile.rs                fixed-(S,N) loop driver for flamegraph / callgrind
```

Module file layout matches the conceptual layering: lattice primitives at the bottom (chipmunk dep), then KAHE + SSS + CS, then bulletin + consensus, then protocol drivers, with codec and mse as application-side encodings on top.

## Protocol overview

Symbols. `sk_j ∈ B_β^{κ_kahe}` = client `j`'s short KAHE secret key. `m_j ∈ R_q^{μ_kahe}` = plaintext, `c_j` = ciphertext. `(s_{j,1}, ..., s_{j,n})` = per-server Shamir shares of `sk_j` (componentwise across `κ_kahe` components). `comm_j` = single CS commitment to the per-server share-vector matrix; `d_{j,i}` = opening for server `i` carrying its `κ_kahe`-component share-vector. `t` = Shamir threshold.

```
Client j  (run_client_round):
  1. sk_j     := KAHE.gen()                                           # short, ‖·‖∞ ≤ β
  2. c_j      := KAHE.enc(sk_j, m_j)                                  # m + A·sk
  3. for k ∈ [κ_kahe]:
       (s_{j,1}^{(k)}, ..., s_{j,n}^{(k)}) := Shamir.share(sk_j[k], t, n)
     transpose into per-server κ_kahe-vectors share_vec_{j,i} = (s_{j,i}^{(0)}, ..., s_{j,i}^{(κ-1)})
  4. (comm_j, (d_{j,1}, ..., d_{j,n}))         := CS.commit(share_vec_{j,1..n})
  5. publish (c_j, comm_j) to the public bulletin
  6. privately send d_{j,i} to server i  (its s() is the κ_kahe share-vector)

Server i  (run_server_round, for canonical client set chosen by consensus):
  7. agg_open_i  := CS.sum_openings([d_{j,i} for j in canonical])
     agg_share_i := agg_open_i.s()                                    # κ_kahe-vector
  8. publish (agg_open_i, agg_share_i)

Verifier  (aggregate_and_decrypt):
  9a. summed_ctxt := KAHE.agg_ctxt([c_j for j in canonical])
      summed_comm := CS.sum_commitments([comm_j for j in canonical])
  9b. for each server i: assert CS.verify(summed_comm, agg_open_i)
                          assert agg_share_i == agg_open_i.s()
  9c. for each k ∈ [κ_kahe]:
        sk_sum[k] := ShamirSharing.recover([(server_id_i, agg_share_i[k]) for any t servers])
      agg_key := KaheAggKey::from_components(sk_sum)
  9d. return KAHE.dec(summed_ctxt, agg_key)             # = Σ m_j over canonical
```

The verifier output is a `μ_kahe`-vector of polynomials whose coefficient-wise meaning is the application's choice (slot mode, MSE peeling, custom encoding). `tests/end_to_end.rs::end_to_end_recovers_sum` is the executable spec.

## Modules

### `kahe` — key-additive homomorphic encryption

Two layers. `RingOtp` is the pad-expansion + OTP primitive: for a public `μ × κ` matrix `A` (NTT-resident), `expand(sk) = A·sk ∈ R_q^μ`, `enc(sk, m) = m + expand(sk)`, `dec(sk, c) = c − expand(sk)`. `expand` is `R`-linear, so it works on both fresh (short) and aggregate (large-norm) seeds.

`Kahe` is the scheme. `Gen` samples a *short* key (Lemma 8 hiding regime, `‖sk‖∞ ≤ β`). The `KaheKey` / `KaheAggKey` newtypes separate fresh keys (sole valid `Enc` input) from aggregate keys in `R_q^κ` (sole valid `Dec` input). The Shamir bridge (`Σ sk_j` interpolated from per-server share sums) lives in `protocol::verify`, which builds a `KaheAggKey` via `KaheAggKey::from_components` after running `ShamirSharing::recover` componentwise.

```rust
pub trait KaheScheme {
    type Params;
    type Key: Clone;
    type AggKey: Clone;
    type Message: Clone;
    type Ciphertext: Clone;

    fn setup<R: Rng>(rng: &mut R) -> Self::Params;
    fn gen<R: Rng>(rng: &mut R, pp: &Self::Params) -> Self::Key;
    fn enc(pp: &Self::Params, k: &Self::Key, m: &Self::Message) -> Self::Ciphertext;
    fn dec(pp: &Self::Params, c: &Self::Ciphertext, k: &Self::AggKey) -> Self::Message;
    fn agg_ctxt(cs: &[Self::Ciphertext]) -> Self::Ciphertext;
    fn agg_key(ks: &[Self::Key]) -> Self::AggKey;
}

pub struct KaheParams {
    pub a_matrix_ntt: Vec<Vec<HVCNTTPoly>>,   // μ_kahe × κ_kahe
    pub mu_kahe: usize,
    pub kappa_kahe: usize,
    pub sk_bound: u32,                        // β
}

pub struct RingOtp;
impl RingOtp {
    pub fn expand(pp: &KaheParams, sk: &[HVCPoly]) -> Vec<HVCPoly>;
    pub fn enc(pp: &KaheParams, sk: &[HVCPoly], m: &[HVCPoly]) -> Vec<HVCPoly>;
    pub fn dec(pp: &KaheParams, sk: &[HVCPoly], c: &[HVCPoly]) -> Vec<HVCPoly>;
}

pub struct KaheKey(/* short, len = κ_kahe */);
pub struct KaheAggKey(/* in R_q^κ */);
impl KaheAggKey {
    pub fn from_components(components: Vec<HVCPoly>) -> Self;
}

pub struct Kahe;
impl Kahe {
    pub fn setup_with_dims<R: Rng>(rng: &mut R, mu: usize, kappa: usize, sk_bound: u32) -> KaheParams;
}
impl KaheScheme for Kahe { /* default setup: (μ, κ, β) = (1, 6, 64) */ }
```

Each round must use a fresh key (standard OTP requirement, met by `gen` once per `run_client_round`).

### `sss` — secret sharing

Two implementations. `AdditiveSharing` is n-of-n over `HVCPoly` (kept for parity / additive-only uses). `ShamirSharing` is t-of-n over `R_q` with evaluation points `1..=n`; pairwise differences are units in `Z_q*`, so Lagrange at `X = 0` is well-defined despite `R_q` not being a field. The protocol uses Shamir; KAHE's `recover_key` is the bridge.

```rust
pub trait Sss {
    type Secret: Clone;
    type Share: Clone;
    fn share<R: Rng>(rng: &mut R, secret: &Self::Secret, n: usize) -> Vec<Self::Share>;
    fn recover(shares: &[Self::Share]) -> Self::Secret;
}

pub struct AdditiveSharing;
impl Sss for AdditiveSharing { type Secret = HVCPoly; type Share = HVCPoly; /* ... */ }

pub struct ShamirParams { pub t: usize, pub n: usize }
impl ShamirParams { pub fn new(t: usize, n: usize) -> Self; }   // asserts 1 ≤ t ≤ n < q

pub struct ShamirSharing;
impl ShamirSharing {
    /// Returns n shares; out[i] = f(i+1) where f(0) = secret and f has degree t-1.
    pub fn share<R: Rng>(rng: &mut R, params: &ShamirParams, secret: &HVCPoly) -> Vec<HVCPoly>;
    /// Recover from any t (0-based-index, share) samples; extras ignored.
    pub fn recover(params: &ShamirParams, samples: &[(usize, HVCPoly)]) -> HVCPoly;
}
```

### `cs` — BDLOP-style hiding vector commitment over a homomorphic Merkle tree

`Commit` packs a per-server `μ_cs`-component share vector `s ∈ R_q^{μ_cs}` into a single BDLOP leaf: with `r ← B_β^{κ_cs}` random, `c¹ = a^T r`, `c²_k = B_k r + s_k`, leaf-block `= (c¹, c²_0, ..., c²_{μ-1})` zero-padded to `block_size = (1 + μ_cs).next_power_of_two()`. The `n_servers` leaf-blocks occupy contiguous tree positions `[block_size·i .. block_size·(i+1))` of a chipmunk `Tree<HVCHash>`. The opening stores `(r, s)`, the **entire decomposed leaf-block subtree** (`2·block_size − 2` decomposed nodes), and the chipmunk Merkle path *above* the leaf-block in decomposed `(left, right)` pairs.

Why store the whole block subtree decomposed? `decompose_r` is non-linear in raw values; `hash_separate_inputs` is linear over decomposed inputs. Summing openings pointwise is correct only in the decomposed representation — recomputing decompositions after summation would not be linear.

```rust
pub trait Cs {
    type Params;
    type Secret;                                   // = Vec<HVCPoly> of length μ_cs
    type Commitment: Clone;
    type Opening: Clone;

    fn setup<R: Rng>(rng: &mut R, num_servers: usize) -> Self::Params;
    fn commit<R: Rng>(rng: &mut R, pp: &Self::Params, shares: &[Self::Secret])
        -> (Self::Commitment, Vec<Self::Opening>);
    fn verify(pp: &Self::Params, c: &Self::Commitment, o: &Self::Opening) -> bool;
    fn sum_commitments(cs: &[Self::Commitment]) -> Self::Commitment;
    /// Borrowed slices — each `Opening` is multi-hundred-KB.
    fn sum_openings(os: &[&Self::Opening]) -> Self::Opening;
}

pub struct CsParams {
    pub a_ntt: Vec<HVCNTTPoly>,                    // length κ_cs
    pub b_matrix_ntt: Vec<Vec<HVCNTTPoly>>,        // μ_cs × κ_cs
    pub mu_cs: usize,
    pub kappa_cs: usize,
    pub r_bound: u32,
    pub r_half_weight: usize,
    pub hasher: HVCHash,
    pub n_servers: usize,
    pub n_leaves: usize,
}
impl CsParams {
    pub fn block_size(&self) -> usize;             // (1 + μ_cs).next_power_of_two()
    pub fn block_height(&self) -> usize;
    pub fn total_path_len(&self) -> usize;
    pub fn stored_path_len(&self) -> usize;        // total − block_height
}

#[derive(Clone)] pub struct Commitment { pub root: HVCPoly }

#[derive(Clone)]
pub struct Opening { /* server_index, path_index, kappa_cs, mu_cs, block_size, stored_path_len, data: Box<[HVCPoly]> */ }
impl Opening {
    pub fn r(&self) -> &[HVCPoly];                 // length κ_cs
    pub fn s(&self) -> &[HVCPoly];                 // length μ_cs (the share-vector)
    pub fn block_node(&self, level: usize, idx: usize) -> &[HVCPoly];   // decomposed (HVC_WIDTH polys)
    pub fn path_node(&self, level: usize) -> (&[HVCPoly], &[HVCPoly]);  // decomposed (left, right)
}

pub struct HidingMerkleCommitment;
impl HidingMerkleCommitment {
    pub fn setup_with_dims<R: Rng>(rng: &mut R, n_servers: usize, mu_cs: usize, kappa_cs: usize) -> CsParams;
}
impl Cs for HidingMerkleCommitment { /* default setup_with_dims(.., 1, 8) */ }
```

`Verify` reconstructs the raw leaf-block from `(r, s, A, B)`, walks up the block subtree using `hash_separate_inputs` (each level's stored decomp must `projection_r` to the freshly hashed parent), then walks the stored decomposed path above the block.

### `bulletin` — in-memory broadcast store

```rust
#[derive(Clone)]
pub struct ClientPublic {
    pub ctxt: Vec<HVCPoly>,     // μ_kahe ring elements
    pub comm: Commitment,        // single CS commitment (μ_cs = κ_kahe packs the share-vector)
}

#[derive(Clone)]
pub struct ServerPublic {
    pub server_id: ServerId,
    pub clients: Vec<ClientId>,  // canonical set
    pub agg_open: Opening,       // s() is the κ_kahe-vector of summed shares
    pub agg_share: Vec<HVCPoly>, // mirrors agg_open.s()
}

pub struct InMemoryBulletin { /* Mutex<{clients, servers, canonical}> */ }
impl InMemoryBulletin {
    pub fn new() -> Self;
    pub fn publish_client(&self, id: ClientId, p: ClientPublic);
    pub fn publish_server(&self, p: ServerPublic);
    pub fn publish_canonical(&self, set: Vec<ClientId>);
    pub fn clients(&self)   -> Vec<(ClientId, ClientPublic)>;
    pub fn servers(&self)   -> Vec<ServerPublic>;
    pub fn canonical(&self) -> Option<Vec<ClientId>>;
}
```

### `consensus` — canonical-client-set selector

Defines which clients servers will jointly decrypt. Honest servers must refuse to decrypt anything other than the canonical set; anonymity holds as long as one honest server enforces this.

```rust
pub trait ClientSetSelector {
    fn canonical_set(&self) -> Vec<ClientId>;
}

pub struct FixedSet(pub Vec<ClientId>);
impl ClientSetSelector for FixedSet { /* returns self.0.clone() */ }
```

### `protocol` — params + round drivers + verifier

`ProtocolParams` couples KAHE, CS, and Shamir in one bundle and enforces `μ_cs = κ_kahe` (one CS pipeline carries the entire `κ_kahe`-component share-vector per server, replacing what would otherwise be `κ_kahe` parallel CS instances).

```rust
pub struct ProtocolParams {
    pub kahe: KaheParams,
    pub cs: <HidingMerkleCommitment as Cs>::Params,
    pub shamir: ShamirParams,
}
impl ProtocolParams {
    /// Threshold `t = max(⌊n/2⌋ + 1, n − 2)`.
    pub fn setup<R: Rng>(rng: &mut R, n_servers: usize) -> Self;
    pub fn setup_with_threshold<R: Rng>(rng: &mut R, n_servers: usize, t: usize) -> Self;
    pub fn setup_with_kahe_dims<R: Rng>(rng: &mut R, n_servers: usize, mu: usize, kappa: usize) -> Self;
    pub fn setup_with_kahe_dims_beta<R: Rng>(rng: &mut R, n_servers: usize, mu: usize, kappa: usize, beta: u32) -> Self;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)] pub struct ClientId(pub u32);
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)] pub struct ServerId(pub u32);

pub struct ClientRound {
    pub client_id: ClientId,
    pub public: ClientPublic,
    /// One Opening per server. Its s() is the κ_kahe-vector of Shamir shares for this server.
    pub private: Vec<(ServerId, Opening)>,
}

pub fn run_client_round<R: Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    client_id: ClientId,
    message: <Kahe as KaheScheme>::Message,        // Vec<HVCPoly> of length μ_kahe
    servers: &[ServerId],
) -> ClientRound;

pub struct ServerInbox {
    pub server_id: ServerId,
    pub items: Vec<(ClientId, Opening)>,
}

#[derive(Debug, PartialEq)]
pub enum ServerRoundError { MissingClient(ClientId) }

pub fn run_server_round(
    inbox: &ServerInbox,
    canonical: &[ClientId],
) -> Result<ServerPublic, ServerRoundError>;

#[derive(Debug, PartialEq)]
pub enum VerifyError {
    MissingClient(ClientId),
    InvalidServerOpening(usize),
    ShareOpeningMismatch(usize),
    NoServers,
    /// `server_outputs.len() < t`, or duplicate / out-of-range `server_id`.
    BadServerCoverage,
    /// A `ServerPublic.clients` field disagreed with the canonical set.
    InconsistentCanonical(ServerId),
    /// `agg_share.len() != κ_kahe`, or ciphertext length != μ_kahe.
    InconsistentKappa(ServerId),
}

pub fn aggregate_and_decrypt(
    pp: &ProtocolParams,
    canonical: &[ClientId],
    publics: &[(ClientId, ClientPublic)],
    server_outputs: &[ServerPublic],
) -> Result<Vec<HVCPoly>, VerifyError>;
```

`aggregate_and_decrypt` requires at least `t` distinct, in-range server outputs (`BadServerCoverage`), enforces every `ServerPublic.clients` matches `canonical` (`InconsistentCanonical`), and checks `agg_share == agg_open.s()` per server (`ShareOpeningMismatch`). It runs **one** CS verification per server (the `μ_cs = κ_kahe` packing) and recovers the aggregate KAHE key by Lagrange interpolation across the first `t` servers, componentwise over the `κ_kahe` components. Tampered Merkle openings reject via `InvalidServerOpening`.

### `codec` — bytes ↔ `HVCPoly`

Two flavors. `encode`/`decode` carry a 4-byte little-endian length header so the decoded length is exact for a single message; safe for round-trips, **not** safe to sum across clients (the per-coefficient header sums multiply by N and overflow). `encode_raw`/`decode_raw` skip the header and return a fixed-size buffer (`polys.len() * 1024` bytes) — use this when multiple clients write into disjoint slots of a fixed buffer and the recovered sum is decoded as a single layout.

```rust
#[derive(Debug, PartialEq)]
pub enum CodecError {
    Empty,
    LengthOverflow { claimed: u32, available: usize },
    CoeffOutOfRange { index: usize, value: i32 },
}

pub fn encode    (bytes: &[u8])      -> Vec<HVCPoly>;
pub fn decode    (polys: &[HVCPoly]) -> Result<Vec<u8>, CodecError>;
pub fn encode_raw(bytes: &[u8])      -> Vec<HVCPoly>;
pub fn decode_raw(polys: &[HVCPoly]) -> Result<Vec<u8>, CodecError>;
```

Layout for both: 2 bytes per coefficient (little-endian `u16`), 512 coefficients per `HVCPoly`, 1024 bytes per poly. Coefficients land in `[0, 65536) ⊂ [0, q)` so a fresh single-message encode/decode is exact. Decoding a sum is meaningful only when the application controls the encoding so per-coefficient sums stay below `q = 202_753`.

### `mse` — additive multi-set encoding (paper §3 Fig. 1)

The application payload. Three matrices `(C, K, V)` of shape `γ × δ` over `Z_q`. Insert one element `x ∈ Z_q` with fresh randomness `r ← Z_q`: for each row `i ∈ [γ]`, compute `j := PRF(prf_key, (i, r)) mod δ` and `(C[i,j] += 1, K[i,j] += r, V[i,j] += x)`. `Decode` peels cells with `C[i, j] = 1`, reads `(r, x)` from `(K, V)`, emits `x`, and subtracts that element's contribution from every row by re-running the PRF. Theorem 3 correctness: `2^{-(γ-2) log ρ} + negl(λ)`. v1 PRF is SHA-256 keyed by `prf_key`.

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MseParams {
    pub gamma: usize,
    pub delta: usize,
    pub prf_key: [u8; 32],
}
impl MseParams {
    pub fn new(gamma: usize, delta: usize, prf_key: [u8; 32]) -> Self;   // asserts γ ≥ 2, δ ≥ 1
    pub fn total_cells(&self) -> usize;                                  // γ · δ
    pub fn total_scalars(&self) -> usize;                                // 3 · γ · δ
    pub fn max_clients(&self) -> u32;                                    // q − 1 (advisory)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MseEncoding {
    pub params: MseParams,
    pub c: Vec<i32>, pub k: Vec<i32>, pub v: Vec<i32>,
}

#[derive(Debug, PartialEq)]
pub enum MseError {
    PeelStalled,
    ParamsMismatch,
}

impl MseEncoding {
    pub fn new(params: MseParams) -> Self;
    pub fn insert<R: Rng>(&mut self, rng: &mut R, x: i32);
    pub fn insert_with_r(&mut self, x: i32, r: i32);
    pub fn add_assign(&mut self, other: &Self) -> Result<(), MseError>;
    pub fn decode(&self) -> Result<Vec<i32>, MseError>;

    pub fn n_polys(params: &MseParams) -> usize;
    pub fn pack(&self) -> Vec<HVCPoly>;
    pub fn unpack(params: &MseParams, polys: &[HVCPoly]) -> Self;
}
```

`pack` flattens `(C, K, V)` row-major into a coefficient stream (C first, then K, then V). `unpack` is the inverse, lifting each `HVCPoly`'s coefficients to canonical signed reps. Pointwise sum of packed encodings unpacks to the multiset union — this is the property the protocol exploits to carry an MSE end-to-end. `tests/mse_e2e.rs::mse_recovers_through_flashnet` is the executable spec.

## Security properties

- **Binding** (commitment): given `comm`, an adversary cannot produce a different `(r', s', path')` that verifies — Module-SIS hardness on the BDLOP leaf and Ring-SIS hardness on every internal Merkle node.
- **Hiding** (commitment, single opening): the BDLOP leaf `(a^T r, B r + s)` is statistically uniform over `R_q^{1+μ_cs}` by the leftover hash lemma when `r` is short and the matrix `(a | B)` has enough min-entropy.
- **Strong hiding** (sum of openings): pointwise sums of low-norm `r`s stay low-norm with margin; the linear hash + decomposed-path representation makes summed openings verify against the summed root; per-client `s_j` remains hidden under any subset of summands.
- **Threshold liveness**: any `t` honest servers' `agg_share`s suffice to recover `Σ sk_j` via Lagrange; up to `n − t` servers can be offline or corrupted.
- **Integrity**: `aggregate_and_decrypt` rejects tampered Merkle openings (`InvalidServerOpening`), tampered key shares (`ShareOpeningMismatch`), inconsistent canonical sets (`InconsistentCanonical`), and shape mismatches (`InconsistentKappa`); honest output is `Σ m_j` over canonical clients, nothing else.
- **Anonymity**: holds as long as at least one honest server refuses to decrypt sets smaller than the canonical set (`consensus` invariant).

## Parameters and limits

Lattice ring (from chipmunk): `Z_q[x]/(x^N+1)` with `q = 202_753`, `N = 512`. `HVC_WIDTH = 3` decomposed polys per node, `ZETA = 29` decomposition base bound. Tree height = `⌈log2(block_size · n_servers)⌉`, where `block_size = (1 + μ_cs).next_power_of_two()`.

KAHE defaults (`Kahe::setup`): `(μ, κ, β) = (1, 6, 64)` — chosen so Lemma 8 hides at λ = 128 for `ρ ≤ 2²⁰` with the chipmunk HVC modulus. The scaling bench compares two operating points: **(μ=16, κ=31, β=1024)** with `block_size = 32` (best 1 MB throughput) and **(μ=8, κ=15, β=2048)** with `block_size = 16` (best per-round latency — `1+κ` exactly hits the lower power-of-two).

CS defaults (`HidingMerkleCommitment::setup`): `μ_cs = 1, κ_cs = 8, r_bound = 64, r_half_weight = 1` (balanced ternary). The protocol's `setup_with_dims` overrides `μ_cs := κ_kahe` so a single CS pipeline carries the entire share-vector.

Threshold: `t = max(⌊n/2⌋ + 1, n − 2)` by default. Per-opening size grows as `(2·block_size − 2)·HVC_WIDTH + 2·stored_path_len·HVC_WIDTH + κ_cs + μ_cs` ring elements; for `μ_cs = κ_kahe = 6` (`block_size = 8`) and `n_servers = 8`, that is roughly 50 KB of decomposed nodes plus per-opening `r` and `s`.

## Tests, benches, demos

```
cargo test                              # all unit + integration tests
cargo bench --bench protocol            # criterion micro-benches per stage at small parameters
cargo bench --bench scaling             # (S, N) cell × (μ, κ, β) variants
                                        #   BENCH_BUDGET_SECS=N to extend (default 300)
cargo run --release --example demo      # 6 clients × 128-byte slots over a 1024-byte buffer
```

`tests/end_to_end.rs::end_to_end_recovers_sum` is the canonical executable spec. Other coverage in that file: `slot_mode_disjoint_clients_recover_each_payload`, `slot_mode_8kb_message_multi_poly`, `tampered_agg_share_rejected`, `high_norm_r_rejected_in_protocol`, `recovers_from_t_of_n_servers`. `tests/mse_e2e.rs::mse_recovers_through_flashnet` carries an MSE multiset end-to-end.

### Scaling bench

Reported wall times **assume parallel deployment**: clients run in parallel on N machines, servers in parallel on S machines, the verifier is one party. Per-round wall = `client_round + server_round + verify_round`. Sub-stages within each role are sequential on the same machine. Each round ships `μ_kahe · 1024` bytes of broadcast payload; the bench extrapolates a 1 MB total as `⌈1024 / μ⌉ · per_round`. Default cell: `(S, N) = (8, 300)` × the two `(μ, κ, β)` points above.

Sample on an 8-core machine (Intel Core Ultra 7 155H, AVX2), `RAYON_NUM_THREADS=8`, `chipmunk_code` built with `--features fast-ntt` (Barrett + AVX2), default `parallel` feature off:

```
S= 8 N= 300 (μ= 16, κ= 31, β= 1024) | client  25.5ms server  10.9ms verify  18.2ms  per-round  54.6ms  ⇒ 1MB (64 rounds)  3.49s  (0.300 MB/s)
S= 8 N= 300 (μ=  8, κ= 15, β= 2048) | client  12.2ms server   6.6ms verify   9.3ms  per-round  28.1ms  ⇒ 1MB (128 rounds)  3.60s  (0.291 MB/s)
```

(Numbers depend on machine; run-to-run noise ≈ 15–20 %.) The two cells trade per-round latency against round count:

- **(μ=16, κ=31, β=1024)** — 64 rounds for 1 MB at 54.6 ms each → **0.30 MB/s**, best 1 MB throughput.
- **(μ=8, κ=15, β=2048)** — `block_size` halves (16 vs 32) because `1 + κ = 16` exactly hits the lower power-of-two; per-round drops to 28 ms (~half), but 128 rounds undo most of that on 1 MB. Pick this if the message you're sending is small (single round) and end-to-end latency matters more than throughput.

A wider sweep over `(μ, κ, β) ∈ {4..24} × {64..4096}` confirmed these as the only two non-dominated cells: `block_size` cliffs at `κ ≤ 15`, `κ ≤ 31` are the dominant variable, and within a `block_size` class higher β (lower κ_min) buys ≤ 2 % per-round.

#### Where time is spent

CPU profile at `(S=8, N=300, μ=16, κ=31, β=1024)`, fast-ntt on, after the cache fixes below:

```
29.6%  ntt_stages_scalar              forward NTT, scalar tail (ht ∈ {4,2,1})
14.5%  ntt_avx2_dispatch              forward NTT, AVX2 (ht ≥ 8)
11.0%  pointwise_mac_avx2_dispatch    NTT-domain inner product
 7.9%  flashnet::sss::scalar_mul      Shamir Lagrange interp scalar mult
 7.4%  HVCPoly::decompose_r           non-linear decomposition for Merkle leaves
 6.9%  inv_ntt_stages_scalar          inverse NTT, scalar head
 6.1%  inv_ntt_avx2_dispatch          inverse NTT, AVX2
 3.2%  __memmove                      Vec growth / HVCPoly clones
```

NTT (forward + inverse, scalar + AVX2) is ~57 % of cycles even with `fast-ntt` enabled — the scalar tail / head paths exist because the AVX2 butterfly needs `ht ≥ 8` per inner iteration but the last 3 forward and first 3 inverse stages have stride < 8. Closing that gap requires in-register shuffles or radix-4/8 fused stages in `chipmunk_code`'s `simd.rs`.

Cache-miss profile (P-core LLC-load-misses) at the same cell:

```
52.5%  cs::HidingMerkleCommitment::sum_openings    streaming sum across ρ openings
26.9%  __memmove                                   per-call buffer / clone allocations
```

`sum_openings` dominates LLC misses: each opening is ~600 KB, summed coefficient-wise across ρ canonical clients. Mitigations applied:

- **Opening-major flat accumulator** (`cs.rs::sum_openings`): one contiguous `Vec<[i32; N]>` of length `total` (was 5 scattered `acc_*` vecs); reads each opening's `data` slice sequentially; single mod-q + center pass at the end. Halved server-round time at large N.
- **Thread-local accumulator**: the 600 KB acc is reused across calls (`SUM_OPENING_ACC` thread-local) so the malloc/zero-fill cost is paid once per worker. ~30 % additional speedup at `(S=8, N=300)`.
- **chipmunk `Tree::new_with_leaf_nodes`**: the `par_iter_mut` calls are now `#[cfg(feature = "parallel")]`-gated. Off by default (rayon overhead exceeds the win at the bench operating point); re-enable when `n_leaves ≫ #cores`.

After these, `sum_openings` is still the cache-miss top — it's a streaming sum bounded by DRAM bandwidth, no further easy win available without a GPU offload or algorithmic reshape.
