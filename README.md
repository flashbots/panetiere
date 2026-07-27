# Panetière

Panetière is an anonymous broadcast protocol intended to allow (TEE) clients to send messages through `t`-of-`n` (non-TEE) servers in a way that preserves anonymity of clients.
This repository is an early proof of concept built using lattice-based key-additive homomorphic encryption (RLWE-based KAHE) and the §5.3 hiding-vector commitment over the chipmunk Ring-SIS Merkle hash.
Everything apart from this section is AI-generated. Do not use anywhere near production data.


## Quick start

```sh
RAYON_NUM_THREADS=8 cargo test -j 8                                    # all unit + integration tests
RAYON_NUM_THREADS=8 cargo bench -j 8 --bench scaling                   # progressive scaling bench (~30s)
RAYON_NUM_THREADS=8 cargo run -j 8 --release --example demo            # slot-mode broadcast demo
```


[bench.sh](bench.sh) and [scripts/run_demo.sh](scripts/run_demo.sh) are thin wrappers around the above. The crate depends on `chipmunk_code` (a pinned git dependency, [github.com/Ruteri/Chipmunk](https://github.com/Ruteri/Chipmunk)) for the lattice primitives (Ring-SIS hash, dynamic-height Merkle tree, NTT polynomial multiplication; the `fast-ntt` feature enables Barrett + AVX2 NTT across all three rings — HVC, CS, KAHE).

## Repository layout

```
src/
  lib.rs                    re-exports
  kahe.rs                   matrix-form RLWE KAHE (Gaussian short keys / agg keys + CS↔KAHE ring bridge)
  sss.rs                    AdditiveSharing (n-of-n) + ShamirSharing (t-of-n) over the CS ring R_{q_cs}
  cs.rs                     BDLOP §5.3 hiding vector commitment over a chipmunk Merkle tree
  bulletin.rs               in-memory typed broadcast store
  codec.rs                  bytes <-> KahePoly (header-prefixed / fixed-buffer flavors) + beacon-driven slot allocation
  mse.rs                    additive multi-set encoding (paper §3 Fig. 1) + pack/unpack to KahePoly
  protocol/{mod,client,server,verify}.rs   ProtocolParams + round drivers + public verifier
tests/
  end_to_end.rs             core protocol integration tests (slot-mode, t-of-n, tamper rejection)
  mse_e2e.rs                MSE carried through the protocol end-to-end
benches/
  protocol.rs               criterion micro-benches per stage at small parameters
  scaling.rs                (S, N) × (μ_kahe, κ_kahe) bench, payload extrapolated to 1 MB
examples/
  demo.rs                   slot-mode broadcast over byte messages
  profile.rs                fixed-(S,N) loop driver for flamegraph / callgrind
```

Module file layout matches the conceptual layering: lattice primitives at the bottom (chipmunk dep), then KAHE + SSS + CS, then bulletin, then protocol drivers, with codec and mse as application-side encodings on top.

## Protocol overview

Symbols. $sk_j ∈ D_{σ_s}^{κ_{kahe}}$ = client $j$'s short (discrete-Gaussian) KAHE secret key on the KAHE ring $R_{q_{kahe}}$. $m_j ∈ R_{q_{kahe}}^{μ_{kahe}}$, $c_j$ = ciphertext (both on the KAHE ring). $(s_{j,1}, ..., s_{j,n})$ = per-server Shamir shares of $sk_j$ on the CS ring $R_{q_{cs}}$ (each KAHE-key component is bridged into $R_{q_{cs}}$ and shared componentwise across $κ_{kahe}$ components). $comm_j$ = single CS commitment to the per-server share-vector matrix; $d_{j,i}$ = opening for server $i$ carrying its $κ_{kahe}$-component share-vector. $t$ = Shamir threshold.

```
Client j  (run_client_round):
  1. sk_j     := KAHE.gen()                                           # short, sk ← D_{σ_s}^{κ_kahe}
  2. c_j      := KAHE.enc(rng, sk_j, m_j)                             # m + A·sk + t·e
  3. for k ∈ [κ_kahe]:
       bridge sk_j[k] (KAHE ring) into the CS ring, then
       (s_{j,1}^{(k)}, ..., s_{j,n}^{(k)}) := Shamir.share(sk_j[k], t, n)   # over R_{q_cs}
     transpose into per-server κ_kahe-vectors share_vec_{j,i} = (s_{j,i}^{(0)}, ..., s_{j,i}^{(κ-1)})
  4. (comm_j, (d_{j,1}, ..., d_{j,n}))         := CS.commit(share_vec_{j,1..n})
  5. publish (c_j, comm_j) to the public bulletin
  6. privately send d_{j,i} to server i  (its s() is the κ_kahe share-vector)

Server i  (run_server_round, over a canonical client set agreed out of band):
  7. agg_open_i  := CS.sum_openings([d_{j,i} for j in canonical])
     agg_share_i := agg_open_i.s()                                    # κ_kahe-vector
  8. publish (agg_open_i, agg_share_i)

Verifier  (aggregate_and_decrypt):
  9a. summed_ctxt := KAHE.agg_ctxt([c_j for j in canonical])
      summed_comm := CS.sum_commitments([comm_j for j in canonical])
  9b. for each server i: assert CS.verify(summed_comm, agg_open_i)
                          assert agg_share_i == agg_open_i.s()
  9c. for each k ∈ [κ_kahe]:
        sk_sum[k] := ShamirSharing.recover([(server_id_i, agg_share_i[k]) for any t servers])   # over R_{q_cs}
                     then lift each recovered component from R_{q_cs} into R_{q_kahe}
      agg_key := KaheAggKey::from_components(sk_sum)
  9d. return KAHE.dec(summed_ctxt, agg_key)             # = Σ m_j over canonical
```

The verifier output is a $μ_{kahe}·l$-vector of KAHE-ring polynomials whose coefficient-wise meaning is the application's choice (slot mode, MSE peeling, custom encoding). [tests/end_to_end.rs](tests/end_to_end.rs)`::end_to_end_recovers_sum` is the executable spec.

### Aggregated flow (optional)

When the public $(c_j, comm_j)$ fan-in dominates (many clients, large payloads), an **aggregator** can sit between clients and the verifier. Clients in a group send their $(c_j, comm_j)$ to the group's aggregator instead of broadcasting them; the aggregator runs `protocol::aggregator::run_aggregator_round` — `KAHE.agg_ctxt` + `CS.sum_commitments` over the group — and forwards one signed aggregate. The verifier re-sums the per-group aggregates (the same two associative ops) and calls `protocol::verify::decrypt_aggregate(pp, summed_ctxt, summed_comm, server_outputs)`, which verifies+decrypts exactly as `aggregate_and_decrypt` but takes the ciphertext/commitment already summed. **Openings are untouched** — they still go to the servers per-server, so the threshold/privacy model is identical to the base flow. This is purely additive: the base path above is unchanged. [tests/end_to_end.rs](tests/end_to_end.rs)`::aggregated_recovers_same_sum` asserts the aggregated path decodes byte-identically; [benches/protocol.rs](benches/protocol.rs) and [benches/scaling.rs](benches/scaling.rs) report the leader-side bytes/CPU saved (G group aggregates vs ρ client posts).

## Modules

### `kahe` — key-additive homomorphic encryption

RLWE-based KAHE on its own ring $R_{q_{kahe}}$ (chipmunk's `KahePoly`, $q_{kahe} ≈ 2^28$), decoupled from the CS ring. Per-poly form: with a public $μ × κ$ matrix `A` (NTT-resident), `Enc(m, sk) = m + A·sk + t·e mod q_kahe` (fresh error $e ← D_{σ_e}$); `Dec(c, sk) = ((c − A·sk) mod q_kahe) reduced mod t` in centered representatives. `enc` batches `l` ciphertext chunks under one key, each chunk using its own matrix $A_i$; messages and ciphertexts are flat `Vec<KahePoly>` of length $μ_kahe · l$.

`Gen` samples a *short* key (discrete Gaussian $D_{σ_s}^κ$). The `KaheKey` / `KaheAggKey` newtypes separate fresh keys (sole valid `Enc` input) from aggregate keys in $R_{q_{kahe}}^κ$ (sole valid `Dec` input). The Shamir bridge ($Σ sk_j$ interpolated from per-server share sums on the CS ring, then lifted into the KAHE ring) lives in `protocol::verify`, which builds a `KaheAggKey` via `KaheAggKey::from_components` after running `ShamirSharing::recover` componentwise and `lift_cs_to_kahe`.

```rust
pub trait KaheScheme {
    type Params;
    type Key: Clone;
    type AggKey: Clone;
    type Message: Clone;
    type Ciphertext: Clone;

    fn setup<R: Rng>(rng: &mut R) -> Self::Params;
    fn gen<R: Rng>(rng: &mut R, pp: &Self::Params) -> Self::Key;
    fn enc<R: Rng>(rng: &mut R, pp: &Self::Params, k: &Self::Key, m: &Self::Message) -> Self::Ciphertext;
    fn dec(pp: &Self::Params, c: &Self::Ciphertext, k: &Self::AggKey) -> Self::Message;
    fn agg_ctxt(cs: &[Self::Ciphertext]) -> Self::Ciphertext;
    fn agg_key(ks: &[Self::Key]) -> Self::AggKey;
}

pub struct KaheParams {
    pub a_matrices_ntt: Vec<Vec<Vec<KaheNTTPoly>>>,   // l × (μ_kahe × κ_kahe)
    pub mu_kahe: usize,
    pub kappa_kahe: usize,
    pub l: usize,                             // ciphertext chunks per enc/dec
    pub sigma_s: f64,                         // key std dev
    pub sigma_e: f64,                         // error std dev
    pub t_modulus: u32,                       // plaintext modulus t
}

pub struct KaheKey(/* short, len = κ_kahe */);
pub struct KaheAggKey(/* in R_{q_kahe}^κ */);
impl KaheAggKey {
    pub fn from_components(components: Vec<KahePoly>) -> Self;
}

// CS ↔ KAHE bridge (centered-rep re-interpretation; lossless for ‖·‖∞ ≤ q_cs/2):
pub fn lift_cs_to_kahe(p: &CsPoly) -> KahePoly;       // verifier side, CS → KAHE for decryption
pub fn kahe_to_cs_centered(p: &KahePoly) -> CsPoly;   // client side, KAHE → CS for Shamir input

pub struct Kahe;
impl Kahe {
    pub fn setup_with_dims<R: Rng>(
        rng: &mut R, mu_kahe: usize, kappa_kahe: usize, l: usize,
        sigma_s: f64, sigma_e: f64, t_modulus: u32,
    ) -> KaheParams;
}
impl KaheScheme for Kahe { /* default setup: (μ, κ, l) = (1, 1, 1), σ_s=σ_e=15.72, t=2^16 */ }
```

Each round must use a fresh key (standard requirement, met by `gen` once per `run_client_round`).

### `sss` — secret sharing

Two implementations, both over the CS ring $R_{q_{cs}}$ (chipmunk's `CsPoly`). `AdditiveSharing` is n-of-n (kept for parity / additive-only uses). `ShamirSharing` is t-of-n over $R_{q_{cs}}$ with evaluation points $1..=n$; pairwise differences are units in $Z_{q_{cs}}*$, so Lagrange at $X = 0$ is well-defined despite $R_{q_{cs}}$ not being a field. The protocol uses Shamir; the CS↔KAHE bridge (`kahe_to_cs_centered` / `lift_cs_to_kahe`) crosses to the KAHE ring.

```rust
pub trait Sss {
    type Secret: Clone;
    type Share: Clone;
    fn share<R: Rng>(rng: &mut R, secret: &Self::Secret, n: usize) -> Vec<Self::Share>;
    fn recover(shares: &[Self::Share]) -> Self::Secret;
}

pub struct AdditiveSharing;
impl Sss for AdditiveSharing { type Secret = CsPoly; type Share = CsPoly; /* ... */ }

pub struct ShamirParams { pub t: usize, pub n: usize }
impl ShamirParams { pub fn new(t: usize, n: usize) -> Self; }   // asserts 1 ≤ t ≤ n and n < q_cs = 147457

pub struct ShamirSharing;
impl ShamirSharing {
    /// Returns n shares; out[i] = f(i+1) where f(0) = secret and f has degree t-1.
    pub fn share<R: Rng>(rng: &mut R, params: &ShamirParams, secret: &CsPoly) -> Vec<CsPoly>;
    /// Recover from any t (0-based-index, share) samples; extras ignored.
    pub fn recover(params: &ShamirParams, samples: &[(usize, CsPoly)]) -> CsPoly;
}
```

### `cs` — BDLOP-style hiding vector commitment over a homomorphic Merkle tree

All BDLOP arithmetic lives on the **CS ring** $R_{q_{cs}}$ (chipmunk's `CsPoly`, $q_{cs} = 147457$); the chipmunk Merkle-tree hash lives on the **HVC ring** $R_{q_{hvc}}$ ($q_{hvc} = 40961$). `Commit` packs a per-server $μ_{cs}$-component share vector $s ∈ R_{q_cs}^{μ_{cs}}$ into a single BDLOP leaf: with $r ← B_{β_{cs}}^{κ_{cs}}$ random, $c¹ = a^T r$, $c²_k = B_k r + s_k$, leaf-block $= (c¹, c²_0, ..., c²_{μ-1})$ (all `CsPoly`) zero-padded to `block_size = (1 + μ_cs).next_power_of_two()`. The `n_servers` block roots occupy positions of a chipmunk `Tree<HVCHash>`. The opening stores `(r, s)` on the CS ring, the **entire decomposed leaf-block subtree** (`2·block_size − 2` decomposed nodes), and the chipmunk Merkle path *above* the block root in decomposed `(left, right)` pairs — all on the HVC ring.

**CS → HVC bridge.** A BDLOP leaf is a `CsPoly` whose coefficients span $[-q_{cs}/2, q_{cs}/2]$, larger than $q_{hvc}$. To feed it into the HVC tree hash, each leaf element is base-69 ($2ζ+1$) decomposed into `HVC_WIDTH = 3` `HVCPoly` digits (`CsPoly::decompose_r_to_hvc`); digits are tiny ($|·| ≤ ζ = 34 ≪ q_hvc$) so they embed losslessly. The left-inverse `CsPoly::project_r_from_hvc` is linear in the digits.

Why store the whole block subtree decomposed? `decompose_r` is non-linear in raw values; `hash_separate_inputs` is linear over decomposed inputs. Summing openings pointwise is correct only in the decomposed representation — recomputing decompositions after summation would not be linear.

```rust
pub trait Cs {
    type Params;
    type Secret;                                   // = Vec<CsPoly> of length μ_cs
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
    pub a_ntt: Vec<CsNTTPoly>,                     // length κ_cs
    pub b_matrix_ntt: Vec<Vec<CsNTTPoly>>,         // μ_cs × κ_cs
    pub mu_cs: usize,
    pub kappa_cs: usize,
    pub beta_cs: u32,                              // fresh randomness sampling radius
    pub r_bound: u32,                              // aggregated-opening verify bound on r
    pub beta_agg_hvc: u32,                         // aggregated HVC digit bound (ρ_max·ζ)
    pub hasher: HVCHash,
    pub n_servers: usize,
    pub n_leaves: usize,
}
impl CsParams {
    pub fn block_size(&self) -> usize;             // (1 + μ_cs).next_power_of_two()
    pub fn block_height(&self) -> usize;
    pub fn total_path_len(&self) -> usize;
    pub fn stored_path_len(&self) -> usize;
}

#[derive(Clone)] pub struct Commitment { pub root: HVCPoly }

#[derive(Clone)]
pub struct Opening { /* server_index, path_index, kappa_cs, mu_cs, block_size, stored_path_len,
                        rs: Box<[CsPoly]>, data: Box<[HVCPoly]> */ }
impl Opening {
    pub fn r(&self) -> &[CsPoly];                  // length κ_cs (CS ring)
    pub fn s(&self) -> &[CsPoly];                  // length μ_cs, the share-vector (CS ring)
    pub fn block_node(&self, level: usize, idx: usize) -> &[HVCPoly];   // decomposed (HVC_WIDTH polys)
    pub fn path_node(&self, level: usize) -> (&[HVCPoly], &[HVCPoly]);  // decomposed (left, right)
    pub fn pack(&self, r_bound: u32, s_bound: u32, tree_bound: u32) -> PackedOpening;
    pub fn from_packed(p: &PackedOpening) -> Result<Opening, OpeningDecodeError>;
}

#[derive(Debug, PartialEq, Eq)]
pub enum OpeningDecodeError {
    BadHeader,    // header shape fields inconsistent or implausibly large
    Truncated,    // `bytes` shorter than the declared regions require
}

pub struct HidingMerkleCommitment;
impl HidingMerkleCommitment {
    pub fn setup_with_dims<R: Rng>(rng: &mut R, n_servers: usize, mu_cs: usize, kappa_cs: usize) -> CsParams;
}
impl Cs for HidingMerkleCommitment { /* default setup_with_dims(.., 1, 5) */ }
```

`Verify` reconstructs the raw CS leaf-block from $(r, s, a, B)$, walks up the block subtree (level 0 via `project_r_from_hvc` on the CS ring, higher levels via `projection_r` on the HVC ring, each stored decomp hashed to the freshly computed parent), then walks the stored decomposed path above the block root. It also bounds $‖r‖∞ ≤ r_{bound}$ (CS ring) and every decomposed digit by `beta_agg_hvc`.

### `bulletin` — in-memory broadcast store

```rust
#[derive(Clone)]
pub struct ClientBulletinEntry {
    pub ctxt: Vec<KahePoly>,     // KAHE ciphertext, μ_kahe·l ring elements
    pub comm: Commitment,        // single CS commitment (μ_cs = κ_kahe packs the share-vector)
}

#[derive(Clone)]
pub struct ServerBulletinEntry {
    pub server_id: ServerId,
    pub clients: Vec<ClientId>,  // canonical set
    pub agg_open: Opening,       // s() is the κ_kahe-vector of summed shares (CS ring)
    pub agg_share: Vec<CsPoly>,  // mirrors agg_open.s()
}

pub struct InMemoryBulletin { /* Mutex<{clients, servers, canonical}> */ }
impl InMemoryBulletin {
    pub fn new() -> Self;
    pub fn publish_client(&self, id: ClientId, p: ClientBulletinEntry);
    pub fn publish_server(&self, p: ServerBulletinEntry);
    pub fn publish_canonical(&self, set: Vec<ClientId>);
    pub fn clients(&self)   -> Vec<(ClientId, ClientBulletinEntry)>;
    pub fn servers(&self)   -> Vec<ServerBulletinEntry>;
    pub fn canonical(&self) -> Option<Vec<ClientId>>;
}
```

### `protocol` — params + round drivers + verifier

`ProtocolParams` couples KAHE, CS, and Shamir in one bundle and enforces $μ_{cs} = κ_{kahe}$ (one CS pipeline carries the entire $κ_{kahe}$-component share-vector per server, replacing what would otherwise be $κ_{kahe}$ parallel CS instances).

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
    pub fn setup_with_kahe_dims<R: Rng>(rng: &mut R, n_servers: usize, mu_kahe: usize, kappa_kahe: usize) -> Self;
    pub fn setup_with_kahe_dims_full<R: Rng>(
        rng: &mut R, n_servers: usize, mu_kahe: usize, kappa_kahe: usize, l: usize,
        sigma_s: f64, sigma_e: f64, t_modulus: u32,
    ) -> Self;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)] pub struct ClientId(pub u32);
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)] pub struct ServerId(pub u32);

pub struct ClientRound {
    pub client_id: ClientId,
    pub encrypted_message: ClientBulletinEntry,
    /// Per-server ECIES envelope (pke::encrypt) over the bit-packed Opening.
    /// The Opening's s() is the κ_kahe-vector of Shamir shares for that server.
    pub sealed_openings: Vec<(ServerId, Vec<u8>)>,
}

pub fn run_client_round<R: CryptoRng + Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    client_id: ClientId,
    message: <Kahe as KaheScheme>::Message,        // Vec<KahePoly> of length μ_kahe·l
    servers: &[(ServerId, pke::PublicKey)],
) -> ClientRound;

pub struct ServerInbox {
    pub server_id: ServerId,
    pub items: Vec<(ClientId, Opening)>,
}

#[derive(Debug, PartialEq)]
pub enum ServerRoundError { MissingClient(ClientId) }

/// Open one client's sealed envelope into its Opening.
pub fn unseal_opening(key: &pke::PrivateKey, sealed: &[u8]) -> Option<Opening>;

pub fn run_server_round(
    inbox: &ServerInbox,
    canonical: &[ClientId],
) -> Result<ServerBulletinEntry, ServerRoundError>;

#[derive(Debug, PartialEq)]
pub enum VerifyError {
    MissingClient(ClientId),
    InvalidServerOpening(usize),
    ShareOpeningMismatch(usize),
    NoServers,
    /// `server_outputs.len() < t`, or duplicate / out-of-range `server_id`.
    BadServerCoverage,
    /// A `ServerBulletinEntry.clients` field disagreed with the canonical set.
    InconsistentCanonical(ServerId),
    /// `agg_share.len() != κ_kahe`, or ciphertext length != μ_kahe·l.
    InconsistentKappa(ServerId),
}

pub fn aggregate_and_decrypt(
    pp: &ProtocolParams,
    canonical: &[ClientId],
    client_entries: &[(ClientId, ClientBulletinEntry)],
    server_outputs: &[ServerBulletinEntry],
) -> Result<Vec<KahePoly>, VerifyError>;
```

`aggregate_and_decrypt` requires at least $t$ distinct, in-range server outputs (`BadServerCoverage`), enforces every `ServerBulletinEntry.clients` matches `canonical` (`InconsistentCanonical`), and checks `agg_share == agg_open.s()` per server (`ShareOpeningMismatch`). It runs **one** CS verification per server (the $μ_{cs} = κ_{kahe}$ packing) and recovers the aggregate KAHE key by Lagrange interpolation across the first $t$ servers, componentwise over the $κ_{kahe}$ components (then lifting each from the CS ring to the KAHE ring). Tampered Merkle openings reject via `InvalidServerOpening`.

### `codec` — bytes ↔ `KahePoly`

Two flavors. `encode`/`decode` carry a 4-byte little-endian length header so the decoded length is exact for a single message; safe for round-trips, **not** safe to sum across clients (the per-coefficient header sums multiply by N and overflow). `encode_raw`/`decode_raw` skip the header and return a fixed-size buffer (`polys.len() * 4096` bytes) — use this when multiple clients write into disjoint slots of a fixed buffer and the recovered sum is decoded as a single layout.

```rust
#[derive(Debug, PartialEq)]
pub enum CodecError {
    Empty,
    LengthOverflow { claimed: u32, available: usize },
    CoeffOutOfRange { index: usize, value: i32 },
}

pub fn encode    (bytes: &[u8])       -> Vec<KahePoly>;
pub fn decode    (polys: &[KahePoly]) -> Result<Vec<u8>, CodecError>;
pub fn encode_raw(bytes: &[u8])       -> Vec<KahePoly>;
pub fn decode_raw(polys: &[KahePoly]) -> Result<Vec<u8>, CodecError>;

// Beacon-driven slot allocation over a fixed shared message vector.
pub fn beacon(rands: &[u16]) -> u16;                                             // hash of all rands
pub fn allocate(reservations: &[(u16, usize)], beacon: u16, vector_bytes: usize) // (rand, size)
    -> Vec<Option<usize>>;                                                       // per-reservation offset
pub fn encode_at(offset: usize, vector_bytes: usize, payload: &[u8]) -> Vec<KahePoly>;
pub fn decode_ranges(plain: &[KahePoly], ranges: &[(usize, usize)]) -> Result<Vec<Vec<u8>>, CodecError>;
```

Layout for both: 2 bytes per coefficient (little-endian `u16`), `N = 2048` coefficients per `KahePoly`, 4096 bytes per poly. Coefficients land in `[0, 65536) ⊂ [0, q_kahe)` so a fresh single-message encode/decode is exact. Decoding a sum is meaningful only when the application controls the encoding so per-coefficient sums stay below `t = 65536` (symbols are mod-`t`).

**Schedule-and-message.** A two-round anonymous-broadcast schedule can share the (dominant) opening/key wire between rounds. Each client's plaintext is **one joint vector** `[reservation region ‖ message region]`, encrypted under **one** KAHE key with **one** Shamir-share + CS-opening set — the reservation region carries a tiny MSE `(rand, size)` token, the message region carries the payload placed at the client's allocated offset. `beacon` (hash of every decoded `rand`) fixes a per-round starting point so no client can grief its position; `allocate` visits reservations in `rand` order from the beacon and packs them contiguously into `vector_bytes` (2-byte aligned; `None` on overflow or a `rand` tie). Because the two regions ride one key, the openings — which dominate the wire (see the scaling bench) — are paid once instead of twice, and the message region skips the MSE's ~3× coding blowup. `encode_at`/`decode_ranges` write/read disjoint byte ranges into a fixed buffer that decode per-range under summation (no length header). Round `r`'s message is placed by round `r-1`'s decoded allocation, so the two regions of a joint ciphertext pipeline one round apart.

### `mse` — additive multi-set encoding (paper §3 Fig. 1)

The application payload. Matrices $(C, K_0…K_{L-1}, V_0…V_{ξ-1})$ of shape $γ × δ$ over $Z_t$ (`t = T_MODULUS_DEFAULT = 2^16`, the KAHE plaintext modulus). `L = K_LIMBS = 2` base-$t$ randomness limbs, so per-element randomness $r ∈ Z_{t^2} = Z_{2^32}$; `ξ = payload_symbols` symbols per element, each in $Z_t$, so one insert rides a $ξ · 16$-bit message. C and K are shared across the V symbols. Insert one element with fresh $r ← Z_{t^L}$: split $r$ into limbs $r_ℓ$, and for each row $i ∈ [γ]$ compute `j := PRF(prf_key, (i, r)) mod row_delta(i)` and $C[i,j] += 1$, $K_ℓ[i,j] += r_ℓ$, $V_s[i,j] += x_s$. `Decode` peels cells with $C = 1$, reconstructs $r = Σ r_ℓ·t^ℓ$, reads the $ξ$-symbol payload, emits it, and subtracts the element's contribution from every row via the PRF. Theorem 3 correctness: $2^{-(γ-2) log ρ} + negl(λ)$. PRF is SHA-256 keyed by `prf_key`. `RowLayout` lets later rows shrink geometrically for a smaller structure.

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum RowLayout {
    Uniform,
    Geometric { shrink: f64 },   // row i bucket count *= shrink^i (floored at 1)
}

#[derive(Clone, Debug, PartialEq)]
pub struct MseParams {
    pub gamma: usize,
    pub delta: usize,             // buckets in row 0
    pub payload_symbols: usize,   // ξ
    pub row_layout: RowLayout,
    pub prf_key: [u8; 32],
}
impl MseParams {
    pub fn new(gamma: usize, delta: usize, payload_symbols: usize, prf_key: [u8; 32]) -> Self;  // Uniform; asserts γ≥2, δ≥1, ξ≥1
    pub fn with_layout(gamma: usize, delta: usize, payload_symbols: usize, row_layout: RowLayout, prf_key: [u8; 32]) -> Self;
    pub fn payload_symbols_for_bits(bits: usize) -> usize;   // ⌈bits / BITS_PER_SYMBOL⌉
    pub fn row_delta(&self, row: usize) -> usize;
    pub fn row_offset(&self, row: usize) -> usize;
    pub fn total_cells(&self) -> usize;
    pub fn total_scalars(&self) -> usize;                    // (1 + K_LIMBS + ξ) · total_cells
    pub fn r_space(&self) -> u64;                            // t^K_LIMBS
}

pub const K_LIMBS: usize = 2;
pub const BITS_PER_SYMBOL: usize = 16;   // log2(t)

#[derive(Clone, Debug, PartialEq)]
pub struct MseEncoding {
    pub params: MseParams,
    pub c: Vec<i32>,            // γ × δ counters
    pub k: Vec<Vec<i32>>,       // K_LIMBS limbs, each total_cells entries
    pub v: Vec<Vec<i32>>,       // ξ symbols, each total_cells entries
}

#[derive(Debug, PartialEq)]
pub enum MseError {
    PeelStalled,
    ParamsMismatch,
    PayloadArity,               // payload.len() != params.payload_symbols
}

impl MseEncoding {
    pub fn new(params: MseParams) -> Self;
    pub fn insert<R: Rng>(&mut self, rng: &mut R, payload: &[i32]);   // payload.len() == ξ
    pub fn insert_with_r(&mut self, payload: &[i32], r: u64);
    pub fn add_assign(&mut self, other: &Self) -> Result<(), MseError>;
    pub fn decode(&self) -> Result<Vec<Vec<i32>>, MseError>;          // recovered payload tuples, sorted

    pub fn n_polys(params: &MseParams) -> usize;
    pub fn pack(&self) -> Vec<KahePoly>;
    pub fn unpack(params: &MseParams, polys: &[KahePoly]) -> Self;
}
```

`pack` flattens $(C, K_0…K_{L-1}, V_0…V_{ξ-1})$ row-major into a coefficient stream in that order. `unpack` is the inverse, lifting each `KahePoly`'s coefficients to canonical signed reps. Pointwise sum of packed encodings unpacks to the multiset union — this is the property the protocol exploits to carry an MSE end-to-end. [tests/mse_e2e.rs](tests/mse_e2e.rs)`::mse_recovers_through_panetiere` is the executable spec.

## Security properties

- **Binding** (commitment): given `comm`, an adversary cannot produce a different `(r', s', path')` that verifies — Module-SIS hardness on the BDLOP leaf (CS ring) and Ring-SIS hardness on every internal Merkle node (HVC ring).
- **Hiding** (commitment, single opening): the BDLOP leaf $(a^T r, B r + s)$ is statistically uniform over $R_{q_{cs}}^{1+μ_{cs}}$ by the leftover hash lemma when $r$ is short and the matrix $(a | B)$ has enough min-entropy.
- **Strong hiding** (sum of openings): pointwise sums of low-norm $r$s stay low-norm with margin; the linear hash + decomposed-path representation makes summed openings verify against the summed root; per-client $s_j$ remains hidden under any subset of summands.
- **Threshold liveness**: any $t$ honest servers' `agg_share`s suffice to recover $Σ sk_j$ via Lagrange; up to $n − t$ servers can be offline or corrupted.
- **Integrity**: `aggregate_and_decrypt` rejects tampered Merkle openings (`InvalidServerOpening`), tampered key shares (`ShareOpeningMismatch`), inconsistent canonical sets (`InconsistentCanonical`), and shape mismatches (`InconsistentKappa`); honest output is $Σ m_j$ over canonical clients, nothing else.
- **Anonymity**: assumes at least one honest server refuses to decrypt any set other than the canonical one. **This is NOT enforced by this PoC** — there is no consensus / canonical-set-enforcement module; a `canonical` slice is passed in by the caller and `aggregate_and_decrypt` only checks that every server agrees on it, not that it is the "right" set. Enforcing canonicality is out of scope here.

## Parameters and limits

Three rings, all $Z_q[x]/(x^N+1)$ with `N = 2048` and `q ≡ 1 mod 2N` (from chipmunk):
- **HVC** (`HVCPoly`, Merkle internal hash nodes): `q_hvc = 40_961` (~15.3 bits), `HVC_WIDTH = 3` decomposed polys per node, `ZETA = 34` decomposition base bound.
- **CS** (`CsPoly`, BDLOP commitment leaf + Shamir sharing + CS-side aggregation): `q_cs = 147_457` (~17.2 bits), `q_cs/2 = 73_728`.
- **KAHE** (`KahePoly`, KAHE encryption + codec + MSE): `q_kahe = 271_163_393` (~28 bits).

Tree height = `⌈log2(block_size · n_servers)⌉`, where `block_size = (1 + μ_cs).next_power_of_two()`.

KAHE defaults (`Kahe::setup`): `(μ, κ, l) = (1, 1, 1)`, `σ_s = σ_e = 15.72`, `t = 2^16` — sized for ~128-bit RLWE at N=2048 with noise budget `t·8σ_e·√ρ + ρ·t/2 < q_kahe/2` holding for ρ ≲ 240. The scaling bench compares two operating points: **$(μ=16, κ=31)$** with `block_size = 32` (best 1 MB throughput) and **$(μ=8, κ=15)$** with `block_size = 16` (best per-round latency — `1+κ` exactly hits the lower power-of-two).

CS defaults (`HidingMerkleCommitment::setup`): `μ_cs = 1, κ_cs = 5, β_cs = 122, r_bound = 36600, β_agg_hvc = 300·ζ = 10200`. The protocol couples `μ_cs := κ_kahe` so a single CS pipeline carries the entire share-vector.

Threshold: `t = max(⌊n/2⌋ + 1, n − 2)` by default. Per-opening data size grows as `(2·block_size − 2)·HVC_WIDTH + 2·stored_path_len·HVC_WIDTH` HVC ring elements, plus `κ_cs + μ_cs` CS ring elements (`r ‖ s`).

## Tests, benches, demos

```
cargo test                              # all unit + integration tests
cargo bench --bench protocol            # criterion micro-benches per stage at small parameters
cargo bench --bench scaling             # (S, N) cell × (μ_kahe, κ_kahe) variants
                                        #   BENCH_BUDGET_SECS=N to extend (default 300)
cargo run --release --example demo      # 6 clients × 128-byte slots over a 1024-byte buffer
```

[tests/end_to_end.rs](tests/end_to_end.rs_`::end_to_end_recovers_sum` is the canonical executable spec. Other coverage in that file: `slot_mode_disjoint_clients_recover_each_payload`, `slot_mode_8kb_message_multi_poly`, `tampered_agg_share_rejected`, `high_norm_r_rejected_in_protocol`, `recovers_from_t_of_n_servers`. [tests/mse_e2e.rs](tests/mse_e2e.rs)`::mse_recovers_through_panetiere` carries an MSE multiset end-to-end.

### Scaling bench

> **Caveat:** the measured millisecond/percentage values in this section predate the move to three rings at `N = 2048`; treat them as historical and re-run the bench for current numbers. The `β` column in the sample table below is a stale KAHE secret-key bound that no longer exists (the key is now discrete-Gaussian, σ_s = 15.72).

Reported wall times **assume parallel deployment**: clients run in parallel on N machines, servers in parallel on S machines, the verifier is one party. Per-round wall = `client_round + server_round + verify_round`. Sub-stages within each role are sequential on the same machine. Each round ships `μ_kahe · (bytes-per-poly)` bytes of broadcast payload; the bench extrapolates a 1 MB total over the corresponding round count. Default cell: `(S, N) = (8, 300)` × the two `(μ_kahe, κ_kahe)` points above.

Sample on an 8-core machine (Intel Core Ultra 7 155H, AVX2), `RAYON_NUM_THREADS=8`, `chipmunk_code` built with `--features fast-ntt` (Barrett + AVX2), default `parallel` feature off:

```
S= 8 N= 300 (μ= 16, κ= 31, β= 1024) | client  25.5ms server  10.9ms verify  18.2ms  per-round  54.6ms  ⇒ 1MB (64 rounds)  3.49s  (0.300 MB/s)
S= 8 N= 300 (μ=  8, κ= 15, β= 2048) | client  12.2ms server   6.6ms verify   9.3ms  per-round  28.1ms  ⇒ 1MB (128 rounds)  3.60s  (0.291 MB/s)
```

(Numbers depend on machine; run-to-run noise ≈ 15–20 %.) The two cells trade per-round latency against round count:

- **(μ=16, κ=31, β=1024)** — 64 rounds for 1 MB at 54.6 ms each → **0.30 MB/s**, best 1 MB throughput.
- **(μ=8, κ=15, β=2048)** — `block_size` halves (16 vs 32) because `1 + κ = 16` exactly hits the lower power-of-two; per-round drops to 28 ms (~half), but 128 rounds undo most of that on 1 MB. Pick this if the message you're sending is small (single round) and end-to-end latency matters more than throughput.

A wider sweep over `(μ, κ, β) ∈ {4..24} × {64..4096}` confirmed these as the only two non-dominated cells: `block_size` cliffs at `κ ≤ 15`, `κ ≤ 31` are the dominant variable, and within a `block_size` class higher β (lower κ_min) buys ≤ 2 % per-round.

#### Schedule-and-message

[benches/scaling.rs](benches/scaling.rs) also sweeps `AppCodec::Scheduled { message_bytes }` — the joint schedule-and-message
round (`codec` docs above), measured directly as one KAHE round over `[reservation IBLT ‖ message
vector]`. The reservation region is a tiny `(rand, size)` MSE token per client; the message region
is the shared `message_bytes` vector split into an equal slot per active client. Because both ride
one key, the opening/commitment/server-entry wire is counted once, and the message region skips the
MSE's ~3× coding blowup — so at comparable useful throughput it roughly halves the ciphertext and
lifts bandwidth efficiency vs. carrying the same payload through a single MSE round.

#### Where time is spent

CPU profile at `(S=8, N=300, μ=16, κ=31, β=1024)`, fast-ntt on, after the cache fixes below:

```
29.6%  ntt_stages_scalar              forward NTT, scalar tail (ht ∈ {4,2,1})
14.5%  ntt_avx2_dispatch              forward NTT, AVX2 (ht ≥ 8)
11.0%  pointwise_mac_avx2_dispatch    NTT-domain inner product
 7.9%  panetiere::sss::scalar_mul      Shamir Lagrange interp scalar mult
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
