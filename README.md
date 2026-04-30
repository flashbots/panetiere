# Flashnet

Flashnet is an anonymous broadcast protocol intended to allow (TEE) clients to send messages through a t-of-n (non-TEE) servers in a way that preserves anonymity of clients.  
This repository is an early proof of concept built using lattice-based additive homomorphic encryption and commitments from [Chipmunk: Better Synchronized Multi-Signatures from Lattices](https://eprint.iacr.org/2023/1820.pdf).  
Everything apart from this section is AI-generated. Do not use anywhere near production data.  

## Quick start

```sh
RAYON_NUM_THREADS=8 cargo test -j 8                                    # all unit + integration tests
RAYON_NUM_THREADS=8 cargo bench -j 8 --bench scaling                   # progressive scaling bench (~30s)
RAYON_NUM_THREADS=8 cargo run -j 8 --release --example demo            # slot-mode broadcast demo
```

`bench.sh` and `scripts/run_demo.sh` are thin wrappers around the above. The crate path-deps a sister crate `chipmunk_code` for the lattice primitives (Ring-SIS hash, dynamic-height Merkle tree, NTT polynomial multiplication).

## Repository layout

```
src/
  lib.rs                    re-exports
  kahe.rs                   key-additive homomorphic encryption (KAHE) trait + RingOtp
  sss.rs                    secret sharing trait + AdditiveSharing (n-of-n)
  cs.rs                     commitment scheme trait + HidingMerkleCommitment
  bulletin.rs               in-memory typed broadcast store
  consensus.rs              canonical-client-set selector + FixedSet stub
  codec.rs                  bytes <-> HVCPoly (header-prefixed and fixed-buffer flavors)
  iblt.rs                   Invertible Bloom Lookup Table over Z_q^L + pack/unpack to HVCPoly
  protocol/{client,server,verify,message}.rs   round drivers + public verifier
tests/
  end_to_end.rs             core protocol integration tests
  iblt_e2e.rs               IBLT carried through the protocol end-to-end
benches/
  protocol.rs               per-stage micro-benches
  scaling.rs                progressive (S,N) grid scaling bench
examples/
  demo.rs                   slot-mode broadcast over byte messages
```

Module file layout matches the conceptual layering: lattice primitives at the bottom (chipmunk dep), then KAHE + SSS + CS, then bulletin + consensus, then protocol drivers, with codec and iblt as application-side encodings on top.

## Protocol overview

Symbols. `K` = KAHE key, `m` = client's plaintext message, `c` = ciphertext, `s_i` = SSS share for server i, `comm` = commitment to the share vector, `d_i` = opening for share index i. Subscript `j` denotes the j-th client.

```
Client j  (run_client_round):
  1. K_j     := KAHE.gen()
  2. c_j     := KAHE.enc(K_j, m_j)
  3. (s_j_1, ..., s_j_S)            := SSS.share(K_j, S)              # one share per server
  4. (comm_j, (d_j_1, ..., d_j_S))  := CS.commit(s_j_1, ..., s_j_S)
  5. publish (c_j, comm_j) to the public bulletin
  6. privately send (d_j_i, s_j_i) to server i

Server i  (run_server_round, for canonical client set chosen by consensus):
  7. agg_open_i  := CS.sum_openings([d_j_i for j in canonical])
     agg_share_i := SSS.recover([s_j_i for j in canonical])             # = sum, additively
  8. publish (agg_open_i, agg_share_i)

Verifier  (aggregate_and_decrypt):
  9a. summed_ctxt := KAHE.agg_ctxt([c_j for j in canonical])
      summed_comm := CS.sum_commitments([comm_j for j in canonical])
  9b. for each server i: assert CS.verify(summed_comm, agg_open_i)
                          assert agg_share_i == agg_open_i.s
  9c. agg_key := KAHE.agg_key([agg_share_i for i in 1..S])
  9d. return KAHE.dec(summed_ctxt, agg_key)             # = sum of m_j over canonical
```

The verifier output is the polynomial sum of all canonical clients' messages. How that sum gets interpreted is the application's choice (slot mode, IBLT decoding, custom encoding). The integration test `tests/end_to_end.rs::end_to_end_recovers_sum` is the executable spec.

## Modules

### `kahe` — key-additive homomorphic encryption

A KAHE scheme encrypts a message under a key and supports pointwise sums of both ciphertexts and keys, so the sum of ciphertexts decrypts (with the sum of keys) to the sum of messages. `RingOtp` is a one-time pad over `chipmunk_code::HVCPoly`: enc = `+`, dec = `-`, both aggregations are `Σ` mod q.

```rust
pub trait Kahe {
    type Key: Clone;
    type Message: Clone;
    type Ciphertext: Clone;

    fn gen<R: Rng>(rng: &mut R) -> Self::Key;
    fn enc(k: &Self::Key, m: &Self::Message) -> Self::Ciphertext;
    fn dec(c: &Self::Ciphertext, k: &Self::Key) -> Self::Message;
    fn agg_ctxt(cs: &[Self::Ciphertext]) -> Self::Ciphertext;
    fn agg_key (ks: &[Self::Key])         -> Self::Key;
}

pub struct RingOtp;

impl Kahe for RingOtp {
    type Key = HVCPoly;
    type Message = HVCPoly;
    type Ciphertext = HVCPoly;
    /* ... */
}
```

Each round must use a fresh key (standard OTP requirement, met by calling `gen` once per `run_client_round`).

### `sss` — secret sharing

Splits a secret into `n` shares whose pointwise sum recovers the secret. v1 ships additive n-of-n sharing; threshold (Shamir) is a follow-up. The trait shape lets the impl swap without touching protocol code.

```rust
pub trait Sss {
    type Secret: Clone;
    type Share: Clone;

    fn share<R: Rng>(rng: &mut R, secret: &Self::Secret, n: usize) -> Vec<Self::Share>;
    fn recover(shares: &[Self::Share]) -> Self::Secret;
}

pub struct AdditiveSharing;

impl Sss for AdditiveSharing {
    type Secret = HVCPoly;
    type Share  = HVCPoly;
    /* s_1..s_{n-1} random; s_n = secret − Σ s_i; recover = Σ s_i */
}
```

### `cs` — hiding commitment over a homomorphic Merkle tree

`Commit(shares)` produces a single Merkle-root commitment plus per-share openings. The leaf for share `i` is the pair `(a·R_i, b·R_i + s_i)` placed at tree positions `(2i, 2i+1)` of a chipmunk `Tree<HVCHash>`; `R_i` is a low-norm random vector providing hiding (leftover hash lemma). Sums of commitments and openings compose pointwise and verify against the summed root — the strong-hiding property the protocol relies on.

```rust
pub trait Cs {
    type Params;
    type Secret;
    type Commitment: Clone;
    type Opening: Clone;

    fn setup<R: Rng>(rng: &mut R, num_servers: usize) -> Self::Params;
    fn commit<R: Rng>(
        rng: &mut R,
        pp: &Self::Params,
        shares: &[Self::Secret],
    ) -> (Self::Commitment, Vec<Self::Opening>);
    fn verify(pp: &Self::Params, c: &Self::Commitment, o: &Self::Opening) -> bool;
    fn sum_commitments(cs: &[Self::Commitment]) -> Self::Commitment;
    fn sum_openings(os: &[Self::Opening]) -> Self::Opening;  // same `server_index`
}

pub struct CsParams {
    pub a: Vec<HVCPoly>,
    pub b: Vec<HVCPoly>,
    pub r_len: usize,
    pub r_bound: u32,
    pub r_half_weight: usize,
    pub hasher: HVCHash,
    pub n_servers: usize,
    pub n_leaves: usize,
}

#[derive(Clone)]
pub struct Commitment { pub root: HVCPoly }

#[derive(Clone)]
pub struct Opening {
    pub server_index: usize,
    pub r: Vec<HVCPoly>,
    pub s: HVCPoly,
    pub path_nodes: Vec<(Vec<HVCPoly>, Vec<HVCPoly>)>,   // decomposed pairs, top→bottom
    pub path_index: usize,
}

pub struct HidingMerkleCommitment;
impl Cs for HidingMerkleCommitment { /* ... */ }
```

`path_nodes` stores already-decomposed Merkle siblings (each side `HVC_WIDTH` low-norm polys). This is required because chipmunk's `decom_then_hash` is linear over decomposed inputs but non-linear over raw `(left, right)` pairs — so summing openings pointwise must happen in the decomposed representation. Verification recomputes the leaf pair from `(r, s)`, replaces `path_nodes.last()` with it, then walks `hash_separate_inputs` + `projection_r` up to the root.

### `bulletin` — in-memory broadcast store

Concrete typed bulletin: per-client public payload and per-server output, plus the canonical-set tag. v1 is `Arc<Mutex<...>>`; a networked impl can replace it without API churn.

```rust
#[derive(Clone)]
pub struct ClientPublic {
    pub ctxt: HVCPoly,
    pub comm: Commitment,
}

#[derive(Clone)]
pub struct ServerPublic {
    pub server_id: ServerId,
    pub clients: Vec<ClientId>,
    pub agg_open: Opening,
    pub agg_share: HVCPoly,
}

pub struct InMemoryBulletin { /* Arc<Mutex<{clients, servers, canonical}>> */ }

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

### `protocol` — round drivers + verifier

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ClientId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ServerId(pub u32);

pub struct ClientRound {
    pub client_id: ClientId,
    pub public: ClientPublic,
    pub private: Vec<(ServerId, Opening, <RingOtp as Kahe>::Key)>,
}

pub fn run_client_round<R: Rng>(
    rng: &mut R,
    pp: &<HidingMerkleCommitment as Cs>::Params,
    client_id: ClientId,
    message: <RingOtp as Kahe>::Message,
    servers: &[ServerId],
) -> ClientRound;

pub struct ServerInbox {
    pub server_id: ServerId,
    pub items: Vec<(ClientId, Opening, <RingOtp as Kahe>::Key)>,
}

pub fn run_server_round(
    inbox: &ServerInbox,
    canonical: &[ClientId],
) -> Option<ServerPublic>;

#[derive(Debug, PartialEq)]
pub enum VerifyError {
    MissingClient(ClientId),
    InvalidServerOpening(usize),
    ShareOpeningMismatch(usize),
    NoServers,
}

pub fn aggregate_and_decrypt(
    pp: &<HidingMerkleCommitment as Cs>::Params,
    canonical: &[ClientId],
    publics: &[(ClientId, ClientPublic)],
    server_outputs: &[ServerPublic],
) -> Result<HVCPoly, VerifyError>;
```

`aggregate_and_decrypt` is the public verifier of step 9. It cross-checks `agg_share == agg_open.s` (both are the same pointwise sum of per-client shares; mismatch implies tampering) and rejects via `ShareOpeningMismatch`. Tampered Merkle openings reject via `InvalidServerOpening`.

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

### `iblt` — Invertible Bloom Lookup Table over `Z_q^L`

The application payload. An IBLT is a multi-level bucket array supporting `insert`, `union via pointwise add`, and `recover` (queue-based peeling of pure cells). The flashnet IBLT lives natively in `Z_q^L` rather than a 385-bit prime field — buckets are length-`L` `Vec<i32>` with pointwise add/sub mod `q`, where `L = ⌈384 / base_bits⌉` and a 48-byte chunk splits into base-`2^base_bits` limbs. Correctness condition: `N_clients · (2^base_bits − 1) < q`.

```rust
pub const IBLT_N_LEVELS: usize = 4;
pub const IBLT_SHRINK: f64 = 0.75;
pub const IBLT_CHUNK_BYTES: usize = 48;          // 384 bits
pub const IBLT_CHUNK_BITS: usize = 384;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IbltParams {
    pub message_slots: u32,
    pub base_bits: u32,
}

impl IbltParams {
    pub fn limbs_per_chunk(&self) -> usize;
    pub fn level_size(&self, level: usize) -> usize;
    pub fn total_buckets(&self) -> usize;
    pub fn max_clients(&self) -> u32;            // floor(q / (2^base_bits - 1))
}

#[derive(Clone, Debug)]
pub struct IbltVector {
    pub params: IbltParams,
    pub chunks:   [Vec<Vec<i32>>; IBLT_N_LEVELS],
    pub counters: [Vec<i32>;      IBLT_N_LEVELS],
}

#[derive(Debug, PartialEq)]
pub enum IbltError {
    UnexpectedZeroCounter,
    ParamsMismatch,
}

impl IbltVector {
    pub fn new(params: IbltParams) -> Self;
    pub fn insert_chunk(&mut self, chunk: [u8; IBLT_CHUNK_BYTES]);
    pub fn add_assign(&mut self, other: &Self) -> Result<(), IbltError>;
    pub fn recover(&self) -> Result<Vec<[u8; IBLT_CHUNK_BYTES]>, IbltError>;

    pub fn n_polys(params: &IbltParams) -> usize;
    pub fn pack(&self) -> Vec<HVCPoly>;
    pub fn unpack(params: &IbltParams, polys: &[HVCPoly]) -> Self;
}

pub fn chunk_to_limbs(chunk: &[u8; IBLT_CHUNK_BYTES], base_bits: u32) -> Vec<i32>;
pub fn limbs_to_chunk(limbs: &[i32], base_bits: u32) -> [u8; IBLT_CHUNK_BYTES];
pub fn chunk_index   (chunk: &[u8; IBLT_CHUNK_BYTES], level: usize, items_in_level: usize) -> u64;
```

`pack` flattens all bucket-limbs (level/slot order) followed by all counters into a coefficient stream and packs into 512-coeff `HVCPoly`. `unpack` is the inverse, reading lifted coefficients back. The protocol carries the IBLT by running one flashnet round per packed poly; the verifier's recovered polys unpack into the union IBLT, which then peels.

## Security properties

- **Binding** (commitment): given `comm`, an adversary cannot produce a different `(R', s', path')` that verifies — Ring-SIS hardness on the leaf `(a·R, b·R + s)` and on every internal Merkle node.
- **Hiding** (commitment): a single opening reveals nothing about `s` because `R` is long and low-norm; the `(a·R, b·R + s)` leaf is statistically uniform by the leftover hash lemma.
- **Strong hiding** (sum of openings): pointwise sums of low-norm `R`s stay low-norm (with margin), and the linear hash makes summed paths verify; per-client `s_i` remains hidden under any subset of summands.
- **Integrity**: `aggregate_and_decrypt` rejects tampered Merkle openings (`InvalidServerOpening`) and tampered key shares (`ShareOpeningMismatch`); honest output is the polynomial sum of canonical-client messages, nothing else.
- **Anonymity**: holds as long as at least one honest server refuses to decrypt sets smaller than the canonical set (`consensus` invariant).

## Parameters and limits

Lattice ring (from chipmunk): `Z_q[x]/(x^N+1)` with `q = 202_753`, `N = 512`. `HVC_WIDTH = 3` decomposed polys per node, `ZETA = 29` decomposition base bound. Tree height = `log2(2 · n_servers)` rounded up.

Commitment-scheme placeholders pending parameter tuning: `r_len = 8`, `r_bound = 64`, `r_half_weight = 1`. These pin the leftover-hash margin and the per-coefficient sum budget for per-server openings; tighten before a real deployment.

IBLT max-clients constraint: `N_clients · (2^base_bits − 1) < q`.

| `base_bits` | `B`  | `L = ⌈384/b⌉` | max clients (`floor(q/(B−1))`) |
|------------:|-----:|--------------:|-------------------------------:|
|          12 | 4096 |            32 |                             49 |
|          10 | 1024 |            39 |                            197 |
|           8 |  256 |            48 |                            794 |
|           7 |  128 |            55 |                          1 596 |

`encode_raw` round-trips bytes; `encode` adds a 4-byte length header (don't sum across clients). `Kahe::Message` is `HVCPoly` (one ring element per round); multi-poly messages run multiple rounds per logical message.

## Tests, benches, demos

```
cargo test                              # 25 tests across lib + integration:
                                        #   kahe, sss, codec, cs (9), iblt (11), protocol (6), iblt_e2e (2)
cargo bench --bench protocol            # criterion micro-benches per stage at small parameters
cargo bench --bench scaling             # progressive (S, N) grid; 60s default budget
                                        #   BENCH_BUDGET_SECS=N to extend
cargo run --release --example demo      # 6 clients × 128-byte slots over a 1024-byte buffer;
                                        # prints recovered slots
```

The scaling bench walks (S, N) cells in increasing cost order: `(4|8|12) × (1|10|100|1000)`. Each row prints per-stage timings (client_round, server_round, verify) plus end-to-end-one-poly time and the corresponding MB/s for a 1 KB payload. Time-to-1MB at any cell = `e2e_ms × 1024 / 1000` seconds.

`tests/end_to_end.rs::end_to_end_recovers_sum` is the canonical executable spec: it runs the full 9-step flow with `|S|=4`, `N_clients=8` and asserts the recovered sum equals `Σ m_i`. `tests/iblt_e2e.rs::iblt_recovers_through_flashnet` is the IBLT-over-flashnet equivalent.
