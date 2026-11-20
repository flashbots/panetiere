//! Public bulletin (in-memory v1).

use std::sync::Mutex;

use crate::cs::{
    add_unpacked_poly64, aggregated_opening_pack_bounds, pack_cs_shares, pack_poly64,
    poly_packed_len, poly_packed_len64, unpack_cs_shares, unpack_poly64, Commitment, Opening,
    PackedOpening,
};
use crate::hvc_sum::reduce_hvc_sum;
use crate::protocol::{ClientId, ProtocolParams, ServerId};
use crate::rings::{center_i32, center_i64};
use crate::{CsPoly, KahePoly, CS_MODULUS, KAHE_MODULUS, N as POLY_N};
use chipmunk_code::HVC_MODULUS;

#[derive(Clone)]
pub struct ClientBulletinEntry {
    pub(crate) ctxt: Vec<KahePoly>,
    pub(crate) comm: Commitment,
}

impl ClientBulletinEntry {
    /// The commitment to the client's key shares, which every server checks
    /// its opening against.
    pub fn commitment(&self) -> &Commitment {
        &self.comm
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            2 + self.ctxt.len() * poly_packed_len64(KAHE_MODULUS) + poly_packed_len(HVC_MODULUS),
        );
        out.extend_from_slice(&(self.ctxt.len() as u16).to_le_bytes());
        for p in &self.ctxt {
            pack_poly64(p.coeffs(), KAHE_MODULUS, &mut out);
        }
        out.extend_from_slice(&self.comm.to_bytes());
        out
    }

    pub(crate) fn packed_len(mu_kahe: usize) -> usize {
        2 + mu_kahe * poly_packed_len64(KAHE_MODULUS) + poly_packed_len(HVC_MODULUS)
    }

    pub fn try_from_bytes(bytes: &[u8]) -> Option<Self> {
        let n_ctxt = u16::from_le_bytes([*bytes.first()?, *bytes.get(1)?]) as usize;
        let kahe_len = poly_packed_len64(KAHE_MODULUS);
        let hvc_len = poly_packed_len(HVC_MODULUS);
        if bytes.len() != 2 + n_ctxt * kahe_len + hvc_len {
            return None;
        }
        let mut ctxt = Vec::with_capacity(n_ctxt);
        let mut start = 2;
        for _ in 0..n_ctxt {
            let (coeffs, next) = unpack_poly64(bytes, start, KAHE_MODULUS)?;
            start = next;
            ctxt.push(KahePoly::from_coeffs(coeffs));
        }
        let comm = Commitment::from_bytes(&bytes[start..])?;
        Some(ClientBulletinEntry { ctxt, comm })
    }
}

/// Why a post cannot be added to a [`PostSum`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PostError {
    /// The post does not parse.
    Malformed,
    /// The post's ciphertext does not have exactly `μ` ring elements.
    WrongCiphertextLength,
}

/// Most posts one [`PostSum`] takes: every coefficient added is below
/// `4q + 2` in absolute value, and the sums must stay in `i64`.
const MAX_POSTS: usize = (i64::MAX / (4 * KAHE_MODULUS + 2)) as usize;

/// A running sum of client posts, ciphertexts and commitments alike. Both are
/// added unreduced and reduced once, in [`PostSum::finish`]. Posts can be
/// added from their wire bytes, without building their polynomials
/// ([`PostSum::add`]), or already parsed ([`PostSum::add_entry`]); a post that
/// cannot be added leaves the sum unchanged. Sums over disjoint sets of posts
/// combine with [`PostSum::merge`], in any order.
#[derive(Clone)]
pub(crate) struct PostSum {
    mu: usize,
    /// Empty until the first post, so an unused sum costs nothing.
    ctxt: Vec<[i64; POLY_N]>,
    comm: Vec<i64>,
    posts: usize,
}

impl PostSum {
    /// An empty sum of posts whose ciphertexts have `mu` ring elements.
    pub fn new(mu: usize) -> Self {
        Self {
            mu,
            ctxt: Vec::new(),
            comm: Vec::new(),
            posts: 0,
        }
    }

    /// Add a post given as wire bytes ([`ClientBulletinEntry::to_bytes`]). The
    /// ciphertext is unpacked straight into the sum.
    pub fn add(&mut self, post: &[u8]) -> Result<(), PostError> {
        let (Some(&lo), Some(&hi)) = (post.first(), post.get(1)) else {
            return Err(PostError::Malformed);
        };
        if u16::from_le_bytes([lo, hi]) as usize != self.mu {
            return Err(PostError::WrongCiphertextLength);
        }
        if post.len() != ClientBulletinEntry::packed_len(self.mu) {
            return Err(PostError::Malformed);
        }
        let ctxt_end = 2 + self.mu * poly_packed_len64(KAHE_MODULUS);
        let comm = Commitment::from_bytes(&post[ctxt_end..]).ok_or(PostError::Malformed)?;
        self.start_post();
        let mut start = 2;
        for (k, acc) in self.ctxt.iter_mut().enumerate() {
            match add_unpacked_poly64(post, start, KAHE_MODULUS, acc) {
                Some(next) => start = next,
                None => {
                    // A field outside its bound: the decoded values of
                    // polynomials 0..=k are already in the sum. Take them
                    // out again, so a rejected post leaves the sum unchanged.
                    self.undo_ctxt(post, k);
                    return Err(PostError::Malformed);
                }
            }
        }
        self.add_commitment(&comm);
        Ok(())
    }

    /// Subtract the decoded values of the post's first `k + 1` ciphertext
    /// polynomials, which [`Self::add`] added before it found a malformed
    /// one, and forget the post.
    #[cold]
    fn undo_ctxt(&mut self, post: &[u8], k: usize) {
        let mut start = 2;
        for acc in &mut self.ctxt[..=k] {
            let mut added = [0i64; POLY_N];
            let _ = add_unpacked_poly64(post, start, KAHE_MODULUS, &mut added);
            for (x, y) in acc.iter_mut().zip(added) {
                *x -= y;
            }
            start += poly_packed_len64(KAHE_MODULUS);
        }
        self.posts -= 1;
    }

    /// Add a parsed post.
    pub fn add_entry(&mut self, post: &ClientBulletinEntry) -> Result<(), PostError> {
        if post.ctxt.len() != self.mu {
            return Err(PostError::WrongCiphertextLength);
        }
        self.start_post();
        for (acc, poly) in self.ctxt.iter_mut().zip(&post.ctxt) {
            for (x, &c) in acc.iter_mut().zip(poly.coeffs()) {
                *x += center_i64(c, KAHE_MODULUS);
            }
        }
        self.add_commitment(&post.comm);
        Ok(())
    }

    /// Add a sum over other posts into this one.
    pub fn merge(&mut self, other: PostSum) {
        assert_eq!(self.mu, other.mu, "PostSum::merge: different μ");
        if other.posts == 0 {
            return;
        }
        if self.posts == 0 {
            *self = other;
            return;
        }
        self.posts += other.posts;
        assert!(
            self.posts <= MAX_POSTS,
            "PostSum: too many posts to sum in i64"
        );
        for (a, b) in self.ctxt.iter_mut().zip(&other.ctxt) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        for (x, y) in self.comm.iter_mut().zip(&other.comm) {
            *x += y;
        }
    }

    /// The summed ciphertext and commitment, reduced. An empty sum is `μ`
    /// zero ring elements and the zero commitment.
    pub fn finish(self) -> (Vec<KahePoly>, Commitment) {
        if self.posts == 0 {
            return (
                vec![KahePoly::default(); self.mu],
                Commitment {
                    root: reduce_hvc_sum([0; POLY_N]),
                },
            );
        }
        let ctxt = self
            .ctxt
            .into_iter()
            .map(|coeffs| {
                let mut p = KahePoly::from_coeffs(coeffs);
                p.normalize();
                p
            })
            .collect();
        let comm: [i64; POLY_N] = self.comm.try_into().expect("N coefficients");
        (
            ctxt,
            Commitment {
                root: reduce_hvc_sum(comm),
            },
        )
    }

    fn start_post(&mut self) {
        if self.posts == 0 {
            self.ctxt = vec![[0; POLY_N]; self.mu];
            self.comm = vec![0; POLY_N];
        }
        self.posts += 1;
        assert!(
            self.posts <= MAX_POSTS,
            "PostSum: too many posts to sum in i64"
        );
    }

    fn add_commitment(&mut self, comm: &Commitment) {
        for (x, &c) in self.comm.iter_mut().zip(comm.root.coeffs()) {
            *x += center_i32(c, HVC_MODULUS) as i64;
        }
    }
}

#[derive(Clone)]
pub struct ServerBulletinEntry {
    pub(crate) server_id: ServerId,
    pub(crate) clients: Vec<ClientId>,
    pub(crate) agg_open: Opening,
    pub(crate) agg_share: CsPoly,
}

impl ServerBulletinEntry {
    pub fn server_id(&self) -> ServerId {
        self.server_id
    }

    /// The client set this server summed over.
    pub fn clients(&self) -> &[ClientId] {
        &self.clients
    }

    /// Wire form: `server_id: u32 LE ‖ |clients|: u32 LE ‖ clients: u32 LE each
    /// ‖ agg_open ‖ agg_share`. The aggregated opening is bit-packed against
    /// the bounds for `|clients|` summed openings
    /// (`aggregated_opening_pack_bounds`) and carries its own header; the
    /// share is one packed CS polynomial, so it has a fixed length and comes
    /// last.
    pub fn to_bytes(&self, pp: &ProtocolParams) -> Vec<u8> {
        let (r_b, s_b, t_b) = aggregated_opening_pack_bounds(&pp.cs, self.clients.len() as u32);
        let opening = self.agg_open.pack(r_b, s_b, t_b).to_bytes();
        let share = pack_cs_shares(std::slice::from_ref(&self.agg_share));
        let mut out = Vec::with_capacity(8 + 4 * self.clients.len() + opening.len() + share.len());
        out.extend_from_slice(&self.server_id.0.to_le_bytes());
        out.extend_from_slice(&(self.clients.len() as u32).to_le_bytes());
        for c in &self.clients {
            out.extend_from_slice(&c.0.to_le_bytes());
        }
        out.extend_from_slice(&opening);
        out.extend_from_slice(&share);
        out
    }

    /// Inverse of [`Self::to_bytes`]. Checks lengths and the opening's header
    /// before allocating; whether the opening has the shape the parameters
    /// require, and whether it verifies, is left to the recipient.
    pub fn try_from_bytes(bytes: &[u8]) -> Option<Self> {
        let word = |i: usize| -> Option<u32> {
            Some(u32::from_le_bytes(
                bytes.get(4 * i..4 * i + 4)?.try_into().ok()?,
            ))
        };
        let server_id = ServerId(word(0)?);
        let n_clients = word(1)? as usize;
        let share_len = poly_packed_len(CS_MODULUS);
        let clients_end = 8usize.checked_add(n_clients.checked_mul(4)?)?;
        let opening_end = bytes.len().checked_sub(share_len)?;
        if opening_end < clients_end {
            return None;
        }
        let clients = (0..n_clients)
            .map(|i| word(2 + i).map(ClientId))
            .collect::<Option<Vec<_>>>()?;
        let packed = PackedOpening::from_bytes(&bytes[clients_end..opening_end])?;
        let agg_open = Opening::from_packed(&packed).ok()?;
        let agg_share = unpack_cs_shares(&bytes[opening_end..], 1)?.pop()?;
        Some(Self {
            server_id,
            clients,
            agg_open,
            agg_share,
        })
    }
}

#[derive(Default)]
pub struct InMemoryBulletin {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    clients: Vec<(ClientId, ClientBulletinEntry)>,
    servers: Vec<ServerBulletinEntry>,
    canonical: Option<Vec<ClientId>>,
}

impl InMemoryBulletin {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn publish_client(&self, id: ClientId, p: ClientBulletinEntry) {
        self.inner.lock().unwrap().clients.push((id, p));
    }

    pub fn publish_server(&self, p: ServerBulletinEntry) {
        self.inner.lock().unwrap().servers.push(p);
    }

    pub fn publish_canonical(&self, set: Vec<ClientId>) {
        self.inner.lock().unwrap().canonical = Some(set);
    }

    pub fn clients(&self) -> Vec<(ClientId, ClientBulletinEntry)> {
        self.inner.lock().unwrap().clients.clone()
    }

    pub fn servers(&self) -> Vec<ServerBulletinEntry> {
        self.inner.lock().unwrap().servers.clone()
    }

    pub fn canonical(&self) -> Option<Vec<ClientId>> {
        self.inner.lock().unwrap().canonical.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chipmunk_code::HVCPoly;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn client_bulletin_entry_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        for mu_kahe in [1usize, 2, 4] {
            let ctxt: Vec<KahePoly> = (0..mu_kahe)
                .map(|_| KahePoly::rand_poly(&mut rng))
                .collect();
            let entry = ClientBulletinEntry {
                ctxt,
                comm: Commitment {
                    root: HVCPoly::rand_poly(&mut rng),
                },
            };
            let bytes = entry.to_bytes();
            let back = ClientBulletinEntry::try_from_bytes(&bytes).unwrap();
            assert_eq!(back.ctxt, entry.ctxt);
            assert_eq!(back.comm.root, entry.comm.root);
        }
        assert!(ClientBulletinEntry::try_from_bytes(&[0u8]).is_none());
    }

    fn rand_entry(rng: &mut ChaCha20Rng, mu: usize) -> ClientBulletinEntry {
        ClientBulletinEntry {
            ctxt: (0..mu).map(|_| KahePoly::rand_poly(rng)).collect(),
            comm: Commitment {
                root: HVCPoly::rand_poly(rng),
            },
        }
    }

    fn reference_sum(entries: &[ClientBulletinEntry]) -> (Vec<KahePoly>, Commitment) {
        let ctxts: Vec<&[KahePoly]> = entries.iter().map(|e| e.ctxt.as_slice()).collect();
        let comms: Vec<&Commitment> = entries.iter().map(|e| &e.comm).collect();
        (
            crate::kahe::Kahe::agg_ctxt_refs(&ctxts),
            crate::cs::HidingMerkleCommitment::sum_commitment_refs(&comms),
        )
    }

    /// Adding posts from their bytes, adding them parsed, and merging partial
    /// sums in either order all give the sum of the parsed posts.
    #[test]
    fn post_sum_from_bytes_parsed_and_merged_agree() {
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        let entries: Vec<_> = (0..5).map(|_| rand_entry(&mut rng, 3)).collect();
        let (want_ctxt, want_comm) = reference_sum(&entries);

        let mut bytes = PostSum::new(3);
        let mut parsed = PostSum::new(3);
        for e in &entries {
            bytes.add(&e.to_bytes()).unwrap();
            parsed.add_entry(e).unwrap();
        }
        let mut front = PostSum::new(3);
        let mut back = PostSum::new(3);
        for (i, e) in entries.iter().enumerate() {
            let half = if i < 2 { &mut front } else { &mut back };
            half.add(&e.to_bytes()).unwrap();
        }
        let mut merged_ab = front.clone();
        merged_ab.merge(back.clone());
        let mut merged_ba = back;
        merged_ba.merge(front);
        let mut with_empty = PostSum::new(3);
        with_empty.merge(merged_ab.clone());
        with_empty.merge(PostSum::new(3));

        for sum in [bytes, parsed, merged_ab, merged_ba, with_empty] {
            assert_eq!(sum.posts, 5);
            let (ctxt, comm) = sum.finish();
            assert_eq!(ctxt, want_ctxt);
            assert_eq!(comm.root, want_comm.root);
        }
    }

    /// An empty sum is μ zero ring elements and the zero commitment.
    #[test]
    fn empty_post_sum_is_zero() {
        let (ctxt, comm) = PostSum::new(2).finish();
        assert_eq!(ctxt, vec![KahePoly::default(); 2]);
        assert!(comm.root.coeffs().iter().all(|&c| c == 0));
    }

    /// A post that cannot be added is reported and leaves the sum unchanged.
    #[test]
    fn post_sum_rejects_bad_posts_and_stays_unchanged() {
        let mut rng = ChaCha20Rng::from_seed([6u8; 32]);
        let good = rand_entry(&mut rng, 2);
        let mut sum = PostSum::new(2);
        sum.add(&good.to_bytes()).unwrap();
        let (want_ctxt, want_comm) = sum.clone().finish();

        let bytes = good.to_bytes();
        let longer = rand_entry(&mut rng, 3);
        for (post, why) in [
            (&bytes[..bytes.len() - 1], PostError::Malformed),
            (&[2u8][..], PostError::Malformed),
            (&longer.to_bytes()[..], PostError::WrongCiphertextLength),
        ] {
            assert_eq!(sum.add(post), Err(why));
        }
        let mut extended = bytes.clone();
        extended.push(0);
        assert_eq!(sum.add(&extended), Err(PostError::Malformed));
        // A ciphertext field above q - 1, in the second polynomial, after the
        // first has been added: all ones in the 46 bits of its coefficient 5.
        let mut non_canonical = bytes.clone();
        let bits = 8 * poly_packed_len64(KAHE_MODULUS) / POLY_N;
        let at = 8 * (2 + poly_packed_len64(KAHE_MODULUS)) + 5 * bits;
        for k in at..at + bits {
            non_canonical[k / 8] |= 1 << (k % 8);
        }
        assert_eq!(sum.add(&non_canonical), Err(PostError::Malformed));
        assert_eq!(
            sum.add_entry(&longer),
            Err(PostError::WrongCiphertextLength)
        );

        assert_eq!(sum.posts, 1);
        let (ctxt, comm) = sum.finish();
        assert_eq!(ctxt, want_ctxt);
        assert_eq!(comm.root, want_comm.root);
    }
}
