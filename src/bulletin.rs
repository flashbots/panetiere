//! Public bulletin (in-memory v1).

use std::sync::Mutex;

use crate::cs::{pack_poly64, poly_packed_len, poly_packed_len64, unpack_poly64, Commitment, Opening};
use crate::protocol::{ClientId, NodeId, ServerId, SessionId};
use crate::rs::Share;
use crate::sig;
use chipmunk_code::{
    CsPoly, DgtNTTPoly, KahePoly, HVC_MODULUS, KAHE_MODULUS, N as POLY_N,
};

#[derive(Clone)]
pub struct ClientBulletinEntry {
    pub ctxt: Vec<KahePoly>,
    pub comm: Commitment,
}

impl ClientBulletinEntry {
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

    pub fn packed_len(mu_kahe: usize) -> usize {
        2 + mu_kahe * poly_packed_len64(KAHE_MODULUS) + poly_packed_len(HVC_MODULUS)
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let n_ctxt = u16::from_le_bytes([*bytes.first()?, *bytes.get(1)?]) as usize;
        let kahe_len = poly_packed_len64(KAHE_MODULUS);
        let hvc_len = poly_packed_len(HVC_MODULUS);
        if bytes.len() != 2 + n_ctxt * kahe_len + hvc_len {
            return None;
        }
        let mut ctxt = Vec::with_capacity(n_ctxt);
        let mut start = 2;
        for _ in 0..n_ctxt {
            let (coeffs, next) = unpack_poly64(bytes, start, KAHE_MODULUS);
            start = next;
            ctxt.push(KahePoly::from_coeffs(coeffs));
        }
        let comm = Commitment::from_bytes(&bytes[start..])?;
        Some(ClientBulletinEntry { ctxt, comm })
    }
}

#[derive(Clone)]
pub struct RsClientBulletinEntry {
    pub comm: Commitment,
    /// `A · ct` over the digest ring, `DIGEST_POLYS` elements.
    pub digest: Vec<DgtNTTPoly>,
    pub pubkey: [u8; sig::PUBKEY_LEN],
    pub sig: [u8; sig::SIG_LEN],
}

pub fn dgt_packed_len() -> usize {
    POLY_N * 8
}

fn pack_dgt(p: &DgtNTTPoly, out: &mut Vec<u8>) {
    for &c in p.coeffs().iter() {
        out.extend_from_slice(&c.to_le_bytes());
    }
}

fn unpack_dgt(bytes: &[u8], start: usize) -> (DgtNTTPoly, usize) {
    let mut raw = [0u64; POLY_N];
    for (i, r) in raw.iter_mut().enumerate() {
        let o = start + i * 8;
        *r = u64::from_le_bytes(bytes[o..o + 8].try_into().expect("8 bytes"));
    }
    (DgtNTTPoly::from_raw(&raw), start + dgt_packed_len())
}

impl RsClientBulletinEntry {
    pub fn signing_bytes(
        sid: &SessionId,
        client_id: ClientId,
        comm: &Commitment,
        digest: &[DgtNTTPoly],
    ) -> Vec<u8> {
        const DOMAIN: &[u8] = b"panetiere/rs-bulletin/v2";
        let mut out = Vec::with_capacity(DOMAIN.len() + 36 + poly_packed_len(HVC_MODULUS));
        out.extend_from_slice(DOMAIN);
        out.extend_from_slice(&sid.0);
        out.extend_from_slice(&client_id.0.to_le_bytes());
        out.extend_from_slice(&(digest.len() as u16).to_le_bytes());
        out.extend_from_slice(&comm.to_bytes());
        for p in digest {
            pack_dgt(p, &mut out);
        }
        out
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::packed_len(self.digest.len()));
        out.extend_from_slice(&self.comm.to_bytes());
        for p in &self.digest {
            pack_dgt(p, &mut out);
        }
        out.extend_from_slice(&self.pubkey);
        out.extend_from_slice(&self.sig);
        out
    }

    /// Independent of the message length — that is the point of the mode.
    pub fn packed_len(digest_polys: usize) -> usize {
        poly_packed_len(HVC_MODULUS)
            + digest_polys * dgt_packed_len()
            + sig::PUBKEY_LEN
            + sig::SIG_LEN
    }

    pub fn from_bytes(bytes: &[u8], digest_polys: usize) -> Option<Self> {
        if bytes.len() != Self::packed_len(digest_polys) {
            return None;
        }
        let hvc_len = poly_packed_len(HVC_MODULUS);
        let comm = Commitment::from_bytes(&bytes[..hvc_len])?;
        let mut digest = Vec::with_capacity(digest_polys);
        let mut start = hvc_len;
        for _ in 0..digest_polys {
            let (p, next) = unpack_dgt(bytes, start);
            digest.push(p);
            start = next;
        }
        let mut pubkey = [0u8; sig::PUBKEY_LEN];
        pubkey.copy_from_slice(&bytes[start..start + sig::PUBKEY_LEN]);
        let mut s = [0u8; sig::SIG_LEN];
        s.copy_from_slice(&bytes[start + sig::PUBKEY_LEN..]);
        Some(RsClientBulletinEntry {
            comm,
            digest,
            pubkey,
            sig: s,
        })
    }
}

#[derive(Clone)]
pub struct RsNodeBulletinEntry {
    pub node_id: NodeId,
    pub clients: Vec<ClientId>,
    pub share_sum: Share,
}

#[derive(Clone)]
pub struct ServerBulletinEntry {
    pub server_id: ServerId,
    pub clients: Vec<ClientId>,
    pub agg_open: Opening,
    pub agg_share: CsPoly,
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
    use chipmunk_code::{HVCPoly, Polynomial};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn client_bulletin_entry_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        for mu_kahe in [1usize, 2, 4] {
            let ctxt: Vec<KahePoly> = (0..mu_kahe).map(|_| KahePoly::rand_poly(&mut rng)).collect();
            let entry = ClientBulletinEntry {
                ctxt,
                comm: Commitment { root: HVCPoly::rand_poly(&mut rng) },
            };
            let bytes = entry.to_bytes();
            let back = ClientBulletinEntry::from_bytes(&bytes).unwrap();
            assert_eq!(back.ctxt, entry.ctxt);
            assert_eq!(back.comm.root, entry.comm.root);
        }
        assert!(ClientBulletinEntry::from_bytes(&[0u8]).is_none());
    }
}
