//! Public bulletin (in-memory v1).

use std::sync::Mutex;

use crate::cs::{pack_poly64, poly_packed_len, poly_packed_len64, unpack_poly64, Commitment, Opening};
use crate::protocol::{ClientId, ServerId};
use chipmunk_code::{CsPoly, KahePoly, HVC_MODULUS, KAHE_MODULUS};

#[derive(Clone)]
pub struct ClientBulletinEntry {
    /// KAHE ciphertext, one KAHE ring element per `μ_kahe` slot.
    pub ctxt: Vec<KahePoly>,
    /// Single CS commitment (μ_cs = κ_kahe packs the share-vector).
    pub comm: Commitment,
}

impl ClientBulletinEntry {
    /// Bit-packed wire form: a `u16` ciphertext-slot count, the ciphertext
    /// polynomials packed against `KAHE_MODULUS`, then the commitment.
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

    /// Byte length of [`to_bytes`](Self::to_bytes) for `mu_kahe` ciphertext
    /// slots, without building an entry — for wire-budget planning.
    pub fn packed_len(mu_kahe: usize) -> usize {
        2 + mu_kahe * poly_packed_len64(KAHE_MODULUS) + poly_packed_len(HVC_MODULUS)
    }

    /// Inverse of [`ClientBulletinEntry::to_bytes`]; `None` on any length
    /// mismatch.
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
pub struct ServerBulletinEntry {
    pub server_id: ServerId,
    pub clients: Vec<ClientId>,
    /// Single aggregated `Opening` whose `s()` is the κ_kahe-vector of summed shares.
    pub agg_open: Opening,
    /// Mirrors `agg_open.s()` (componentwise sum of per-client shares at this server's point).
    pub agg_share: Vec<CsPoly>,
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
