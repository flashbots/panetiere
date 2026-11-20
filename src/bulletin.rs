//! Public bulletin (in-memory v1).

use std::sync::Mutex;

use crate::cs::{
    aggregated_opening_pack_bounds, pack_cs_shares, pack_poly64, poly_packed_len,
    poly_packed_len64, unpack_cs_shares, unpack_poly64, Commitment, Opening, PackedOpening,
};
use crate::protocol::{ClientId, ProtocolParams, ServerId};
use crate::{CsPoly, KahePoly, CS_MODULUS, KAHE_MODULUS};
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
}
