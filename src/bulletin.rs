//! Public bulletin (in-memory v1).

use std::sync::Mutex;

use crate::cs::{Commitment, Opening};
use crate::protocol::{ClientId, ServerId};
use chipmunk_code::{HVCPoly, KahePoly};

#[derive(Clone)]
pub struct ClientBulletinEntry {
    /// KAHE ciphertext, one KAHE ring element per `μ_kahe` slot.
    pub ctxt: Vec<KahePoly>,
    /// Single CS commitment (μ_cs = κ_kahe packs the share-vector).
    pub comm: Commitment,
}

#[derive(Clone)]
pub struct ServerBulletinEntry {
    pub server_id: ServerId,
    pub clients: Vec<ClientId>,
    /// Single aggregated `Opening` whose `s()` is the κ_kahe-vector of summed shares.
    pub agg_open: Opening,
    /// Mirrors `agg_open.s()` (componentwise sum of per-client shares at this server's point).
    pub agg_share: Vec<HVCPoly>,
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
