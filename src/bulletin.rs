//! Public bulletin (in-memory v1).
//!
//! Holds the publicly broadcast values from the protocol: per-client ciphertext
//! and commitment, per-server aggregated output, and the canonical client set.
//! Concrete typed storage for v1; abstract over a `Bulletin` trait when we need
//! a networked impl.

use std::sync::Mutex;

use crate::cs::Commitment;
use crate::protocol::message::{ClientId, ServerId};
use chipmunk_code::HVCPoly;

#[derive(Clone)]
pub struct ClientPublic {
    pub ctxt: HVCPoly,
    pub comm: Commitment,
}

#[derive(Clone)]
pub struct ServerPublic {
    pub server_id: ServerId,
    pub clients: Vec<ClientId>,
    pub agg_open: crate::cs::Opening,
    pub agg_share: HVCPoly,
}

#[derive(Default)]
pub struct InMemoryBulletin {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    clients: Vec<(ClientId, ClientPublic)>,
    servers: Vec<ServerPublic>,
    canonical: Option<Vec<ClientId>>,
}

impl InMemoryBulletin {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn publish_client(&self, id: ClientId, p: ClientPublic) {
        self.inner.lock().unwrap().clients.push((id, p));
    }

    pub fn publish_server(&self, p: ServerPublic) {
        self.inner.lock().unwrap().servers.push(p);
    }

    pub fn publish_canonical(&self, set: Vec<ClientId>) {
        self.inner.lock().unwrap().canonical = Some(set);
    }

    pub fn clients(&self) -> Vec<(ClientId, ClientPublic)> {
        self.inner.lock().unwrap().clients.clone()
    }

    pub fn servers(&self) -> Vec<ServerPublic> {
        self.inner.lock().unwrap().servers.clone()
    }

    pub fn canonical(&self) -> Option<Vec<ClientId>> {
        self.inner.lock().unwrap().canonical.clone()
    }
}
