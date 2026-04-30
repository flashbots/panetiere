//! Client-set consensus stub.
//!
//! v1: returns a fixed list of client ids. The honest-server invariant from
//! README §5 (servers must refuse to decrypt anything other than the canonical
//! set) is the responsibility of the server logic.

use crate::protocol::message::ClientId;

pub trait ClientSetSelector {
    fn canonical_set(&self) -> Vec<ClientId>;
}

pub struct FixedSet(pub Vec<ClientId>);

impl ClientSetSelector for FixedSet {
    fn canonical_set(&self) -> Vec<ClientId> {
        self.0.clone()
    }
}
