use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ClientReport {
    pub(crate) index: u32,
    pub(crate) compute_ms: u128,
    pub(crate) upload_ms: u128,
    pub(crate) post_bytes: usize,
    pub(crate) opening_bytes: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ServerReport {
    pub(crate) index: u32,
    pub(crate) download_ms: u128,
    pub(crate) compute_ms: u128,
    pub(crate) upload_ms: u128,
    pub(crate) entry_bytes: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct RecipientReport {
    pub(crate) download_ms: u128,
    pub(crate) download_bytes: usize,
    pub(crate) compute_ms: u128,
    pub(crate) messages: Vec<String>,
}

/// One line of the coordinator's output. The phase times are wall-clock as
/// seen by the coordinator, so they include the control requests.
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct RoundReport {
    pub(crate) round: usize,
    pub(crate) session: String,
    pub(crate) clients_ms: u128,
    pub(crate) servers_ms: u128,
    pub(crate) recipient_ms: u128,
    pub(crate) total_ms: u128,
    pub(crate) clients: Vec<ClientReport>,
    pub(crate) servers: Vec<ServerReport>,
    pub(crate) recipient: RecipientReport,
}
