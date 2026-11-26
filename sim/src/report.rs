use serde::{Deserialize, Serialize};

use crate::metrics::{Link, Span, Usage};
use crate::setup::Setup;

/// Everything one node timed during a round.
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct NodeReport {
    pub(crate) node: String,
    pub(crate) spans: Vec<Span>,
    pub(crate) usage: Usage,
    pub(crate) link: Link,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct RecipientReport {
    #[serde(flatten)]
    pub(crate) report: NodeReport,
    pub(crate) messages: Vec<String>,
}

/// One line of the coordinator's output.
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct RoundReport {
    pub(crate) label: String,
    pub(crate) round: usize,
    pub(crate) session: String,
    pub(crate) setup: Setup,
    /// The coordinator's own view of each phase, control requests included.
    pub(crate) phases: Vec<Span>,
    /// Every node's report: clients, then servers, then the recipient.
    pub(crate) nodes: Vec<NodeReport>,
}
