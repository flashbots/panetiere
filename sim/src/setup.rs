use clap::Args;
use eyre::{ensure, eyre};
use panetiere::channel::ChannelParams;
use panetiere::protocol::ProtocolParams;
use rand::SeedableRng as _;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

/// Round shape shared by every node. Every node must get the same values,
/// since each one derives the public parameters from them on its own.
#[derive(Args, Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Setup {
    /// Seed for the shared public parameters.
    #[arg(long, env = "SIM_SEED", default_value_t = 0)]
    pub(crate) seed: u64,
    #[arg(long, env = "SIM_SERVERS", default_value_t = 4)]
    pub(crate) servers: usize,
    #[arg(long, env = "SIM_CLIENTS", default_value_t = 8)]
    pub(crate) clients: usize,
    /// Clients `0..messages` send a real message, the rest send cover traffic.
    #[arg(long, env = "SIM_MESSAGES", default_value_t = 6)]
    pub(crate) messages: usize,
    #[arg(long, env = "SIM_MESSAGE_BYTES", default_value_t = 32)]
    pub(crate) message_bytes: usize,
    /// Smallest client set the recipient will decrypt over.
    #[arg(long, env = "SIM_MIN_CLIENTS", default_value_t = 1)]
    pub(crate) min_clients: usize,
}

pub(crate) struct Params {
    pub(crate) channel: ChannelParams,
    pub(crate) protocol: ProtocolParams,
}

impl Setup {
    pub(crate) fn params(&self) -> eyre::Result<Params> {
        ensure!(self.servers > 0, "need at least one server");
        ensure!(self.messages > 0, "need at least one message");
        ensure!(
            self.messages <= self.clients,
            "{} messages but only {} clients",
            self.messages,
            self.clients,
        );
        let capacity = u32::try_from(self.messages).map_err(|_| eyre!("too many messages"))?;

        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&self.seed.to_le_bytes());
        let mut rng = ChaCha20Rng::from_seed(seed);
        let channel = ChannelParams::for_messages(&mut rng, capacity, self.message_bytes);
        let mut protocol = channel.protocol_params(&mut rng, self.servers);
        protocol.min_clients = self.min_clients;
        ensure!(
            self.clients <= protocol.max_clients,
            "{} clients is more than the {} these parameters can sum",
            self.clients,
            protocol.max_clients,
        );
        Ok(Params { channel, protocol })
    }

    /// The payload client `index` sends, or `None` if it only sends cover.
    pub(crate) fn message(&self, index: u32) -> Option<Vec<u8>> {
        ((index as usize) < self.messages)
            .then(|| format!("message from client {index}").into_bytes())
    }
}
