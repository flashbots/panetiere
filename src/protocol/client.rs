use rand::Rng;

use crate::bulletin::ClientPublic;
use crate::cs::{Cs, HidingMerkleCommitment, Opening};
use crate::kahe::{Kahe, RingOtp};
use crate::sss::{AdditiveSharing, Sss};

use super::message::{ClientId, ServerId};

/// Output of a single client round (README steps 1–6 in concrete form).
pub struct ClientRound {
    pub client_id: ClientId,
    pub public: ClientPublic,
    /// Per-server private payload: opening and key share.
    pub private: Vec<(ServerId, Opening, <RingOtp as Kahe>::Key)>,
}

pub fn run_client_round<R: Rng>(
    rng: &mut R,
    pp: &<HidingMerkleCommitment as Cs>::Params,
    client_id: ClientId,
    message: <RingOtp as Kahe>::Message,
    servers: &[ServerId],
) -> ClientRound {
    let key = RingOtp::gen(rng);
    let ctxt = RingOtp::enc(&key, &message);

    let n = servers.len();
    assert_eq!(n, pp.n_servers, "server count must match CS params");

    let shares = AdditiveSharing::share(rng, &key, n);
    let (comm, openings) = HidingMerkleCommitment::commit(rng, pp, &shares);

    let private: Vec<_> = servers
        .iter()
        .copied()
        .zip(openings)
        .zip(shares)
        .map(|((sid, op), sh)| (sid, op, sh))
        .collect();

    ClientRound {
        client_id,
        public: ClientPublic { ctxt, comm },
        private,
    }
}
