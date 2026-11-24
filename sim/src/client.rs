use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::routing::post;
use axum::{Json, Router};
use eyre::WrapErr as _;
use panetiere::channel;
use panetiere::pke;
use panetiere::protocol::client::run_client_round;
use panetiere::protocol::{ClientId, ServerId};
use rand::SeedableRng as _;
use rand_chacha::ChaCha20Rng;
use tokio::net::TcpListener;
use tokio::task::JoinSet;

use crate::http;
use crate::report::ClientReport;
use crate::setup::{Params, Setup};

#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long, env = "SIM_LISTEN", default_value = "0.0.0.0:8080")]
    pub(crate) listen: SocketAddr,
    /// This client's id. Decides whether it sends a message or cover.
    #[arg(long, env = "SIM_INDEX")]
    pub(crate) index: u32,
    #[arg(long, env = "SIM_BULLETIN")]
    pub(crate) bulletin: String,
    /// Server urls, in server id order.
    #[arg(long, env = "SIM_SERVER_URLS", value_delimiter = ',')]
    pub(crate) servers: Vec<String>,
}

struct Node {
    setup: Setup,
    params: Params,
    args: Args,
    http: reqwest::Client,
}

pub(crate) async fn run(listener: TcpListener, setup: Setup, args: Args) -> eyre::Result<()> {
    let node = Arc::new(Node {
        params: setup.params().wrap_err("failed to derive parameters")?,
        setup,
        args,
        http: reqwest::Client::new(),
    });
    let router = Router::new()
        .route("/round/{session}", post(round))
        .with_state(node);
    http::serve(listener, router).await
}

async fn round(
    State(node): State<Arc<Node>>,
    Path(session): Path<String>,
) -> Result<Json<ClientReport>, http::Error> {
    let sid = http::parse_session(&session)?;
    let index = node.args.index;

    // keys are fetched every round, outside the timed part: a real client
    // would already know them
    let urls = node
        .args
        .servers
        .iter()
        .map(|s| format!("{s}/key"))
        .collect();
    let servers = http::get_all(&node.http, urls)
        .await?
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            let key = pke::PublicKey::from_bytes(bytes)
                .map_err(|e| eyre::eyre!("key of server {i} does not parse: {e:?}"))?;
            Ok((ServerId(i as u32), key))
        })
        .collect::<eyre::Result<Vec<_>>>()?;

    let start = Instant::now();
    let worker = node.clone();
    let output = tokio::task::spawn_blocking(move || -> eyre::Result<_> {
        let mut rng = ChaCha20Rng::from_entropy();
        let ch = &worker.params.channel;
        let contribution = match worker.setup.message(index) {
            Some(m) => channel::encode_message(&mut rng, ch, &sid, &m)
                .map_err(|e| eyre::eyre!("failed to encode message: {e:?}"))?,
            None => channel::cover(ch),
        };
        let pp = &worker.params.protocol;
        Ok(run_client_round(
            &mut rng,
            pp,
            &sid,
            ClientId(index),
            contribution,
            &servers,
        ))
    })
    .await
    .wrap_err("client round task panicked")??;
    let compute_ms = start.elapsed().as_millis();

    // the post and every opening share this client's uplink, so send them together
    let start = Instant::now();
    let post = output.encrypted_message.to_bytes();
    let post_bytes = post.len();
    let mut opening_bytes = 0;
    let mut uploads = JoinSet::new();
    let url = format!("{}/{session}/client/{index}", node.args.bulletin);
    let http_client = node.http.clone();
    uploads.spawn(async move { http::post_bytes(&http_client, url, post).await });
    for (sid, sealed) in output.sealed_openings {
        opening_bytes += sealed.len();
        let url = format!(
            "{}/opening/{session}/{index}",
            node.args.servers[sid.0 as usize]
        );
        let http_client = node.http.clone();
        uploads.spawn(async move { http::post_bytes(&http_client, url, sealed).await });
    }
    while let Some(joined) = uploads.join_next().await {
        joined.wrap_err("upload task panicked")??;
    }
    let upload_ms = start.elapsed().as_millis();

    Ok(Json(ClientReport {
        index,
        compute_ms,
        upload_ms,
        post_bytes,
        opening_bytes,
    }))
}
