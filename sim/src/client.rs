use std::net::SocketAddr;
use std::sync::Arc;

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
use crate::metrics::{self, Span, Timer};
use crate::report::NodeReport;
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
) -> Result<Json<NodeReport>, http::Error> {
    let sid = http::parse_session(&session)?;
    let index = node.args.index;
    let mut spans = Vec::new();

    let requests = node
        .args
        .servers
        .iter()
        .enumerate()
        .map(|(i, s)| (format!("download_key_{i}"), format!("{s}/key")))
        .collect();
    let keys = http::get_all(&node.http, requests).await?;
    let servers = keys
        .iter()
        .enumerate()
        .map(|(i, (bytes, _))| {
            let key = pke::PublicKey::from_bytes(bytes)
                .map_err(|e| eyre::eyre!("key of server {i} does not parse: {e:?}"))?;
            Ok((ServerId(i as u32), key))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    let key_spans: Vec<Span> = keys.into_iter().map(|(_, span)| span).collect();
    spans.push(metrics::merge("download_keys", &key_spans));

    let timer = Timer::start_cpu("client_round");
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
    spans.push(timer.finish());

    // the post and every opening share this client's uplink, so send them together
    let mut uploads = JoinSet::new();
    let post = output.encrypted_message.to_bytes();
    let url = format!("{}/{session}/client/{index}", node.args.bulletin);
    uploads.spawn(upload(node.http.clone(), "upload_post".into(), url, post));
    for (sid, sealed) in output.sealed_openings {
        let server = sid.0 as usize;
        let url = format!("{}/opening/{session}/{index}", node.args.servers[server]);
        let name = format!("upload_opening_{server}");
        uploads.spawn(upload(node.http.clone(), name, url, sealed));
    }
    while let Some(joined) = uploads.join_next().await {
        spans.push(joined.wrap_err("upload task panicked")??);
    }

    Ok(Json(NodeReport {
        node: format!("client-{index}"),
        spans,
    }))
}

async fn upload(
    http: reqwest::Client,
    name: String,
    url: String,
    body: Vec<u8>,
) -> eyre::Result<Span> {
    let timer = Timer::start(name);
    let len = body.len();
    http::post_bytes(&http, url, body).await?;
    Ok(timer.finish_bytes(len))
}
