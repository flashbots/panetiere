use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use eyre::{eyre, WrapErr as _};
use panetiere::cs::Commitment;
use panetiere::pke;
use panetiere::protocol::server::{run_server_round, unseal_openings, ServerInbox};
use panetiere::protocol::{ClientId, ServerId};
use rand::rngs::OsRng;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use crate::http;
use crate::report::ServerReport;
use crate::setup::{Params, Setup};

#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long, env = "SIM_LISTEN", default_value = "0.0.0.0:8080")]
    pub(crate) listen: SocketAddr,
    /// This server's position in the server list.
    #[arg(long, env = "SIM_INDEX")]
    pub(crate) index: u32,
    #[arg(long, env = "SIM_BULLETIN")]
    pub(crate) bulletin: String,
}

/// Sealed openings received so far, by session.
type Inbox = HashMap<String, Vec<(ClientId, Vec<u8>)>>;

struct Node {
    params: Params,
    id: ServerId,
    key: pke::PrivateKey,
    bulletin: String,
    http: reqwest::Client,
    inbox: Mutex<Inbox>,
}

pub(crate) async fn run(listener: TcpListener, setup: Setup, args: Args) -> eyre::Result<()> {
    let node = Arc::new(Node {
        params: setup.params().wrap_err("failed to derive parameters")?,
        id: ServerId(args.index),
        key: pke::PrivateKey::generate(&mut OsRng),
        bulletin: args.bulletin,
        http: reqwest::Client::new(),
        inbox: Mutex::default(),
    });
    let router = Router::new()
        .route("/key", get(key))
        .route("/opening/{session}/{client}", post(opening))
        .route("/round/{session}", post(round))
        .with_state(node);
    http::serve(listener, router).await
}

async fn key(State(node): State<Arc<Node>>) -> Vec<u8> {
    node.key.public().to_bytes()
}

async fn opening(
    State(node): State<Arc<Node>>,
    Path((session, client)): Path<(String, u32)>,
    body: Bytes,
) {
    let mut inbox = node.inbox.lock().await;
    inbox
        .entry(session)
        .or_default()
        .push((ClientId(client), body.to_vec()));
}

async fn round(
    State(node): State<Arc<Node>>,
    Path(session): Path<String>,
) -> Result<Json<ServerReport>, http::Error> {
    let sid = http::parse_session(&session)?;
    let base = format!("{}/{session}", node.bulletin);

    let start = Instant::now();
    let canonical: Vec<u32> =
        serde_json::from_slice(&http::get_bytes(&node.http, format!("{base}/canonical")).await?)
            .wrap_err("canonical set is not json")?;
    let urls = canonical
        .iter()
        .map(|c| format!("{base}/commitment/{c}"))
        .collect();
    let commitments = http::get_all(&node.http, urls)
        .await?
        .iter()
        .zip(&canonical)
        .map(|(bytes, c)| {
            Commitment::from_bytes(bytes)
                .map(|comm| (ClientId(*c), comm))
                .ok_or_else(|| eyre!("commitment of client {c} does not parse"))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    let download_ms = start.elapsed().as_millis();

    let sealed = node.inbox.lock().await.remove(&session).unwrap_or_default();
    let start = Instant::now();
    let worker = node.clone();
    let entry = tokio::task::spawn_blocking(move || -> eyre::Result<Vec<u8>> {
        let opened = unseal_openings(&worker.key, &sid, worker.id, &sealed);
        let items = sealed
            .iter()
            .zip(opened)
            .map(|((cid, _), op)| {
                op.map(|op| (*cid, op))
                    .ok_or_else(|| eyre!("opening from client {} does not unseal", cid.0))
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        let inbox = ServerInbox {
            server_id: worker.id,
            items,
        };
        let canonical: Vec<ClientId> = canonical.into_iter().map(ClientId).collect();
        let cs = &worker.params.protocol.cs;
        let entry = run_server_round(cs, &inbox, &canonical, &commitments)
            .map_err(|e| eyre!("server round failed: {e:?}"))?;
        Ok(entry.to_bytes(cs))
    })
    .await
    .wrap_err("server round task panicked")??;
    let compute_ms = start.elapsed().as_millis();

    let start = Instant::now();
    let entry_bytes = entry.len();
    http::post_bytes(&node.http, format!("{base}/server/{}", node.id.0), entry).await?;
    let upload_ms = start.elapsed().as_millis();

    Ok(Json(ServerReport {
        index: node.id.0,
        download_ms,
        compute_ms,
        upload_ms,
        entry_bytes,
    }))
}
