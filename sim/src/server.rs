use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use eyre::{eyre, WrapErr as _};
use panetiere::pke;
use panetiere::protocol::server::{run_server_round, unseal_openings, ServerInbox};
use panetiere::protocol::{ClientId, Commitment, ServerId};
use rand::rngs::OsRng;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use crate::http;
use crate::metrics::{self, Span, Timer, Window};
use crate::report::NodeReport;
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

/// Sealed openings received for one session, and when they arrived.
#[derive(Default)]
struct Inbox {
    sealed: Vec<(ClientId, Vec<u8>)>,
    received: Window,
}

struct Node {
    params: Params,
    id: ServerId,
    key: pke::PrivateKey,
    bulletin: String,
    http: reqwest::Client,
    inboxes: Mutex<HashMap<String, Inbox>>,
}

pub(crate) async fn run(listener: TcpListener, setup: Setup, args: Args) -> eyre::Result<()> {
    let node = Arc::new(Node {
        params: setup.params().wrap_err("failed to derive parameters")?,
        id: ServerId::new(args.index),
        key: pke::PrivateKey::generate(&mut OsRng),
        bulletin: args.bulletin,
        http: reqwest::Client::new(),
        inboxes: Mutex::default(),
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
    let mut inboxes = node.inboxes.lock().await;
    let inbox = inboxes.entry(session).or_default();
    inbox.received.record(body.len());
    inbox.sealed.push((ClientId::new(client), body.to_vec()));
}

async fn round(
    State(node): State<Arc<Node>>,
    Path(session): Path<String>,
) -> Result<Json<NodeReport>, http::Error> {
    let sid = http::parse_session(&session)?;
    let base = format!("{}/{session}", node.bulletin);
    let mut spans = Vec::new();

    let timer = Timer::start("download_canonical");
    let body = http::get_bytes(&node.http, format!("{base}/canonical")).await?;
    spans.push(timer.finish_bytes(body.len()));
    let canonical: Vec<u32> =
        serde_json::from_slice(&body).wrap_err("canonical set is not json")?;

    let requests = canonical
        .iter()
        .map(|c| {
            (
                format!("download_commitment_{c}"),
                format!("{base}/commitment/{c}"),
            )
        })
        .collect();
    let downloads = http::get_all(&node.http, requests).await?;
    let commitments = downloads
        .iter()
        .zip(&canonical)
        .map(|((bytes, _), c)| {
            Commitment::try_from_bytes(bytes)
                .map(|comm| (ClientId::new(*c), comm))
                .ok_or_else(|| eyre!("commitment of client {c} does not parse"))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    let download_spans: Vec<Span> = downloads.into_iter().map(|(_, span)| span).collect();
    spans.push(metrics::merge("download_commitments", &download_spans));

    let inbox = node
        .inboxes
        .lock()
        .await
        .remove(&session)
        .unwrap_or_default();
    spans.extend(inbox.received.span("receive_openings"));

    let worker = node.clone();
    let (entry, compute_spans) =
        tokio::task::spawn_blocking(move || -> eyre::Result<(Vec<u8>, Vec<Span>)> {
            let timer = Timer::start_cpu("unseal");
            let opened = unseal_openings(&worker.key, &sid, worker.id, &inbox.sealed);
            let unseal = timer.finish();
            let items = inbox
                .sealed
                .iter()
                .zip(opened)
                .map(|((cid, _), op)| {
                    op.map(|op| (*cid, op))
                        .ok_or_else(|| eyre!("opening from client {} does not unseal", cid.get()))
                })
                .collect::<eyre::Result<Vec<_>>>()?;
            let server_inbox = ServerInbox::new(worker.id, items);
            let canonical: Vec<ClientId> = canonical.into_iter().map(ClientId::new).collect();
            let pp = &worker.params.protocol;
            let timer = Timer::start_cpu("server_round");
            let entry = run_server_round(pp, &server_inbox, &canonical, &commitments)
                .map_err(|e| eyre!("server round failed: {e:?}"))?;
            let bytes = entry.to_bytes(pp);
            Ok((bytes, vec![unseal, timer.finish()]))
        })
        .await
        .wrap_err("server round task panicked")??;
    spans.extend(compute_spans);

    let timer = Timer::start("upload_entry");
    let len = entry.len();
    http::post_bytes(
        &node.http,
        format!("{base}/server/{}", node.id.get()),
        entry,
    )
    .await?;
    spans.push(timer.finish_bytes(len));

    Ok(Json(NodeReport {
        node: format!("server-{}", node.id.get()),
        spans,
    }))
}
