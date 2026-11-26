use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use panetiere::bulletin::ClientBulletinEntry;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use crate::http;
use crate::metrics::Window;
use crate::report::NodeReport;

#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long, env = "SIM_LISTEN", default_value = "0.0.0.0:8080")]
    pub(crate) listen: SocketAddr,
}

/// Everything published for one session, kept as the bytes that were posted,
/// and when posts and entries arrived. Serving is not timed here: a handler
/// returns long before its response finishes sending, so the recipient and
/// servers time their own downloads instead.
#[derive(Default)]
struct Session {
    clients: HashMap<u32, Bytes>,
    commitments: HashMap<u32, Bytes>,
    servers: HashMap<u32, Bytes>,
    canonical: Option<Vec<u32>>,
    received_posts: Window,
    received_entries: Window,
}

type Sessions = Arc<Mutex<HashMap<String, Session>>>;

pub(crate) async fn run(listener: TcpListener) -> eyre::Result<()> {
    let router = Router::new()
        .route("/{session}/client/{id}", post(post_client).get(get_client))
        .route("/{session}/commitment/{id}", get(get_commitment))
        .route("/{session}/server/{id}", post(post_server).get(get_server))
        .route(
            "/{session}/canonical",
            post(post_canonical).get(get_canonical),
        )
        .route("/{session}/report", get(report))
        .with_state(Sessions::default());
    http::serve(listener, router).await
}

/// Servers only need the commitment, so it is split out here and served on
/// its own instead of making every server download every ciphertext.
async fn post_client(
    State(sessions): State<Sessions>,
    Path((session, id)): Path<(String, u32)>,
    body: Bytes,
) -> Result<(), (StatusCode, String)> {
    let entry = ClientBulletinEntry::try_from_bytes(&body).ok_or((
        StatusCode::BAD_REQUEST,
        format!("post from client {id} does not parse"),
    ))?;
    let commitment = Bytes::from(entry.commitment().to_bytes());
    let mut sessions = sessions.lock().await;
    let s = sessions.entry(session).or_default();
    s.received_posts.record(body.len());
    s.clients.insert(id, body);
    s.commitments.insert(id, commitment);
    Ok(())
}

async fn get_client(
    State(sessions): State<Sessions>,
    Path((session, id)): Path<(String, u32)>,
) -> Result<Bytes, StatusCode> {
    let sessions = sessions.lock().await;
    sessions
        .get(&session)
        .and_then(|s| s.clients.get(&id).cloned())
        .ok_or(StatusCode::NOT_FOUND)
}

async fn get_commitment(
    State(sessions): State<Sessions>,
    Path((session, id)): Path<(String, u32)>,
) -> Result<Bytes, StatusCode> {
    let sessions = sessions.lock().await;
    sessions
        .get(&session)
        .and_then(|s| s.commitments.get(&id).cloned())
        .ok_or(StatusCode::NOT_FOUND)
}

async fn post_server(
    State(sessions): State<Sessions>,
    Path((session, id)): Path<(String, u32)>,
    body: Bytes,
) {
    let mut sessions = sessions.lock().await;
    let s = sessions.entry(session).or_default();
    s.received_entries.record(body.len());
    s.servers.insert(id, body);
}

async fn get_server(
    State(sessions): State<Sessions>,
    Path((session, id)): Path<(String, u32)>,
) -> Result<Bytes, StatusCode> {
    let sessions = sessions.lock().await;
    sessions
        .get(&session)
        .and_then(|s| s.servers.get(&id).cloned())
        .ok_or(StatusCode::NOT_FOUND)
}

async fn post_canonical(
    State(sessions): State<Sessions>,
    Path(session): Path<String>,
    Json(set): Json<Vec<u32>>,
) {
    let mut sessions = sessions.lock().await;
    sessions.entry(session).or_default().canonical = Some(set);
}

async fn get_canonical(
    State(sessions): State<Sessions>,
    Path(session): Path<String>,
) -> Result<Json<Vec<u32>>, StatusCode> {
    let sessions = sessions.lock().await;
    sessions
        .get(&session)
        .and_then(|s| s.canonical.clone())
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

async fn report(
    State(sessions): State<Sessions>,
    Path(session): Path<String>,
) -> Result<Json<NodeReport>, StatusCode> {
    let sessions = sessions.lock().await;
    let s = sessions.get(&session).ok_or(StatusCode::NOT_FOUND)?;
    let spans = [
        s.received_posts.span("receive_posts"),
        s.received_entries.span("receive_entries"),
    ];
    Ok(Json(NodeReport {
        node: "bulletin".into(),
        spans: spans.into_iter().flatten().collect(),
    }))
}
