use std::net::SocketAddr;

use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Router;
use eyre::{bail, WrapErr as _};
use panetiere::protocol::SessionId;
use serde::de::DeserializeOwned;
use tokio::net::TcpListener;
use tokio::task::JoinSet;

use crate::metrics::{Span, Timer};

/// Handler error: answers 500 with the whole error chain as the body.
pub(crate) struct Error(eyre::Report);

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("{:#}", self.0)).into_response()
    }
}

impl<E: Into<eyre::Report>> From<E> for Error {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

pub(crate) async fn bind(addr: SocketAddr) -> eyre::Result<TcpListener> {
    TcpListener::bind(addr)
        .await
        .wrap_err_with(|| format!("failed to listen on {addr}"))
}

pub(crate) async fn serve(listener: TcpListener, router: Router) -> eyre::Result<()> {
    let addr = listener
        .local_addr()
        .wrap_err("failed to read listen address")?;
    tracing::info!(%addr, "listening");
    // client posts are megabytes, far past axum's default limit
    let router = router
        .route("/health", axum::routing::get(|| async { "ok" }))
        .layer(DefaultBodyLimit::disable());
    axum::serve(listener, router)
        .await
        .wrap_err("http server stopped")
}

pub(crate) fn parse_session(hex: &str) -> eyre::Result<SessionId> {
    let bytes = hex::decode(hex).wrap_err("session id is not hex")?;
    let id: [u8; 32] = bytes
        .try_into()
        .map_err(|_| eyre::eyre!("session id is not 32 bytes"))?;
    Ok(SessionId(id))
}

async fn check(resp: reqwest::Response, url: &str) -> eyre::Result<reqwest::Response> {
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!("{url} answered {status}: {body}");
    }
    Ok(resp)
}

pub(crate) async fn get_bytes(http: &reqwest::Client, url: String) -> eyre::Result<Vec<u8>> {
    let resp = http
        .get(&url)
        .send()
        .await
        .wrap_err_with(|| format!("failed to send request to {url}"))?;
    let bytes = check(resp, &url)
        .await?
        .bytes()
        .await
        .wrap_err_with(|| format!("failed to read body from {url}"))?;
    Ok(bytes.to_vec())
}

pub(crate) async fn post_bytes(
    http: &reqwest::Client,
    url: String,
    body: Vec<u8>,
) -> eyre::Result<()> {
    let resp = http
        .post(&url)
        .body(body)
        .send()
        .await
        .wrap_err_with(|| format!("failed to send request to {url}"))?;
    check(resp, &url).await?;
    Ok(())
}

/// Empty POST, JSON answer. The coordinator's way to start a step on a node.
pub(crate) async fn post_for_json<T: DeserializeOwned>(
    http: &reqwest::Client,
    url: String,
) -> eyre::Result<T> {
    let resp = http
        .post(&url)
        .send()
        .await
        .wrap_err_with(|| format!("failed to send request to {url}"))?;
    let bytes = check(resp, &url)
        .await?
        .bytes()
        .await
        .wrap_err_with(|| format!("failed to read body from {url}"))?;
    serde_json::from_slice(&bytes).wrap_err_with(|| format!("bad json from {url}"))
}

/// GETs every url at once. Returns each body in request order, with a span
/// named after the request.
pub(crate) async fn get_all(
    http: &reqwest::Client,
    requests: Vec<(String, String)>,
) -> eyre::Result<Vec<(Vec<u8>, Span)>> {
    let mut tasks = JoinSet::new();
    let n = requests.len();
    for (i, (name, url)) in requests.into_iter().enumerate() {
        let http = http.clone();
        tasks.spawn(async move {
            let timer = Timer::start(name);
            let body = get_bytes(&http, url).await;
            (
                i,
                body.map(|b| {
                    let span = timer.finish_bytes(b.len());
                    (b, span)
                }),
            )
        });
    }
    let mut out: Vec<Option<(Vec<u8>, Span)>> = (0..n).map(|_| None).collect();
    while let Some(joined) = tasks.join_next().await {
        let (i, body) = joined.wrap_err("download task panicked")?;
        out[i] = Some(body?);
    }
    Ok(out.into_iter().flatten().collect())
}
