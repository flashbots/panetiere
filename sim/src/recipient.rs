use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::routing::post;
use axum::{Json, Router};
use eyre::{eyre, WrapErr as _};
use panetiere::bulletin::ServerBulletinEntry;
use panetiere::channel;
use panetiere::protocol::recipient::{recover_unverified_direct_from_bytes, SetPolicy};
use panetiere::protocol::ClientId;
use tokio::net::TcpListener;

use crate::http;
use crate::metrics::{self, Span, Timer};
use crate::report::{NodeReport, RecipientReport};
use crate::setup::{Params, Setup};

#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long, env = "SIM_LISTEN", default_value = "0.0.0.0:8080")]
    pub(crate) listen: SocketAddr,
    #[arg(long, env = "SIM_BULLETIN")]
    pub(crate) bulletin: String,
}

struct Node {
    setup: Setup,
    params: Params,
    bulletin: String,
    http: reqwest::Client,
}

pub(crate) async fn run(listener: TcpListener, setup: Setup, args: Args) -> eyre::Result<()> {
    let node = Arc::new(Node {
        params: setup.params().wrap_err("failed to derive parameters")?,
        setup,
        bulletin: args.bulletin,
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
) -> Result<Json<RecipientReport>, http::Error> {
    let sid = http::parse_session(&session)?;
    let base = format!("{}/{session}", node.bulletin);
    let mut spans = Vec::new();

    let timer = Timer::start("download_canonical");
    let body = http::get_bytes(&node.http, format!("{base}/canonical")).await?;
    spans.push(timer.finish_bytes(body.len()));
    let canonical: Vec<u32> =
        serde_json::from_slice(&body).wrap_err("canonical set is not json")?;

    // one span per download, so the timeline shows when each post arrived
    let server_requests = (0..node.setup.servers)
        .map(|i| (format!("download_entry_{i}"), format!("{base}/server/{i}")))
        .collect();
    let client_requests = canonical
        .iter()
        .map(|c| (format!("download_post_{c}"), format!("{base}/client/{c}")))
        .collect();
    let (server_downloads, client_downloads) = tokio::try_join!(
        http::get_all(&node.http, server_requests),
        http::get_all(&node.http, client_requests),
    )?;
    let (server_bytes, server_spans): (Vec<_>, Vec<_>) = server_downloads.into_iter().unzip();
    let (client_bytes, client_spans): (Vec<_>, Vec<_>) = client_downloads.into_iter().unzip();
    spans.extend(server_spans);
    spans.extend(client_spans);

    let worker = node.clone();
    let (messages, compute_spans) =
        tokio::task::spawn_blocking(move || -> eyre::Result<(Vec<Vec<u8>>, Vec<Span>)> {
            let timer = Timer::start_cpu("recover");
            let servers = server_bytes
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    ServerBulletinEntry::try_from_bytes(b)
                        .ok_or_else(|| eyre!("entry of server {i} does not parse"))
                })
                .collect::<eyre::Result<Vec<_>>>()?;
            let posts: HashMap<ClientId, Bytes> = canonical
                .iter()
                .map(|c| ClientId::new(*c))
                .zip(client_bytes)
                .collect();
            let canonical: Vec<ClientId> = canonical.into_iter().map(ClientId::new).collect();
            let pp = &worker.params.protocol;
            let policy =
                SetPolicy::anchored(&canonical, worker.setup.min_clients, pp.max_clients());
            let recovered = recover_unverified_direct_from_bytes(
                pp,
                &policy,
                |cid| posts.get(&cid).map(Bytes::as_ref),
                &servers,
            )
            .map_err(|e| eyre!("recovery failed: {e:?}"))?;
            let recover = timer.finish();

            let timer = Timer::start_cpu("decode");
            let expected = canonical
                .iter()
                .filter(|c| worker.setup.message(c.get()).is_some())
                .count();
            let messages = channel::decode_messages(
                &worker.params.channel,
                &sid,
                recovered.plaintext(),
                Some(expected),
            )
            .map_err(|e| eyre!("decoding failed: {e:?}"))?;
            Ok((messages, vec![recover, timer.finish()]))
        })
        .await
        .wrap_err("recovery task panicked")??;
    spans.extend(compute_spans);

    Ok(Json(RecipientReport {
        report: NodeReport {
            node: "recipient".into(),
            spans,
            usage: metrics::take_usage(),
            link: metrics::Link::from_env(),
        },
        messages: messages
            .iter()
            .map(|m| String::from_utf8_lossy(m).into_owned())
            .collect(),
    }))
}
