use std::time::{Duration, Instant};

use eyre::{bail, ensure, WrapErr as _};
use serde::de::DeserializeOwned;
use tokio::task::JoinSet;

use crate::http;
use crate::report::RoundReport;
use crate::setup::Setup;

#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long, env = "SIM_BULLETIN")]
    pub(crate) bulletin: String,
    /// Server urls, in server id order.
    #[arg(long, env = "SIM_SERVER_URLS", value_delimiter = ',')]
    pub(crate) servers: Vec<String>,
    /// Client urls, in client id order.
    #[arg(long, env = "SIM_CLIENT_URLS", value_delimiter = ',')]
    pub(crate) clients: Vec<String>,
    #[arg(long, env = "SIM_RECIPIENT")]
    pub(crate) recipient: String,
    #[arg(long, env = "SIM_ROUNDS", default_value_t = 1)]
    pub(crate) rounds: usize,
    /// How long to wait for every node to come up.
    #[arg(long, env = "SIM_READY_TIMEOUT_SECS", default_value_t = 60)]
    pub(crate) ready_timeout_secs: u64,
}

/// Runs `args.rounds` rounds one after another, printing one json line per
/// round to stdout.
pub(crate) async fn run(setup: &Setup, args: &Args) -> eyre::Result<Vec<RoundReport>> {
    ensure!(
        args.servers.len() == setup.servers,
        "got {} server urls for {} servers",
        args.servers.len(),
        setup.servers,
    );
    ensure!(
        args.clients.len() == setup.clients,
        "got {} client urls for {} clients",
        args.clients.len(),
        setup.clients,
    );
    let http = reqwest::Client::new();
    wait_ready(&http, args).await?;

    let mut expected: Vec<String> = (0..setup.clients as u32)
        .filter_map(|i| setup.message(i))
        .map(|m| String::from_utf8_lossy(&m).into_owned())
        .collect();
    expected.sort();

    let mut reports = Vec::with_capacity(args.rounds);
    for round in 0..args.rounds {
        let session = hex::encode(rand::random::<[u8; 32]>());
        tracing::info!(round, %session, "starting round");
        let round_start = Instant::now();

        let start = Instant::now();
        let clients = start_all(&http, &args.clients, &session)
            .await
            .wrap_err("client phase failed")?;
        let clients_ms = start.elapsed().as_millis();

        // every client is honest here, so all of them make the canonical set
        let canonical: Vec<u32> = (0..setup.clients as u32).collect();
        let url = format!("{}/{session}/canonical", args.bulletin);
        let resp = http
            .post(&url)
            .header("content-type", "application/json")
            .body(serde_json::to_vec(&canonical).wrap_err("failed to encode canonical set")?)
            .send()
            .await
            .wrap_err_with(|| format!("failed to send request to {url}"))?;
        ensure!(
            resp.status().is_success(),
            "{url} answered {}",
            resp.status()
        );

        let start = Instant::now();
        let servers = start_all(&http, &args.servers, &session)
            .await
            .wrap_err("server phase failed")?;
        let servers_ms = start.elapsed().as_millis();

        let start = Instant::now();
        let recipient: crate::report::RecipientReport =
            http::post_for_json(&http, format!("{}/round/{session}", args.recipient))
                .await
                .wrap_err("recipient phase failed")?;
        let recipient_ms = start.elapsed().as_millis();

        if recipient.messages != expected {
            bail!(
                "round {round} recovered {:?}, expected {:?}",
                recipient.messages,
                expected
            );
        }

        let report = RoundReport {
            round,
            session,
            clients_ms,
            servers_ms,
            recipient_ms,
            total_ms: round_start.elapsed().as_millis(),
            clients,
            servers,
            recipient,
        };
        println!(
            "{}",
            serde_json::to_string(&report).wrap_err("failed to encode report")?
        );
        reports.push(report);
    }
    Ok(reports)
}

async fn wait_ready(http: &reqwest::Client, args: &Args) -> eyre::Result<()> {
    let nodes: Vec<&String> = std::iter::once(&args.bulletin)
        .chain(&args.servers)
        .chain(&args.clients)
        .chain(std::iter::once(&args.recipient))
        .collect();
    let deadline = Instant::now() + Duration::from_secs(args.ready_timeout_secs);
    for node in nodes {
        let url = format!("{node}/health");
        while !http
            .get(&url)
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            ensure!(Instant::now() < deadline, "{node} did not come up in time");
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
    tracing::info!("all nodes are up");
    Ok(())
}

/// Starts the round on every node at once and collects the reports in node order.
async fn start_all<T: DeserializeOwned + Send + 'static>(
    http: &reqwest::Client,
    nodes: &[String],
    session: &str,
) -> eyre::Result<Vec<T>> {
    let mut tasks = JoinSet::new();
    for (i, node) in nodes.iter().enumerate() {
        let http = http.clone();
        let url = format!("{node}/round/{session}");
        tasks.spawn(async move { (i, http::post_for_json::<T>(&http, url).await) });
    }
    let mut out: Vec<Option<T>> = (0..nodes.len()).map(|_| None).collect();
    while let Some(joined) = tasks.join_next().await {
        let (i, report) = joined.wrap_err("round task panicked")?;
        out[i] = Some(report?);
    }
    Ok(out.into_iter().flatten().collect())
}
