use std::future::Future;

use tokio::net::TcpListener;

use crate::setup::Setup;
use crate::{bulletin, client, coordinator, recipient, server};

async fn spawn<F>(start: impl FnOnce(TcpListener) -> F) -> eyre::Result<String>
where
    F: Future<Output = eyre::Result<()>> + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("http://{}", listener.local_addr()?);
    tokio::spawn(start(listener));
    Ok(url)
}

#[tokio::test(flavor = "multi_thread")]
async fn every_role_over_localhost_recovers_the_messages() -> eyre::Result<()> {
    let setup = Setup {
        seed: 7,
        servers: 3,
        clients: 4,
        messages: 3,
        message_bytes: 32,
        min_clients: 4,
    };
    let any = "127.0.0.1:0".parse()?;

    let bulletin = spawn(bulletin::run).await?;
    let mut servers = Vec::new();
    for index in 0..setup.servers as u32 {
        let args = server::Args {
            listen: any,
            index,
            bulletin: bulletin.clone(),
        };
        let s = setup.clone();
        servers.push(spawn(move |l| server::run(l, s, args)).await?);
    }
    let mut clients = Vec::new();
    for index in 0..setup.clients as u32 {
        let args = client::Args {
            listen: any,
            index,
            bulletin: bulletin.clone(),
            servers: servers.clone(),
        };
        let s = setup.clone();
        clients.push(spawn(move |l| client::run(l, s, args)).await?);
    }
    let args = recipient::Args {
        listen: any,
        bulletin: bulletin.clone(),
    };
    let s = setup.clone();
    let recipient = spawn(move |l| recipient::run(l, s, args)).await?;

    let reports = coordinator::run(
        &setup,
        &coordinator::Args {
            bulletin,
            servers,
            clients,
            recipient,
            label: "test".into(),
            rounds: 2,
            ready_timeout_secs: 30,
        },
    )
    .await?;

    assert_eq!(reports.len(), 2);
    for r in &reports {
        // clients, servers, recipient and bulletin each report once
        assert_eq!(r.nodes.len(), setup.clients + setup.servers + 2);
        assert_eq!(r.phases.len(), 4);
        let recipient = r
            .nodes
            .iter()
            .find(|n| n.node == "recipient")
            .ok_or_else(|| eyre::eyre!("no recipient report"))?;
        let posts = recipient
            .spans
            .iter()
            .filter(|s| s.name.starts_with("download_post_"))
            .count();
        assert_eq!(posts, setup.clients);
    }
    Ok(())
}
