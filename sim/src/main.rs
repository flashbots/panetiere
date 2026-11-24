//! Runs one Panetière role as a networked node, so a round can be timed over
//! real (or shaped) links instead of in one process.

mod bulletin;
mod client;
mod coordinator;
mod http;
mod recipient;
mod report;
mod server;
mod setup;
#[cfg(test)]
mod tests;

use clap::{Parser, Subcommand};
use eyre::WrapErr as _;
use tracing_subscriber::EnvFilter;

use setup::Setup;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    setup: Setup,
    #[command(subcommand)]
    role: Role,
}

#[derive(Subcommand)]
enum Role {
    Bulletin(bulletin::Args),
    Server(server::Args),
    Client(client::Args),
    Recipient(recipient::Args),
    Coordinator(coordinator::Args),
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    match cli.role {
        Role::Bulletin(args) => bulletin::run(http::bind(args.listen).await?).await,
        Role::Server(args) => server::run(http::bind(args.listen).await?, cli.setup, args).await,
        Role::Client(args) => client::run(http::bind(args.listen).await?, cli.setup, args).await,
        Role::Recipient(args) => {
            recipient::run(http::bind(args.listen).await?, cli.setup, args).await
        }
        Role::Coordinator(args) => coordinator::run(&cli.setup, &args)
            .await
            .map(|_| ())
            .wrap_err("coordinator failed"),
    }
}
