//! Seed a scratch CAS with the upstream ingester, then serve via upstream LocalServer.
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use xet::xet_session::{Sha256Policy, XetSessionBuilder};
use xet_client::cas_client::{LocalServer, LocalServerConfig};
use xet_runtime::core::XetContext;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let port: u16 = args.next().context("port")?.parse()?;
    let data_directory = PathBuf::from(args.next().context("CAS scratch directory")?);
    let file = PathBuf::from(args.next().context("safetensors seed file")?);
    ensure!(
        port != 0 && data_directory.is_absolute() && args.next().is_none(),
        "usage: hfr_xet_fixture PORT ABSOLUTE_CAS_DIR FILE"
    );

    let session = XetSessionBuilder::new()
        .with_tokio_handle(tokio::runtime::Handle::current())
        .build()?;
    let commit = session
        .new_upload_commit()?
        .with_endpoint(format!("local://{}", data_directory.display()))
        .build()
        .await?;
    let handle = commit.upload_from_path(file, Sha256Policy::Compute).await?;
    let metadata = handle.finalize_ingestion().await?.xet_info;
    commit.commit().await?;
    drop(session);

    let ctx = XetContext::default().map_err(|e| anyhow::anyhow!("{e}"))?;
    let server = LocalServer::new(
        ctx,
        LocalServerConfig {
            // The local:// session uses its xet/xorbs directory for LocalClient records.
            data_directory: data_directory.join("xet").join("xorbs"),
            port,
            ..Default::default()
        },
    )
    .await?;
    println!("{}", serde_json::to_string(&metadata)?);
    server.run().await?;
    Ok(())
}
