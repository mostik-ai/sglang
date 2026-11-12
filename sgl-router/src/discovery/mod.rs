// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod k8s;
pub mod static_urls;
pub mod types;
pub use types::*;

use crate::config::{Config, DiscoveryBackend};
use anyhow::Result;
use tokio::sync::mpsc;

/// Channel capacity for discovery → registry events.  Bounded to 128 —
/// pod-add/remove is infrequent, but a bound prevents unbounded memory
/// growth under any pathological burst.
pub const DISCOVERY_CHANNEL_CAP: usize = 128;

/// A running discovery backend.
pub struct Discovery {
    /// The workers the backend adds and removes, for the worker manager.
    pub events: mpsc::Receiver<DiscoveryEvent>,
    /// The backend's task. The static_urls backend's task exits once the
    /// initial fan-out completes; the k8s backend's task runs for the lifetime
    /// of the watch.
    pub task: tokio::task::JoinHandle<()>,
    /// Where `/workers` sends the workers it adds and removes under
    /// [`DiscoveryBackend::WorkerApi`]; `None` otherwise.
    pub worker_api_events: Option<mpsc::Sender<DiscoveryEvent>>,
}

/// Spawn the configured discovery backend.
pub async fn spawn_discovery(cfg: &Config) -> Result<Discovery> {
    let (tx, events) = mpsc::channel(DISCOVERY_CHANNEL_CAP);
    let worker_api_events =
        matches!(cfg.discovery, DiscoveryBackend::WorkerApi(_)).then(|| tx.clone());
    let task = match &cfg.discovery {
        DiscoveryBackend::StaticUrls(s) | DiscoveryBackend::WorkerApi(s) => {
            static_urls::spawn(s.clone(), tx).await?
        }
        DiscoveryBackend::K8s(k) => k8s::spawn(k.clone(), tx).await?,
    };
    Ok(Discovery {
        events,
        task,
        worker_api_events,
    })
}
