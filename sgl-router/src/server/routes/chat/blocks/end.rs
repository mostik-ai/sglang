// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::super::forward::SelectedWorkers;
use axum::http::{header::AUTHORIZATION, HeaderMap, HeaderValue};
use reqwest::Client;
use std::time::Duration;

const END_CHAIN_PATH: &str = "/franky/end_chain";
const END_CHAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Ends a chain on every engine it reached, when it is dropped: the engines
/// abort the chain's requests and release its holds. A chain that ended with
/// its last block left no hold, and needs no end.
pub(super) struct ChainEnd {
    client: Client,
    chain_id: String,
    authorization: Option<HeaderValue>,
    engine_urls: Vec<String>,
    needed: bool,
}

impl ChainEnd {
    pub(super) fn new(client: Client, chain_id: String, headers: &HeaderMap) -> Self {
        Self {
            client,
            chain_id,
            authorization: headers.get(AUTHORIZATION).cloned(),
            engine_urls: Vec::new(),
            needed: true,
        }
    }

    /// Remember the engines a block went to.
    pub(super) fn reach(&mut self, workers: &SelectedWorkers) {
        let engines = [
            Some(&workers.prefill),
            workers.decode.as_ref(),
            workers.sandy.as_ref(),
        ];
        for worker in engines.into_iter().flatten() {
            if !self.engine_urls.contains(&worker.url) {
                self.engine_urls.push(worker.url.clone());
            }
        }
    }

    pub(super) fn ended_with_last_block(&mut self) {
        self.needed = false;
    }
}

impl Drop for ChainEnd {
    fn drop(&mut self) {
        if !self.needed {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        for url in self.engine_urls.drain(..) {
            let mut request = self
                .client
                .post(format!("{}{END_CHAIN_PATH}", url.trim_end_matches('/')))
                .json(&serde_json::json!({ "chain_id": self.chain_id }))
                .timeout(END_CHAIN_TIMEOUT);
            if let Some(authorization) = &self.authorization {
                request = request.header(AUTHORIZATION, authorization);
            }
            runtime.spawn(async move {
                if let Err(error) = request.send().await.and_then(|r| r.error_for_status()) {
                    tracing::warn!(%url, %error, "ending a Franky chain on an engine failed");
                }
            });
        }
    }
}
