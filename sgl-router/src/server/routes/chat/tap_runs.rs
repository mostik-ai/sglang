// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use serde_json::{json, Value};
use std::ops::Range;

/// Which taps the forward that makes each token of a Ricky request reads: from
/// Ricky's token `start` on, up to the next run, the first `taps` of Sandy's
/// tokens. Tokens before the first run read none. The engine gets a request's
/// runs as `franky_tap_runs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TapRun {
    pub(super) start: usize,
    pub(super) taps: usize,
}

/// The runs of a request's earlier answers: an answer's tokens read Sandy's
/// tokens before that answer, the tokens after it none.
///
/// `sandy_spans` are the same answers in Sandy's tokens, one per answer;
/// `None` when Sandy reads Ricky's tokens.
pub(super) fn earlier_answer_runs(
    spans: &[Range<usize>],
    sandy_spans: Option<&[Range<usize>]>,
) -> Vec<TapRun> {
    let sandy_starts = sandy_spans.map_or_else(
        || spans.iter().map(|span| span.start).collect::<Vec<_>>(),
        |sandy| sandy.iter().map(|span| span.start).collect(),
    );
    spans
        .iter()
        .zip(sandy_starts)
        .flat_map(|(span, sandy_start)| {
            [
                TapRun {
                    start: span.start,
                    taps: sandy_start,
                },
                TapRun {
                    start: span.end,
                    taps: 0,
                },
            ]
        })
        .collect()
}

/// `runs`, in order, where a run gives way to the next one when both start
/// at the same token.
pub(super) fn joined(runs: impl IntoIterator<Item = TapRun>) -> Vec<TapRun> {
    let mut joined: Vec<TapRun> = Vec::new();
    for run in runs {
        if joined.last().is_some_and(|last| last.start == run.start) {
            joined.pop();
        }
        joined.push(run);
    }
    joined
}

/// `runs` as the engine reads them: `[start, taps]` pairs.
pub(super) fn wire(runs: &[TapRun]) -> Value {
    json!(runs
        .iter()
        .map(|run| [run.start, run.taps])
        .collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_earlier_answer_reads_sandys_tokens_before_it_and_what_follows_none() {
        let runs = earlier_answer_runs(&[2..5, 9..12], None);

        assert_eq!(wire(&runs), json!([[2, 2], [5, 0], [9, 9], [12, 0]]));
    }

    #[test]
    fn a_run_gives_way_to_one_that_starts_at_the_same_token() {
        let runs = joined([
            TapRun { start: 2, taps: 2 },
            TapRun { start: 5, taps: 0 },
            TapRun { start: 5, taps: 5 },
        ]);

        assert_eq!(wire(&runs), json!([[2, 2], [5, 5]]));
    }

    #[test]
    fn with_sandys_own_tokens_an_answer_reads_up_to_its_start_in_them() {
        let runs = earlier_answer_runs(&[2..5, 9..12], Some(&[1..3, 6..8]));

        assert_eq!(wire(&runs), json!([[2, 1], [5, 0], [9, 6], [12, 0]]));
    }
}
