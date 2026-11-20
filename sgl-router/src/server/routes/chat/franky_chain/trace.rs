// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::super::tap_runs::{joined, TapRun};
use crate::tokenizer::franky_sandy::FrankySandyTokens;
use anyhow::Result;
use std::sync::Arc;

/// A chain's trace as the engines read it: the prompt and every token written
/// since, by Sandy's calls and Ricky's legs, in Ricky's ids and, with a Sandy
/// that tokenizes on its own, in Sandy's ids too; and the tap runs that say
/// which taps each Ricky token reads.
pub(super) struct ChainTrace {
    prompt_ids: Vec<u32>,
    answer_ids: Vec<u32>,
    sandy: Option<SandyTrace>,
    earlier_runs: Vec<TapRun>,
    /// One run per Sandy call or tap leg, from where Ricky's tokens read what
    /// Sandy had run until then.
    sandy_runs: Vec<TapRun>,
}

/// Sandy's ids of a chain whose Sandy tokenizes on its own: its rendering of
/// the prompt, then its own ids of each Sandy call's text and its tokens of
/// each Ricky leg's text, so each call's input extends the last one's and
/// Sandy's cache keeps it.
struct SandyTrace {
    tokens: Arc<FrankySandyTokens>,
    prompt_ids: Vec<u32>,
    answer_ids: Vec<u32>,
}

impl ChainTrace {
    /// The trace of a chain that starts at `prompt_ids`.
    ///
    /// `sandy` is a Sandy that tokenizes on its own with its ids of the prompt;
    /// `earlier_runs` are the tap runs of the prompt's earlier answers.
    pub(super) fn of_prompt(
        prompt_ids: Vec<u32>,
        sandy: Option<(Arc<FrankySandyTokens>, Vec<u32>)>,
        earlier_runs: Vec<TapRun>,
    ) -> Self {
        Self {
            prompt_ids,
            answer_ids: Vec::new(),
            sandy: sandy.map(|(tokens, prompt_ids)| SandyTrace {
                tokens,
                prompt_ids,
                answer_ids: Vec::new(),
            }),
            earlier_runs,
            sandy_runs: Vec::new(),
        }
    }

    /// Ricky's ids of the whole trace.
    pub(super) fn ricky_ids(&self) -> Vec<u32> {
        [self.prompt_ids.as_slice(), self.answer_ids.as_slice()].concat()
    }

    /// Sandy's ids of the whole trace, when Sandy tokenizes on its own.
    pub(super) fn sandy_ids(&self) -> Option<Vec<u32>> {
        let sandy = self.sandy.as_ref()?;
        Some([sandy.prompt_ids.as_slice(), sandy.answer_ids.as_slice()].concat())
    }

    /// How many tokens the chain wrote, in Ricky's ids.
    pub(super) fn num_written(&self) -> u64 {
        self.answer_ids.len() as u64
    }

    /// The tap runs of a Ricky leg that starts now: the earlier answers', then
    /// one per Sandy call or tap leg.
    pub(super) fn tap_runs(&self) -> Vec<TapRun> {
        joined(self.earlier_runs.iter().chain(&self.sandy_runs).copied())
    }

    /// A Sandy call wrote `ricky_ids`, its text in Ricky's ids, and with a Sandy
    /// that tokenizes on its own `sandy_ids`, the same text in Sandy's. Its run
    /// starts where its text does and reads every token Sandy has then.
    pub(super) fn add_sandy_text(&mut self, ricky_ids: &[u32], sandy_ids: &[u32]) {
        let start = self.prompt_ids.len() + self.answer_ids.len();
        self.answer_ids.extend_from_slice(ricky_ids);
        if let Some(sandy) = &mut self.sandy {
            sandy.answer_ids.extend_from_slice(sandy_ids);
        }
        self.sandy_runs.push(TapRun {
            start,
            taps: self.num_sandy_tokens(),
        });
    }

    /// A Sandy leg that writes nothing runs the trace beside the next Ricky
    /// leg, whose tokens read every token Sandy has from here on.
    pub(super) fn start_tap_leg(&mut self) {
        self.add_sandy_text(&[], &[]);
    }

    /// A Ricky leg wrote `ricky_ids`, whose text is `text`.
    ///
    /// Errors: Sandy's tokenizer fails on the text.
    pub(super) fn add_ricky_text(&mut self, ricky_ids: &[u32], text: &str) -> Result<()> {
        self.answer_ids.extend_from_slice(ricky_ids);
        if let Some(sandy) = &mut self.sandy {
            sandy
                .answer_ids
                .extend(sandy.tokens.encode_answer_text(text)?);
        }
        Ok(())
    }

    fn num_sandy_tokens(&self) -> usize {
        match &self.sandy {
            Some(sandy) => sandy.prompt_ids.len() + sandy.answer_ids.len(),
            None => self.prompt_ids.len() + self.answer_ids.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_sandy_call_starts_a_run_reading_every_token_sandy_has() {
        let earlier = vec![TapRun { start: 1, taps: 1 }, TapRun { start: 2, taps: 0 }];
        let mut trace = ChainTrace::of_prompt(vec![1, 2, 3], None, earlier.clone());

        trace.add_sandy_text(&[4, 5], &[]);
        trace.add_ricky_text(&[6, 7, 8], "").unwrap();
        trace.start_tap_leg();

        assert_eq!(trace.ricky_ids(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(trace.num_written(), 5);
        assert_eq!(
            trace.tap_runs(),
            [
                earlier.as_slice(),
                &[TapRun { start: 3, taps: 5 }, TapRun { start: 8, taps: 8 }]
            ]
            .concat()
        );
        assert_eq!(trace.sandy_ids(), None);
    }
}
