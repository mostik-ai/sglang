// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::forward::SelectedWorkers;
use super::preparation::generate_room_id;
use super::tap_runs::{earlier_answer_runs, joined, wire, TapRun};
use crate::server::error::ApiError;
use crate::tokenizer::franky_sandy::SandyChat;
use crate::workers::Worker;
use serde_json::{json, Value};
use std::ops::Range;

/// What a request carries for Franky workers.
#[derive(Debug, Default)]
pub(super) struct FrankyRequest {
    /// Where the request's taps come from.
    pub(super) tap_source: TapSource,
    /// The input of a Sandy that tokenizes on its own, which also keys
    /// Ricky's taps.
    sandy_ids: Option<Vec<u32>>,
    /// The tap runs of the request's earlier answers.
    earlier_runs: Vec<TapRun>,
    /// The tap runs Ricky reads; empty for the engine's own, where the answer
    /// after the input reads all taps.
    tap_runs: Vec<TapRun>,
    /// The room of the attempt in flight; each attempt opens its own.
    room: Option<FrankyRoom>,
}

impl FrankyRequest {
    /// A chain's Sandy call, which sends its taps when the chain reads them.
    pub(super) fn sandy_call(taps: bool) -> Self {
        let tap_source = if taps {
            TapSource::SandyCall
        } else {
            TapSource::None
        };
        Self {
            tap_source,
            ..Self::default()
        }
    }

    /// A chain's Ricky leg, which reads the taps from `tap_source` as
    /// `tap_runs` say.
    pub(super) fn ricky_leg(
        tap_source: TapSource,
        tap_runs: Vec<TapRun>,
        sandy_ids: Option<Vec<u32>>,
    ) -> Self {
        Self {
            tap_source,
            sandy_ids,
            tap_runs,
            ..Self::default()
        }
    }

    /// Takes a client's `franky_generated_spans`, positions of its earlier
    /// answers in `input_ids`, out of `body`.
    ///
    /// Returns: the request that reads them as tap runs; `None` without spans.
    /// Errors: `ApiError::BadRequest` for spans without `input_ids`, or that are
    /// not `[start, end)` pairs.
    pub(super) fn of_client_spans(body: &mut Value) -> Result<Option<Self>, ApiError> {
        let Some(spans) = body
            .as_object_mut()
            .and_then(|fields| fields.remove("franky_generated_spans"))
            .filter(|spans| !spans.is_null())
        else {
            return Ok(None);
        };
        let spans: Vec<[usize; 2]> = serde_json::from_value(spans).map_err(|_| {
            ApiError::BadRequest("franky_generated_spans must be [start, end) pairs".into())
        })?;
        let Some(input_len) = body["input_ids"].as_array().map(Vec::len) else {
            return Err(ApiError::BadRequest(
                "franky_generated_spans are positions in input_ids, so a request that \
                 sends them must send input_ids too"
                    .into(),
            ));
        };
        let spans: Vec<Range<usize>> = spans.iter().map(|[start, end]| *start..*end).collect();
        let mut request = Self::default();
        request.add_earlier_answers(&spans, input_len);
        Ok(Some(request))
    }

    /// Marks the earlier answers at `spans` of an input of `input_len` tokens:
    /// Ricky reads them as tap runs, and the answer after the input all taps.
    pub(super) fn add_earlier_answers(&mut self, spans: &[Range<usize>], input_len: usize) {
        self.set_earlier_runs(earlier_answer_runs(spans, None), input_len, input_len);
    }

    /// Takes the input of a Sandy that tokenizes on its own from `sandy`, and
    /// the tap runs of the earlier answers at `generated_spans` of Ricky's
    /// input of `input_len` tokens.
    pub(super) fn add_sandy_chat(
        &mut self,
        generated_spans: &[Range<usize>],
        input_len: usize,
        sandy: SandyChat,
    ) {
        if !generated_spans.is_empty() {
            let earlier = earlier_answer_runs(generated_spans, Some(&sandy.generated_spans));
            self.set_earlier_runs(earlier, input_len, sandy.input_ids.len());
        }
        self.sandy_ids = Some(sandy.input_ids);
    }

    fn set_earlier_runs(&mut self, earlier_runs: Vec<TapRun>, input_len: usize, num_taps: usize) {
        let answer = TapRun {
            start: input_len,
            taps: num_taps,
        };
        self.tap_runs = joined(earlier_runs.iter().copied().chain([answer]));
        self.earlier_runs = earlier_runs;
    }

    pub(super) fn earlier_runs(&self) -> &[TapRun] {
        &self.earlier_runs
    }

    pub(super) fn sandy_ids(&self) -> Option<&[u32]> {
        self.sandy_ids.as_deref()
    }

    pub(super) fn take_sandy_ids(&mut self) -> Option<Vec<u32>> {
        self.sandy_ids.take()
    }

    /// Opens the room of an attempt on `workers`, in place of the last
    /// attempt's.
    pub(super) fn open_room(&mut self, workers: &SelectedWorkers) {
        self.room = self.tap_source.room_on(workers);
    }

    pub(super) fn room(&self) -> Option<&FrankyRoom> {
        self.room.as_ref()
    }

    pub(super) fn take_room(&mut self) -> Option<FrankyRoom> {
        self.room.take()
    }

    /// The Franky fields of an attempt's engine body: its room and, for a
    /// request that reads taps, its tap runs and Sandy's input.
    pub(super) fn engine_fields(&self) -> Vec<(&'static str, Value)> {
        let mut fields = Vec::new();
        if let Some(room) = &self.room {
            fields.push(("franky_room", room.wire()));
        }
        if !self.tap_source.reads_taps() {
            return fields;
        }
        if !self.tap_runs.is_empty() {
            fields.push(("franky_tap_runs", wire(&self.tap_runs)));
        }
        if let Some(sandy_ids) = &self.sandy_ids {
            fields.push(("franky_sandy_input_ids", json!(sandy_ids)));
        }
        fields
    }

    /// The Franky fields of the body of an attempt's Sandy leg, which sends
    /// the taps of the prompt to the attempt's room and writes nothing.
    pub(super) fn sandy_leg_fields(&self) -> Vec<(&'static str, Value)> {
        let room = self.room.iter().map(|room| ("franky_room", room.wire()));
        room.chain([("franky_taps_only", true.into())]).collect()
    }
}

/// Where a request's taps meet: its Sandy's tap bootstrap server and a room,
/// sent as `franky_room` to the Sandy and every Ricky request reading them.
#[derive(Clone, Debug)]
pub(super) struct FrankyRoom {
    pub(super) host: String,
    pub(super) port: Option<u16>,
    pub(super) room: u64,
}

impl FrankyRoom {
    /// A new room at `sandy`'s tap bootstrap server.
    pub(super) fn at(sandy: &Worker) -> Self {
        Self {
            host: sandy.bootstrap_host().to_string(),
            port: sandy.bootstrap_port(),
            room: generate_room_id(),
        }
    }

    fn wire(&self) -> Value {
        json!({"bootstrap_host": self.host, "bootstrap_port": self.port, "room": self.room})
    }
}

/// Where a request's taps come from, or for a Sandy call, where they go.
#[derive(Clone, Debug, Default)]
pub(super) enum TapSource {
    /// It reads none.
    #[default]
    None,
    /// A Sandy leg beside it, in a room each attempt opens, sends the taps of
    /// its prompt.
    SandyLeg,
    /// A Sandy call already ran; its room holds the taps.
    Room(FrankyRoom),
    /// The request is a Sandy call: its Sandy sends the taps of all it runs to
    /// a room each attempt opens there.
    SandyCall,
}

impl TapSource {
    /// How many of `max_attempts` a request gets: one when it reads the room
    /// of a Sandy call, which sends its taps once.
    pub(super) fn attempts(&self, max_attempts: u32) -> u32 {
        match self {
            Self::Room(_) => 1,
            Self::None | Self::SandyLeg | Self::SandyCall => max_attempts,
        }
    }

    /// Whether each attempt picks a Sandy for a leg beside its workers.
    pub(super) fn needs_sandy_leg(&self) -> bool {
        matches!(self, Self::SandyLeg)
    }

    fn reads_taps(&self) -> bool {
        matches!(self, Self::SandyLeg | Self::Room(_))
    }

    fn room_on(&self, workers: &SelectedWorkers) -> Option<FrankyRoom> {
        match self {
            Self::None => None,
            Self::SandyLeg => workers.sandy.as_deref().map(FrankyRoom::at),
            Self::Room(room) => Some(room.clone()),
            Self::SandyCall => Some(FrankyRoom::at(
                workers.decode.as_deref().unwrap_or(&workers.prefill),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn room() -> FrankyRoom {
        FrankyRoom {
            host: "h".into(),
            port: Some(9),
            room: 42,
        }
    }

    #[test]
    fn a_request_sends_its_tap_runs_and_sandys_input_only_when_it_reads_taps() {
        let runs = vec![TapRun { start: 4, taps: 4 }];
        for (tap_source, reads) in [(TapSource::Room(room()), true), (TapSource::None, false)] {
            let mut request = FrankyRequest::ricky_leg(tap_source, runs.clone(), Some(vec![7]));
            request.room = Some(room());

            let fields: Vec<_> = (request.engine_fields().into_iter())
                .map(|(name, _)| name)
                .collect();

            let expected: &[&str] = if reads {
                &["franky_room", "franky_tap_runs", "franky_sandy_input_ids"]
            } else {
                &["franky_room"]
            };
            assert_eq!(fields, expected);
        }
    }

    #[test]
    fn a_clients_spans_become_the_runs_of_its_earlier_answers_and_its_answer() {
        let mut body = json!({"input_ids": [1, 2, 3, 4], "franky_generated_spans": [[1, 3]]});

        let request = FrankyRequest::of_client_spans(&mut body).unwrap().unwrap();

        assert!(body.get("franky_generated_spans").is_none());
        assert_eq!(
            request.earlier_runs(),
            [TapRun { start: 1, taps: 1 }, TapRun { start: 3, taps: 0 }]
        );
        assert_eq!(wire(&request.tap_runs), json!([[1, 1], [3, 0], [4, 4]]));
        assert!(
            FrankyRequest::of_client_spans(&mut json!({"input_ids": [1]}))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_room_holds_a_request_to_one_attempt() {
        assert_eq!(TapSource::Room(room()).attempts(3), 1);
        for tap_source in [TapSource::None, TapSource::SandyLeg, TapSource::SandyCall] {
            assert_eq!(tap_source.attempts(3), 3);
        }
    }
}
