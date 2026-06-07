# Implementation Plan: WeaveSelf

## Overview

This plan implements WeaveSelf through the requirements' three parallel build tracks plus a
shared-contracts foundation, in the design's contract-first order. Track 0 contracts are coded
first so every track can mock what it does not own. Track A (custom serving + method
reproduction) is the critical path and is built and verified before downstream integration;
Tracks B and C proceed against mocks. Each track has a standalone test and a standalone fallback
demo, and the work converges at the Integration Milestone.

Languages follow the design: Python (`hypothesis` for PBT) for Tracks A and B; TypeScript
(`fast-check` for PBT) for Track C. Property tests reference the design's Correctness Properties;
each property is its own optional sub-task placed next to the code it validates.

## Team Split (3 Engineers)

This plan is built for exactly three engineers working in parallel. Each engineer owns one track
plus that track's slice of the Track 0 shared contracts. Ownership of the top-level task numbers is
fixed as follows:

- **Engineer A — Track A (Model & Serving):** owns tasks **2** and **3**, plus the Track-A-owned
  Track 0 contracts **1.1** (Adapter_File format) and **1.5** (Inference API schemas). Owns the
  adapter file format and the inference API schema.
- **Engineer B — Track B (Data, Orchestration & Eval):** owns tasks **5**, **6**, and **7**, plus
  the Track-B-owned Track 0 contracts **1.3** (Training_Pair / Held_Out_Set schema) and **1.6**
  (Eval_Results schema). Owns the dataset/training-pair schema and the eval artifact schema.
- **Engineer C — Track C (Frontend, Redis & Integration):** owns tasks **9** and **10**, plus the
  Track-C-owned Track 0 contract **1.7** (Redis layout key helpers + client interface). Owns the
  Redis layout.
- **Shared / all engineers:** task **1** kickoff (lock the Track 0 contracts together), the
  checkpoints **4**, **8**, **11**, and **13**, and the Integration Milestone **12**.

### Parallelism

After the Track 0 contracts are locked in task **1**, Tracks A, B, and C run **fully in parallel**.
Each engineer codes against the locked contracts and mocks everything they do not own
(`Mock_Dependency` fixtures), so no engineer blocks another. The three tracks **converge only at the
Integration Milestone (task 12)**, where mocks are swapped for the real cross-track dependencies.

The **Task Dependency Graph** at the bottom of this file is the evidence of this interleaving:
after wave 0 locks the contracts (the `1.x` tasks), every subsequent wave mixes tasks from all three
tracks — for example wave 2 runs Track A's `2.2/2.3/3.1`, Track B's `5.2–5.7`, and Track C's
`9.2/10.1/10.2` side by side, and wave 3 interleaves `2.x/3.x` (A), `6.1/7.x` (B), and `9.x/10.x`
(C). The tracks only collapse into shared waves at `12.x`.

## Tasks

- [x] 1. Project structure and Track 0 shared contracts
  - Create the Python ML package layout (serving, training, data, orchestration, eval) and the
    Node/TS app layout (CopilotKit runtime, React frontend, Redis client)
  - Add `hypothesis` and `pytest` (Python) and `fast-check` + a test runner (TypeScript) to the
    respective dependency manifests
  - _Requirements: 22.1_

  - [x] 1.1 Implement the Adapter_File format and metadata validator (Track A-owned contract)
    - Write the safetensors + sidecar JSON writer/reader for `adapter_<id>.safetensors` and
      `adapter_<id>.json`
    - Implement an eight-field metadata model and a validator that raises `MissingFieldError`
      naming any missing required field
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ]* 1.2 Write property test for Adapter_File round-trip
    - **Property 1: Adapter_File round-trip preserves metadata and gate tensors**
    - **Validates: Requirements 1.1, 1.2, 1.3**

  - [x] 1.3 Implement the Training_Pair and Held_Out_Set schema validator (Track B-owned contract)
    - Write the JSONL Training_Pair model `{prompt, completion, unit_label}` and a row validator
      that rejects a row missing any field and reports the missing field name
    - _Requirements: 4.1, 4.4_

  - [ ]* 1.4 Write property test for missing data-schema field rejection
    - **Property 2: Missing data-schema field is rejected with the field name**
    - **Validates: Requirements 1.4, 4.4**

  - [x] 1.5 Implement the Inference API Pydantic request/response schemas (Track A-owned contract)
    - Define models for `/generate`, `/score`, `/adapters`, `/train` matching Requirement 2,
      including nullable `adapter_id`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 1.6 Implement the Eval_Results schema and validator (Track B-owned contract)
    - Write the `eval_results.json` model with `perplexity`, `confusion_matrix`, `size_bytes`,
      `examples` and a validator enforcing square matrix dimensions equal to label count
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 1.7 Implement the Redis layout key helpers and client interface (Track C-owned contract)
    - Define key builders for `adapter:blob:<id>`, `adapter:meta:<id>`, `adapter:index`,
      `interactions:<unit_label>` and the `RedisClientApi` TypeScript interface
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 2. Track A — Serving Engine (critical path) — _Owner: Engineer A / Track A_
  - [x] 2.1 Implement Base_Model loading and the resident serving engine
    - Load the instruct Base_Model exactly once per process using the `base_model` metadata id;
      cache loaded gate tensors keyed by `adapter_id`
    - _Requirements: 6.1, 7.1_

  - [x] 2.2 Implement per-request gate application and generation/scoring
    - Apply an adapter's gate tensors via forward hooks for a single request and always clear
      them in a `finally` block; generate pure base when `adapter_id` is null; compute
      teacher-forced NLL and perplexity in `score`
    - _Requirements: 6.3, 7.2, 7.3, 8.2_

  - [x] 2.3 Implement adapter listing and unknown-adapter error handling
    - Expose the loadable `adapter_id` set; raise `AdapterNotLoadable` naming a missing
      `adapter_id` before any forward-pass mutation
    - _Requirements: 2.3, 7.5_

  - [ ]* 2.4 Write property test for null-adapter equivalence to base
    - **Property 4: Null adapter equals the pure Base_Model**
    - **Validates: Requirements 2.5, 7.3**

  - [ ]* 2.5 Write property test for adapter output differing from base
    - **Property 5: Adapter output differs from base for every prompt**
    - **Validates: Requirements 7.2, 7.4, 10.1**

  - [ ]* 2.6 Write property test for score non-negativity and consistency
    - **Property 8: Score is non-negative and internally consistent**
    - **Validates: Requirements 8.2**

  - [ ]* 2.7 Write property test for /adapters reflecting the loadable set
    - **Property 9: /adapters reflects the loadable set**
    - **Validates: Requirements 2.3, 8.3**

- [x] 3. Track A — Train_Adapter and Inference API — _Owner: Engineer A / Track A_
  - [x] 3.1 Implement the NKT-Mirror `train_adapter` training loop
    - Train ~5K activation-gating parameters on the frozen instruct base without updating base
      weights; write the Adapter_File pair with `train_rows` equal to consumed rows and
      `unit_label`/`unit_type` equal to arguments; serialize to ≤ 200,000 bytes and set
      `size_bytes`
    - _Requirements: 6.2, 6.3, 6.4, 9.1, 9.2_

  - [x] 3.2 Implement train_adapter input error handling
    - Raise `DatasetNotReadable` naming an unreadable `dataset_path` and `InsufficientTrainingData`
      for a zero-row dataset, writing no Adapter_File in either case
    - _Requirements: 9.4, 9.5_

  - [ ]* 3.3 Write property test for no base-weight mutation during training
    - **Property 6: Training never mutates base weights**
    - **Validates: Requirements 6.3**

  - [ ]* 3.4 Write property test for bounded adapter size
    - **Property 7: Adapter size is bounded**
    - **Validates: Requirements 6.4**

  - [ ]* 3.5 Write property test for train metadata reflecting inputs
    - **Property 10: Train metadata reflects inputs**
    - **Validates: Requirements 9.1, 9.2**

  - [ ]* 3.6 Write property test for unreadable dataset path
    - **Property 11: Unreadable dataset path errors and produces no adapter**
    - **Validates: Requirements 9.4**

  - [x] 3.7 Implement the FastAPI Inference_API service
    - Wire `/generate`, `/score`, `/adapters`, `/train` to the Serving_Engine and `train_adapter`,
      returning populated `tokens`/`latency_ms`; ensure `POST /train` returns the same
      `adapter_path` as the direct call; return HTTP 422 naming the offending field on malformed
      bodies
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 9.3_

  - [ ]* 3.8 Write property test for malformed API request rejection
    - **Property 3: Malformed API request is rejected naming the offending field**
    - **Validates: Requirements 8.4**

  - [ ]* 3.9 Write Track A standalone test and CLI fallback demo
    - Train one adapter on a tiny local dataset and serve base vs adapter on the same prompt
      showing visibly different output, with no dependency on Redis, the batch graph, or frontend
    - _Requirements: 10.1, 10.2, 23.1_

- [ ] 4. Checkpoint - Track A verified
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Track B — Data pipeline and GPT curation — _Owner: Engineer B / Track B_
  - [x] 5.1 Implement the Data_Pipeline split builder
    - Load per-Unit source data; emit `train.jsonl` and `heldout.jsonl` of Training_Pairs with
      `unit_label` matching the Unit; guarantee no train/held-out overlap; include Units with
      `>= min_rows` and exclude+record those below
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 4.2, 4.3_

  - [x]* 5.2 Write property test for valid training-pair schema and labels
    - **Property 12: Training pairs carry valid schema and correct unit labels**
    - **Validates: Requirements 4.1, 4.2, 11.1, 11.2**

  - [x]* 5.3 Write property test for non-overlapping splits
    - **Property 13: Train and held-out splits do not overlap**
    - **Validates: Requirements 4.3, 11.3**

  - [x]* 5.4 Write property test for minimum-rows inclusion threshold
    - **Property 14: Unit inclusion respects the minimum-rows threshold**
    - **Validates: Requirements 11.4**

  - [x] 5.5 Implement the GPT_Curation_Node
    - Turn raw interactions into Training_Pairs conforming to Requirement 4 as the only GPT caller;
      support a swappable local curation model emitting the same schema; discard uncurable
      interactions and record the discarded count
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [x]* 5.6 Write property test for curation schema conformance
    - **Property 15: Curation output conforms to the Training_Pair schema**
    - **Validates: Requirements 12.1, 12.3**

  - [x]* 5.7 Write property test for curation count conservation
    - **Property 16: Curation conserves and accounts for every interaction**
    - **Validates: Requirements 12.4**

- [x] 6. Track B — LangGraph nightly-batch graph — _Owner: Engineer B / Track B_
  - [x] 6.1 Implement the batch graph node sequence and state
    - Build the `collect → curate → train → eval → store` state machine; the `train` node invokes
      Track A's `train_adapter` (mocked until integration) with curated path + labels; the `store`
      node persists each Adapter_File and metadata through the Redis_Client_API
    - _Requirements: 13.1, 13.2, 13.3_

  - [x] 6.2 Implement batch-only execution and chat-trigger blocking
    - Run as a batch job that never trains on a live chat request and blocks chat-triggered graph
      execution while a batch run is in progress
    - _Requirements: 13.4, 13.5_

  - [x] 6.3 Implement per-unit failure isolation and critical-halt handling
    - Record failing node + `unit_label` and continue remaining Units; halt when failure recording
      itself fails or a critical error prevents continuation
    - _Requirements: 13.6, 13.7_

  - [x]* 6.4 Write property test for fixed node execution order
    - **Property 17: Batch graph executes nodes in the fixed order**
    - **Validates: Requirements 13.1**

  - [x]* 6.5 Write property test for isolated and recorded per-unit failures
    - **Property 18: Per-unit failures are isolated and recorded**
    - **Validates: Requirements 13.6**

  - [x]* 6.6 Write property test for live chat never triggering training
    - **Property 19: Live chat never triggers training**
    - **Validates: Requirements 13.4**

- [x] 7. Track B — Weave eval — _Owner: Engineer B / Track B_
  - [x] 7.1 Implement perplexity and context-memory baseline evaluation
    - Score each Unit's Held_Out_Set under adapter and base (pass iff adapter < base, record both);
      run the context-memory baseline by injecting Unit examples into the prompt and record its
      perplexity (pass iff adapter ≤ baseline); log perplexity results to Weave/W&B
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [x] 7.2 Implement the confusion matrix, size chart, and artifact emission
    - Score each held-out set under every trained adapter, pick the lowest-perplexity adapter as
      the predicted Unit, build the square Confusion_Matrix, record `size_bytes` (nktmirror vs
      lora), and emit `eval_results.json` conforming to Requirement 5; optionally run the
      fact-capacity test recording held-out recall as N grows
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

  - [x]* 7.3 Write property test for the personalization pass decision
    - **Property 20: Personalization pass decision**
    - **Validates: Requirements 14.1, 14.2**

  - [x]* 7.4 Write property test for the competitive comparison decision
    - **Property 21: Competitive comparison pass decision**
    - **Validates: Requirements 14.4**

  - [x]* 7.5 Write property test for predicted unit being the minimum-perplexity adapter
    - **Property 22: Predicted unit is the minimum-perplexity adapter**
    - **Validates: Requirements 15.1**

  - [x]* 7.6 Write property test for a well-formed confusion matrix
    - **Property 23: Confusion matrix is well-formed**
    - **Validates: Requirements 15.2, 5.3**

  - [x]* 7.7 Write property test for eval artifact schema conformance
    - **Property 24: Eval artifact conforms to the schema**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 15.3, 15.4**

  - [x]* 7.8 Write Track B standalone test
    - Run the full graph against a Mock_Dependency for `train_adapter` and the inference API and
      emit an `eval_results.json` containing a Confusion_Matrix, with no frontend dependency
    - _Requirements: 16.1, 16.2_

- [ ] 8. Checkpoint - Track B verified
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Track C — Redis layer and client API — _Owner: Engineer C / Track C_
  - [x] 9.1 Implement the Redis_Layer blob and metadata storage
    - Store adapter bytes (or disk path) under `adapter:blob:<id>` and metadata JSON under
      `adapter:meta:<id>`; support fetching metadata independently of the blob and round-tripping
      a 100 KB blob byte-identically
    - _Requirements: 3.1, 3.2, 19.1, 19.2, 19.4_

  - [x] 9.2 Implement the vector index, Route_Function, and interaction queue
    - Maintain `adapter:index` of `unit_label` embeddings, return the top-1 `adapter_id` from
      `route(query_or_user)`, and append raw interactions under `interactions:<unit_label>`
    - _Requirements: 3.3, 3.4, 3.5, 19.3, 19.5_

  - [x]* 9.3 Write property test for blob round-trip byte preservation
    - **Property 25: Redis blob round-trip preserves bytes**
    - **Validates: Requirements 3.1, 19.1, 19.4, 20.2**

  - [x]* 9.4 Write property test for metadata retrievable independently of the blob
    - **Property 26: Metadata is retrievable independently of the blob**
    - **Validates: Requirements 3.2, 19.2**

  - [x]* 9.5 Write property test for top-1 route by vector similarity
    - **Property 27: Route returns the top-1 adapter by vector similarity**
    - **Validates: Requirements 3.5, 19.3**

  - [x]* 9.6 Write property test for interactions appending under the unit key
    - **Property 28: Interactions append under the unit key**
    - **Validates: Requirements 3.4, 19.5**

- [x] 10. Track C — Frontend, dashboard, and AG-UI bridge — _Owner: Engineer C / Track C_
  - [x] 10.1 Implement the CopilotKit chat view
    - Let a user select a Unit and send a message; call the Inference_API and display the
      generated response
    - _Requirements: 17.1_

  - [x] 10.2 Implement the Dashboard views from eval_results.json
    - Render the adapter library (listing `adapter_id`, `unit_label`, `size_bytes`, hiding any
      zero-size adapter), the confusion-matrix heatmap, base-vs-adapter example pairs with
      reference text, and the size chart
    - _Requirements: 17.2, 17.3, 17.4, 17.5_

  - [x]* 10.3 Write property test for adapter-library zero-size filtering
    - **Property 29: Adapter library filters zero-size adapters**
    - **Validates: Requirements 17.2**

  - [x] 10.4 Implement the AG_UI_Bridge between the Node/TS runtime and the Python LangGraph agent
    - Connect over AG-UI SSE, stream agent responses to the frontend, show a connection error when
      the bridge is unavailable, and clear it on restore
    - _Requirements: 18.1, 18.2, 18.3, 18.4_

  - [x]* 10.5 Write Track C standalone test and fallback demo
    - Load a mock `eval_results.json` fixture and render the heatmap, example pairs, size chart,
      and adapter library; store/fetch/route a dummy 100 KB blob through the Redis_Client_API; run
      with no dependency on Track A serving or the live batch graph
    - _Requirements: 20.1, 20.2, 20.3, 23.3_

- [x] 11. Checkpoint - Track C verified
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Integration milestone and fallback demos — _Owner: All engineers (A + B + C converge)_
  - [ ] 12.1 Replace mocks with real cross-track dependencies and wire the end-to-end loop
    - Swap each Mock_Dependency for the real Track A inference API, real Track B adapters and eval
      artifacts, and the real Track C Redis_Layer and Frontend_App; serve each batch-produced
      Adapter_File through the Inference_API by `adapter_id` retrieved via the Redis_Client_API
    - _Requirements: 21.1, 21.2_

  - [ ] 12.2 Pre-bake demo adapters and wire Unit-selection routing to proof visuals
    - Pre-bake Adapter_Files for the demo `day_index` values; on Unit selection, route to the
      correct Adapter, generate a response, and display the corresponding `eval_results.json`
      proof visuals
    - _Requirements: 21.3, 21.4_

  - [ ] 12.3 Implement critical-path governance and blocked-state fallback
    - Keep mocks wired while Track A serving is unverified; record the blocked-integration state
      when serving cannot be verified; provide the Track B confusion-matrix demo and the per-track
      fallback demo entry points
    - _Requirements: 22.1, 22.2, 22.3, 23.2, 23.4_

  - [ ]* 12.4 Write integration tests for the end-to-end demo loop
    - Test route → serve → display with proof visuals on the demo Units and the AG-UI bridge
      handshake/streaming between Node/TS and Python
    - _Requirements: 21.1, 21.2, 21.3, 21.4, 18.1, 18.2_

- [ ] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; they cover property,
  unit, integration, and standalone tests.
- Property tests use `hypothesis` (Python, Tracks A/B) and `fast-check` (TypeScript, Track C),
  each running a minimum of 100 generated cases and tagged
  `Feature: weaveself, Property {number}: {property_text}`.
- Serving-related properties (4, 5, 6, 8) use a small/stub model and mocked gate tensors;
  perplexity-decision properties (20, 21, 22, 23) feed generated numeric score matrices into the
  pure decision and matrix-construction functions to keep iteration cheap.
- Track A is the critical path and is verified first; Tracks B and C run against mocks built to
  the Track 0 contracts until the Integration Milestone.
- Each task references specific requirement sub-clauses for traceability; checkpoints ensure
  incremental validation per track.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.3", "1.5", "1.6", "1.7"] },
    { "id": 1, "tasks": ["1.2", "1.4", "2.1", "5.1", "5.5", "9.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.1", "5.2", "5.3", "5.4", "5.6", "5.7", "9.2", "10.1", "10.2"] },
    { "id": 3, "tasks": ["2.4", "2.5", "2.6", "2.7", "3.2", "3.7", "6.1", "7.1", "7.2", "9.3", "9.4", "9.5", "9.6", "10.3", "10.4"] },
    { "id": 4, "tasks": ["3.3", "3.4", "3.5", "3.6", "3.8", "3.9", "6.2", "6.3", "7.3", "7.4", "7.5", "7.6", "7.7", "10.5"] },
    { "id": 5, "tasks": ["6.4", "6.5", "6.6", "7.8", "12.1"] },
    { "id": 6, "tasks": ["12.2", "12.3"] },
    { "id": 7, "tasks": ["12.4"] }
  ]
}
```
