# Requirements Document

## Introduction

WeaveSelf is a local, overnight "weight-memory" personalization engine. A frozen open-source instruct model is specialized per user or per category by tiny (~100 KB) NKT-Mirror activation-gating adapters. Adapters are retrained in nightly batches from accumulated interactions, served locally by swapping gate tensors against a single resident base model, and proven by an objective evaluation: held-out perplexity, a cross-unit identification confusion matrix, and a comparison against a context-memory baseline at zero extra context cost.

This document is structured to support EXACTLY THREE independent, parallel build tracks plus a shared-contracts foundation, so three engineers never block one another. The shared contracts (Track 0) are defined FIRST. Each track owns one set of interfaces, consumes others (mocking what it does not own until integration), and can run a standalone test and a standalone fallback demo.

**Track map:**
- **Track 0 — Shared Contracts:** adapter file format, inference API schema, Redis layout, dataset/training-pair schema, eval artifact schema.
- **Track A — Model & Serving:** NKT-Mirror training loop on the instruct model; custom adapter serving; FastAPI inference API; `train_adapter`. Owns adapter format + inference API.
- **Track B — Data, Orchestration & Eval:** data pipeline; GPT curation node; LangGraph nightly-batch graph; Weave eval (perplexity, confusion matrix, baseline comparison, size chart); optional fact-capacity test.
- **Track C — Frontend, Redis & Integration:** CopilotKit React app; AG-UI ↔ LangGraph-Python wiring; Redis layer with a clean client API; end-to-end demo glue.

**Hard constraints reflected throughout (do not violate):**
- NKT-Mirror is activation gating, NOT LoRA; serving is custom (load frozen base once, swap ~100 KB gate tensors per request). Multi-LoRA serving stacks are out of scope.
- Training is batch/overnight, never live. The demo is time-compressed with pre-baked adapters.
- Build on the instruct model only.
- Personalize style and preferences, not arbitrary facts ("steer, not teach").
- Two-runtime reality: CopilotKit runtime is Node/TS, ML is Python, bridged via AG-UI and the LangGraph-Python SDK.
- Data-curation quality matters more than pipeline complexity. GPT is the curation node only.

## Glossary

- **WeaveSelf**: The complete personalization system described by this document.
- **Base_Model**: The frozen open-source instruct model. Default Qwen2.5-7B-Instruct; Qwen2.5-1.5B-Instruct is an accepted substitute when GPU memory is constrained.
- **NKT_Mirror_Adapter** (also "Adapter"): A per-channel activation-gating adapter with approximately 5,000 trainable parameters serialized to an approximately 100 KB artifact, applied on top of the frozen Base_Model.
- **Unit**: The personalization granularity for one adapter, identified by `unit_type` ("category" or "user") and a `unit_label`.
- **Adapter_File**: The serialized adapter, stored as `adapter_<id>.safetensors` with a sidecar `adapter_<id>.json` metadata file.
- **Inference_API**: The Track A FastAPI service exposing `/generate`, `/score`, `/adapters`, and `/train`.
- **Serving_Engine**: The Track A component that loads the Base_Model once and applies a selected Adapter's gate tensors per request.
- **Train_Adapter**: The Python function `train_adapter(dataset_path, unit_label, unit_type) -> adapter_path`, also callable as `POST /train`.
- **Data_Pipeline**: The Track B component that loads source data and produces train and held-out splits as training pairs.
- **Training_Pair**: A JSONL row of shape `{ "prompt": str, "completion": str, "unit_label": str }`.
- **Held_Out_Set**: Eval rows of the same shape as a Training_Pair, in a separate file, with no overlap with the train rows for the same Unit.
- **GPT_Curation_Node**: The single LangGraph node that uses GPT to turn raw interactions into clean Training_Pairs. GPT is used for curation only.
- **LangGraph_Batch_Graph**: The Track B nightly-batch orchestration graph executing collect → curate → train → eval → store.
- **Weave_Eval**: The Track B evaluation that produces held-out perplexity, the confusion matrix, the baseline comparison, and the size chart, logged to Weave/W&B.
- **Confusion_Matrix**: The cross-unit identification matrix where rows are the true Unit and columns are the adapter that scored that Unit's held-out text with the lowest perplexity.
- **Context_Memory_Baseline**: A baseline that injects a Unit's example text into the prompt context, used to compare against the Adapter at zero extra context cost.
- **Eval_Results**: The `eval_results.json` artifact consumed by the dashboard.
- **Redis_Layer**: The Track C Redis component providing the adapter blob store, vector routing, and interaction queue, exposed through a client API.
- **Redis_Client_API**: The clean client interface over the Redis_Layer used by Track A and Track B.
- **Route_Function**: The Redis_Layer helper `route(query_or_user) -> adapter_id` performing top-1 vector search over unit-label embeddings.
- **Frontend_App**: The Track C CopilotKit React application providing chat and the dashboard.
- **Dashboard**: The Frontend_App view rendering the adapter library, the confusion-matrix heatmap, base-versus-adapter examples, and the size chart.
- **AG_UI_Bridge**: The AG-UI (SSE) connection between the Node/TS CopilotKit runtime and the Python LangGraph agent.
- **Integration_Milestone**: The state in which Track A serving, Track B eval, and Track C UI/Redis operate end to end on real adapters.
- **Mock_Dependency**: A fixture or stub standing in for an interface a track consumes but does not own, used until the Integration_Milestone.

## Requirements

---

## Track 0 — Shared Contracts

> These interfaces are locked first. All three tracks code against them and mock what they do not own. Changes occur only by team agreement.

### Requirement 1: Adapter File Format Contract

**User Story:** As an engineer on any track, I want a fixed adapter file format, so that adapters produced by Track A can be stored, routed, served, and evaluated without coordination.

#### Acceptance Criteria

1. THE WeaveSelf SHALL define an Adapter_File as a pair of files named `adapter_<id>.safetensors` holding the NKT-Mirror gate tensors and `adapter_<id>.json` holding metadata.
2. THE WeaveSelf SHALL define the Adapter_File metadata fields as `adapter_id` (string), `base_model` (string), `unit_type` (string, either "category" or "user"), `unit_label` (string), `train_rows` (integer), `trained_at` (ISO 8601 timestamp string), `day_index` (integer), and `size_bytes` (integer).
3. WHEN an Adapter_File is produced, THE Serving_Engine SHALL load it without modification to the format.
4. IF an Adapter_File metadata object is missing a required field defined in this requirement, THEN THE consuming component SHALL reject the Adapter_File and report the missing field name.

### Requirement 2: Inference API Schema Contract

**User Story:** As a Track B and Track C engineer, I want a fixed inference API schema, so that I can code and mock against stable request and response shapes before Track A serving is ready.

#### Acceptance Criteria

1. THE Inference_API SHALL expose `POST /generate` accepting `{ "prompt": string, "adapter_id": string | null, "max_new_tokens": integer }` and returning `{ "text": string, "tokens": integer, "latency_ms": integer }`.
2. THE Inference_API SHALL expose `POST /score` accepting `{ "prompt": string, "target": string, "adapter_id": string | null }` and returning `{ "perplexity": number, "nll": number }`.
3. THE Inference_API SHALL expose `GET /adapters` returning a list of currently loadable `adapter_id` values.
4. THE Inference_API SHALL expose `POST /train` accepting a request that triggers `train_adapter` with `dataset_path`, `unit_label`, and `unit_type`, and returning the resulting `adapter_path`.
5. WHERE `adapter_id` is null in a `/generate` or `/score` request, THE Inference_API SHALL use the Base_Model with no adapter applied.

### Requirement 3: Redis Layout Contract

**User Story:** As a Track A and Track B engineer, I want a fixed Redis key layout and client helpers, so that I can store and retrieve adapters and interactions without owning the Redis implementation.

#### Acceptance Criteria

1. THE Redis_Layer SHALL store adapter bytes under key `adapter:blob:<adapter_id>` or store a disk path under that key when blobs are kept on disk.
2. THE Redis_Layer SHALL store adapter metadata JSON under key `adapter:meta:<adapter_id>`.
3. THE Redis_Layer SHALL maintain a vector index of `unit_label` embeddings under key `adapter:index`.
4. THE Redis_Layer SHALL store each Unit's raw daily interactions under key `interactions:<unit_label>`.
5. THE Redis_Layer SHALL expose a Route_Function `route(query_or_user) -> adapter_id` that returns the top-1 adapter by vector search.

### Requirement 4: Dataset and Training-Pair Schema Contract

**User Story:** As a Track A and Track B engineer, I want a fixed training-pair schema, so that data produced by the pipeline trains adapters and feeds the eval consistently.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL emit each Training_Pair as a JSONL row of shape `{ "prompt": string, "completion": string, "unit_label": string }`.
2. THE Data_Pipeline SHALL emit Held_Out_Set rows in the same shape as a Training_Pair, in a file separate from the train rows.
3. THE Data_Pipeline SHALL ensure that for each Unit the train rows and the Held_Out_Set rows do not overlap.
4. IF a row is missing the `prompt`, `completion`, or `unit_label` field, THEN THE consuming component SHALL reject the row and report the missing field name.

### Requirement 5: Eval Artifact Schema Contract

**User Story:** As a Track C engineer, I want a fixed eval artifact schema, so that the Dashboard can render proof visuals against a stable file before the real eval runs.

#### Acceptance Criteria

1. THE Weave_Eval SHALL emit an `eval_results.json` artifact containing the fields `perplexity`, `confusion_matrix`, `size_bytes`, and `examples`.
2. THE Weave_Eval SHALL populate `perplexity` with the keys `base`, `adapter`, and `context_memory`, each mapping to a number.
3. THE Weave_Eval SHALL populate `confusion_matrix` with a `labels` array containing at least one string and a `matrix` array of numeric rows, where the matrix dimensions equal the count of labels.
4. THE Weave_Eval SHALL populate `size_bytes` with the keys `nktmirror` and `lora`, each mapping to an integer.
5. THE Weave_Eval SHALL populate `examples` with objects each containing `prompt`, `base`, `adapter`, and `reference` string fields.

---

## Track A — Model & Serving (Python)

> Goal: reproduce the NKT-Mirror training loop on the instruct model and serve adapters through a local inference API. Owns the adapter file format (Requirement 1) and the inference API schema (Requirement 2). Consumes nothing from B or C to function. This is the critical path.

### Requirement 6: Reproduce NKT-Mirror on the Instruct Model

**User Story:** As a Track A engineer, I want to reproduce the NKT-Mirror method on the instruct base model, so that the personalization mechanism is verified before anything downstream depends on it.

#### Acceptance Criteria

1. THE Serving_Engine SHALL load the Base_Model as an instruct model identified by the `base_model` metadata field.
2. WHEN an NKT_Mirror_Adapter completes training on a benchmark dataset using the instruct Base_Model, THE Train_Adapter SHALL produce a measured accuracy at least equal to the measured accuracy of the same Base_Model with no adapter on that benchmark, where intermediate accuracy during early training phases before convergence is exempt from this criterion.
3. THE NKT_Mirror_Adapter SHALL apply per-channel activation gating on the frozen Base_Model without updating Base_Model weights.
4. THE NKT_Mirror_Adapter SHALL serialize to an Adapter_File whose `size_bytes` is at most 200,000 bytes.

### Requirement 7: Custom Adapter Serving

**User Story:** As a Track A engineer, I want custom serving that loads the base once and swaps gate tensors per request, so that thousands of tiny adapters can be served without reloading the base model.

#### Acceptance Criteria

1. THE Serving_Engine SHALL load the Base_Model into memory exactly once per process lifetime.
2. WHEN a request specifies an `adapter_id`, THE Serving_Engine SHALL apply that adapter's gate tensors to the resident Base_Model for that request.
3. WHEN a request specifies `adapter_id` as null, THE Serving_Engine SHALL generate using the resident Base_Model with no gate tensors applied.
4. WHEN the same prompt is sent with a given `adapter_id` and with `adapter_id` null, THE Serving_Engine SHALL produce different output text for every such prompt.
5. IF a requested `adapter_id` is not loadable, THEN THE Serving_Engine SHALL return an error response that names the missing `adapter_id`.

### Requirement 8: Inference API Service

**User Story:** As a Track A engineer, I want a FastAPI service exposing generate, score, adapters, and train, so that Track B and Track C can call serving over HTTP.

#### Acceptance Criteria

1. WHEN a `POST /generate` request conforming to Requirement 2 is received, THE Inference_API SHALL return generated text with the `tokens` and `latency_ms` fields populated.
2. WHEN a `POST /score` request conforming to Requirement 2 is received, THE Inference_API SHALL return a non-negative `perplexity` and an `nll` computed over the supplied `target`.
3. WHEN a `GET /adapters` request is received, THE Inference_API SHALL return the list of currently loadable `adapter_id` values.
4. IF a request body fails to match the schema in Requirement 2, THEN THE Inference_API SHALL return a validation error identifying the offending field.

### Requirement 9: Train Adapter Function

**User Story:** As a Track A engineer, I want a `train_adapter` function callable directly and over HTTP, so that the LangGraph batch graph can train adapters on curated datasets.

#### Acceptance Criteria

1. WHEN `train_adapter` is called with a `dataset_path`, a `unit_label`, and a `unit_type`, THE Train_Adapter SHALL produce an Adapter_File and return its `adapter_path`.
2. WHEN Train_Adapter produces an Adapter_File, THE Train_Adapter SHALL write metadata whose `train_rows` equals the count of training rows consumed and whose `unit_label` and `unit_type` equal the supplied arguments.
3. WHEN Train_Adapter is invoked through `POST /train`, THE Inference_API SHALL return the same `adapter_path` that the direct function call would return.
4. IF the supplied `dataset_path` does not resolve to a readable training-pair file, THEN THE Train_Adapter SHALL return an error that names the unreadable path.
5. IF the supplied dataset contains zero training rows, THEN THE Train_Adapter SHALL return an error indicating insufficient training data and SHALL NOT produce an Adapter_File.

### Requirement 10: Track A Standalone Test

**User Story:** As a Track A engineer, I want a standalone test independent of other tracks, so that I can prove the method works with no dependency on B or C.

#### Acceptance Criteria

1. WHEN Track A trains one adapter on a tiny local dataset and serves the Base_Model and that adapter on the same prompt, THE Serving_Engine SHALL produce visibly different output for the adapter than for the Base_Model.
2. THE Track A standalone test SHALL run without any dependency on the Redis_Layer, the LangGraph_Batch_Graph, or the Frontend_App.

---

## Track B — Data, Orchestration & Eval (Python)

> Goal: build the data pipeline, the GPT curation node, the LangGraph nightly-batch graph, and the Weave eval that proves personalization. Owns the dataset/training-pair schema (Requirement 4) and the eval artifact schema (Requirement 5). Consumes Track A's `train_adapter` and inference API, mocked until ready.

### Requirement 11: Data Pipeline and Splits

**User Story:** As a Track B engineer, I want a data pipeline that produces per-unit train and held-out splits, so that adapters train on one slice and the eval measures generalization on unseen text.

#### Acceptance Criteria

1. WHEN source data is loaded for a Unit, THE Data_Pipeline SHALL produce a train file and a Held_Out_Set file of Training_Pairs for that Unit.
2. THE Data_Pipeline SHALL assign every emitted Training_Pair a `unit_label` matching the Unit it was derived from.
3. THE Data_Pipeline SHALL ensure the train rows and Held_Out_Set rows for a Unit do not overlap, consistent with Requirement 4.
4. WHERE a Unit has fewer source rows than a configured minimum, THE Data_Pipeline SHALL exclude that Unit from the demo set and record the excluded `unit_label`, and WHERE a Unit has source rows equal to or greater than the configured minimum, THE Data_Pipeline SHALL include that Unit in the demo set.

### Requirement 12: GPT Curation Node

**User Story:** As a Track B engineer, I want a GPT-based curation node that turns raw interactions into clean training pairs, so that the small adapter trains on high-quality data.

#### Acceptance Criteria

1. WHEN the GPT_Curation_Node receives raw interactions for a Unit, THE GPT_Curation_Node SHALL emit Training_Pairs conforming to Requirement 4.
2. THE GPT_Curation_Node SHALL be the only component in WeaveSelf that calls GPT.
3. WHERE a local curation model is configured instead of GPT, THE GPT_Curation_Node SHALL produce Training_Pairs of the same schema.
4. IF the GPT_Curation_Node cannot produce a valid Training_Pair from a raw interaction, THEN THE GPT_Curation_Node SHALL discard that interaction and record the count of discarded interactions.

### Requirement 13: LangGraph Nightly-Batch Graph

**User Story:** As a Track B engineer, I want a LangGraph graph that runs collect, curate, train, eval, and store as a batch, so that adapters are produced overnight rather than live.

#### Acceptance Criteria

1. THE LangGraph_Batch_Graph SHALL execute the nodes in the order collect, curate, train, eval, store.
2. WHEN the train node runs, THE LangGraph_Batch_Graph SHALL invoke Track A's Train_Adapter with the curated dataset, `unit_label`, and `unit_type`.
3. WHEN the store node runs, THE LangGraph_Batch_Graph SHALL persist each produced Adapter_File and its metadata through the Redis_Client_API.
4. THE LangGraph_Batch_Graph SHALL run as a batch job and SHALL NOT perform adapter training in response to a live chat request.
5. WHILE the LangGraph_Batch_Graph is executing a batch run, THE LangGraph_Batch_Graph SHALL block live chat requests from triggering graph execution.
6. IF any node fails for a Unit, THEN THE LangGraph_Batch_Graph SHALL record the failing node and `unit_label` and continue processing the remaining Units.
7. IF failure recording itself fails or a critical error prevents continuation, THEN THE LangGraph_Batch_Graph SHALL halt processing.

### Requirement 14: Weave Eval — Perplexity and Baseline

**User Story:** As a Track B engineer, I want held-out perplexity and a context-memory baseline comparison, so that personalization is proven objectively at zero extra context cost.

#### Acceptance Criteria

1. WHEN the Weave_Eval scores a Unit's Held_Out_Set under that Unit's Adapter and under the Base_Model, THE Weave_Eval SHALL record both perplexity values.
2. THE Weave_Eval SHALL report personalization as passing for a Unit WHEN the adapter held-out perplexity is lower than the Base_Model held-out perplexity for that Unit.
3. WHEN the Weave_Eval runs the Context_Memory_Baseline, THE Weave_Eval SHALL score the same Held_Out_Set with the Unit's examples injected into the prompt and record that perplexity.
4. THE Weave_Eval SHALL report the competitive comparison as passing WHEN the adapter held-out perplexity is less than or equal to the Context_Memory_Baseline perplexity.
5. THE Weave_Eval SHALL log perplexity results to Weave/W&B.

### Requirement 15: Weave Eval — Confusion Matrix and Size Chart

**User Story:** As a Track B engineer, I want a cross-unit confusion matrix and a size chart, so that the headline visual proves each adapter learned its own unit and demonstrates the size advantage.

#### Acceptance Criteria

1. WHEN the Weave_Eval scores each Unit's Held_Out_Set under every trained Adapter, THE Weave_Eval SHALL select the lowest-perplexity Adapter as the predicted Unit for that Held_Out_Set.
2. THE Weave_Eval SHALL build a Confusion_Matrix whose rows are the true Unit and whose columns are the predicted Unit, consistent with Requirement 5.
3. THE Weave_Eval SHALL record `size_bytes` comparing the NKT_Mirror_Adapter size against a LoRA adapter size.
4. THE Weave_Eval SHALL emit all results to an `eval_results.json` artifact conforming to Requirement 5.
5. WHERE the optional fact-capacity test is enabled, THE Weave_Eval SHALL plant a configured number of preferences per Unit and record held-out recall as that number grows.

### Requirement 16: Track B Standalone Test

**User Story:** As a Track B engineer, I want to run the full graph against mock adapters, so that the confusion matrix can be produced without Track A serving.

#### Acceptance Criteria

1. WHEN the LangGraph_Batch_Graph runs against a Mock_Dependency for Track A's Train_Adapter and inference API, THE Weave_Eval SHALL emit an `eval_results.json` containing a Confusion_Matrix.
2. THE Track B standalone test SHALL run without any dependency on the Frontend_App.

---

## Track C — Frontend, Redis & Integration (TS/Node + light Python)

> Goal: build the CopilotKit app, the AG-UI ↔ LangGraph wiring, the Redis layer with a clean client API, and the end-to-end demo glue. Owns the Redis layout (Requirement 3). Consumes Track A's inference API and Track B's eval artifacts, mocked with fixtures until ready.

### Requirement 17: CopilotKit Chat and Dashboard

**User Story:** As a user, I want a chat interface and a proof dashboard, so that I can talk to a personalized assistant and see the evidence that personalization works.

#### Acceptance Criteria

1. WHEN a user selects a Unit and sends a chat message, THE Frontend_App SHALL call the Inference_API and display the generated response.
2. THE Dashboard SHALL render an adapter library view listing each Adapter by `adapter_id`, `unit_label`, and `size_bytes`, and SHALL hide any Adapter whose `size_bytes` is zero from the library view.
3. THE Dashboard SHALL render the Confusion_Matrix from `eval_results.json` as a heatmap.
4. THE Dashboard SHALL render base-versus-adapter example pairs alongside the reference text from the `examples` field of `eval_results.json`.
5. THE Dashboard SHALL render the size chart from the `size_bytes` field of `eval_results.json`.

### Requirement 18: AG-UI to LangGraph Wiring

**User Story:** As a Track C engineer, I want the Node/TS CopilotKit runtime bridged to the Python LangGraph agent over AG-UI, so that the two-runtime stack communicates.

#### Acceptance Criteria

1. THE AG_UI_Bridge SHALL connect the Node/TS CopilotKit runtime to the Python LangGraph agent over AG-UI server-sent events.
2. WHEN the Frontend_App sends a request through the AG_UI_Bridge, THE Python LangGraph agent SHALL receive the request and stream its response back to the Frontend_App.
3. IF the AG_UI_Bridge connection to the Python LangGraph agent is unavailable, THEN THE Frontend_App SHALL display a connection error to the user.
4. WHEN the AG_UI_Bridge connection to the Python LangGraph agent is restored, THE Frontend_App SHALL clear the displayed connection error.

### Requirement 19: Redis Layer and Client API

**User Story:** As a Track C engineer, I want a Redis layer with a clean client API, so that Track A and Track B can store, fetch, and route adapters through one interface.

#### Acceptance Criteria

1. WHEN an Adapter_File and its metadata are stored through the Redis_Client_API, THE Redis_Layer SHALL persist them under the keys defined in Requirement 3.
2. WHEN an `adapter_id` is fetched through the Redis_Client_API, THE Redis_Layer SHALL return the stored metadata for that `adapter_id`, and SHALL return the stored blob bytes or path when blob retrieval is requested, supporting retrieval of metadata independently of the blob.
3. WHEN the Route_Function is called with a query or user identifier, THE Redis_Layer SHALL return the top-1 `adapter_id` from the vector index.
4. WHEN a 100 KB blob is stored and then fetched through the Redis_Client_API, THE Redis_Layer SHALL return bytes identical to the stored bytes.
5. WHEN raw interactions are appended for a Unit, THE Redis_Layer SHALL store them under key `interactions:<unit_label>`.

### Requirement 20: Track C Standalone Test

**User Story:** As a Track C engineer, I want the UI and Redis to work against fixtures, so that the product story is demoable without Track A or Track B.

#### Acceptance Criteria

1. WHEN the Frontend_App loads a mock `eval_results.json` fixture, THE Dashboard SHALL render the heatmap, the example pairs, the size chart, and the adapter library.
2. WHEN a dummy 100 KB blob is stored, fetched, and routed through the Redis_Client_API, THE Redis_Layer SHALL round-trip the blob, return bytes identical to the stored blob, and return a routed `adapter_id`.
3. THE Track C standalone test SHALL run without any dependency on Track A serving or the live LangGraph_Batch_Graph.

---

## Cross-Track Requirements

### Requirement 21: Integration Milestone — Real Adapters End to End

**User Story:** As the team, I want all three tracks wired on real adapters, so that the system demonstrates the full collect-to-chat loop.

#### Acceptance Criteria

1. WHEN the Integration_Milestone is reached, THE WeaveSelf SHALL replace each Mock_Dependency with the real Track A inference API, real Track B adapters and eval artifacts, and the real Track C Redis_Layer and Frontend_App.
2. WHEN the LangGraph_Batch_Graph completes for the demo Units, THE WeaveSelf SHALL serve each produced Adapter through the Inference_API by `adapter_id` retrieved through the Redis_Client_API.
3. THE WeaveSelf SHALL pre-bake Adapter_Files for the demo day indices so that the demo is time-compressed.
4. WHEN a user selects a Unit in the Frontend_App at the Integration_Milestone, THE WeaveSelf SHALL route to the correct Adapter, generate a response, and display the corresponding `eval_results.json` proof visuals.

### Requirement 22: Critical Path Priority

**User Story:** As the team, I want the critical path identified and built first, so that the riskiest work is de-risked before dependent work begins.

#### Acceptance Criteria

1. THE WeaveSelf SHALL treat Track A custom serving (Requirement 7) and method reproduction (Requirement 6) as the critical path built before dependent integration.
2. WHILE Track A serving is not yet verified, THE Track B and Track C work SHALL proceed against a Mock_Dependency for the inference API.
3. IF Track A serving cannot be verified, THEN THE WeaveSelf SHALL record that downstream integration is blocked and fall back to the per-track standalone demos in Requirement 23.

### Requirement 23: Per-Track Standalone Fallback Demos

**User Story:** As the team, I want each track to have a standalone fallback demo, so that the project still demonstrates value if integration slips.

#### Acceptance Criteria

1. IF integration slips, THEN Track A SHALL demo a command-line comparison of the Base_Model versus one Adapter on the same prompt showing visibly different, more on-style output.
2. IF integration slips, THEN Track B SHALL demo the confusion-matrix heatmap and the base-versus-adapter perplexity chart in Weave from pre-trained adapters.
3. IF integration slips, THEN Track C SHALL demo the CopilotKit UI with a mock `eval_results.json` fixture and the Redis adapter library view.
4. THE WeaveSelf SHALL treat the Track B Confusion_Matrix as the single highest-priority artifact such that it is demoable independently of every other track.
