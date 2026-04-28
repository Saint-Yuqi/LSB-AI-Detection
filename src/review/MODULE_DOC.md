# Review Module (`src/review/`)

AI Verifier Protocol V1.2 implementation.

## Responsibilities

Three-layer architecture for training AI review verifiers:

1. **Business JSONL** (`verifier_examples_{family}.jsonl`) — one record per review task, carrying asset references, labels, and revision hashes.
2. **Render Asset Layer** — shared bare context images, per-candidate crops, and EV full-image views; deduped and cached by content key.
3. **Model-Input Layer** (`verifier_chat_{family}.jsonl`) — two-turn user/assistant chat records ready for Llama instruction tuning.

Three isolated verifier families:

| Family | Scope | Label Space |
|---|---|---|
| `satellite_mv` | Candidate-level mask verification | accept, minor_fix, reject, route_to_ev |
| `satellite_ev` | Image-level exhaustivity (GT-driven) | confirm_complete, add_missing, confirm_empty |
| `stream_ev` | Image-level stream exhaustivity (GT-driven) | confirm_complete, add_missing_fragment, delete_fragment, redraw, confirm_empty, uncertain |

### GT-Driven Synthetic Variants (V1.2)

`satellite_ev` and `stream_ev` now use **deterministic GT-driven synthetic variants**
instead of pred-vs-GT exhaustivity matching.  For each image, a small fixed set of
silver labels is generated:

- **`gt_complete`**: All instances visible → `confirm_complete`
- **`gt_empty`**: Zero instances (empty field) → `confirm_empty`
- **`drop_topK`** (`satellite_ev`): Hide the largest up to 3 satellites in one image → `add_missing`
- **`drop_top1`** (`stream_ev`): Hide only the largest fragment in one image → `add_missing_fragment`

Each variant carries `visible_instance_ids` and `hidden_instance_ids` in its signals.
Fragment hints are rendered only from visible instances.  Each variant gets a unique
review image path (via `state_key`) and unique revision hash (via `synthetic_variant_id`).

#### Prompt compatibility note

`stream_ev_v1` prompt is **intentionally unchanged** in this round.  Schema compatibility
is prioritized; the prompt will be tightened in a future round once the data distribution
is validated.

## Module Map

| File | Purpose |
|---|---|
| `schemas.py` | Frozen dataclasses, enums, label-space enforcement, revision-hash computation |
| `key_adapter.py` | `BaseKey ↔ sample_id / halo_id` mapping layer |
| `render_spec.py` | Versioned visual specification registry (YAML-backed) |
| `prompt_registry.py` | Fixed prompt template registry (YAML-backed) |
| `review_render.py` | Crop / bare-context / stamped-context / EV rendering |
| `asset_manager.py` | Shared context dedup, crop cache, asset manifests; `state_key` param for multi-variant EV |
| `candidate_matcher.py` | Pred-centric matching (per-prediction IoU, area ratio, boundary F1, ambiguity) |
| `silver_labeler.py` | Silver label generation: MV pred-centric + EV/Stream GT-driven synthetic |
| `example_builder.py` | Pipeline artifacts → business JSONL (dual-source: authoritative + pre-merge rejected; multi-variant EV) |
| `etl.py` | Business JSONL → chat JSONL deterministic transform |
| `correction.py` | Redraw closed-loop, revision-hash validation, per-family write-back; synthetic variant guard |
| `holdout.py` | Group-wise holdout splitting by `halo_id` |
| `round_manager.py` | Round artifact assembly with content/provenance separation |

## Key Invariants

- **Dedup**: N candidates from the same sample share 1 bare context file.
- **Determinism**: Content artifacts (crops, chat JSONL) are byte-deterministic given the same inputs.
- **Content / Provenance separation**: Manifest files carry timestamps; content files never do.
- **Family isolation**: No cross-family batching at any training stage.
- **Revision-hash integrity**: Any change to a task's canonical payload invalidates the hash.
- **Backward-compat hash**: Omitting `synthetic_variant_id` (or passing `None`) produces the same hash as legacy records.
- **Correction eligibility**: Only `candidate_source == "authoritative"` MV examples can be correction targets.
- **Synthetic write-back guard**: Only `gt_complete`, `gt_empty`, or `None` variants may enter the correction chain.

## I/O Contracts

### Input
- Pipeline renders: `{render_dir}/{variant}/{BaseKey}/0000.png`
- Authoritative GT: `{gt_dir}/{BaseKey}/instance_map_uint8.png`, `instances.json`
- Pre-merge artifacts: `satellites_cache.npz` (SAM2), `sam3_predictions_raw.json` (SAM3)

### Output
- `silver_labels_{family}.jsonl` — intermediate silver labels (multi-variant for EV)
- `verifier_examples_{family}.jsonl` — business JSONL
- `verifier_chat_{family}.jsonl` — model-input chat JSONL
- `renders/review_crops/` — candidate crop PNGs
- `renders/review_context/` — shared bare context PNGs
- `renders/review_stamped/` — per-candidate stamped context PNGs (ETL output)
- `renders/review_ev/` — EV full-image PNGs (multi-variant suffix in filename)
- `asset_manifest.json`, `etl_manifest.json`, `round_manifest.json` — provenance metadata

## Failure Modes

- **Stale revision hash**: Correction import rejects the record with `ValueError`.
- **Reject candidate correction**: Attempting to correct a `pre_merge_rejected` candidate raises `ValueError`.
- **Synthetic variant rejection**: Attempting to correct a `drop_*` variant raises `ValueError`.
- **Label space violation**: `VerifierExample.__post_init__` raises `ValueError`.
- **Missing render**: CLI scripts print warnings and skip the sample.

## Tidal_v1 (3-class) Additions

### `authoritative_gt.rebuild_id_map(instances)` (extended)
- Now tolerates both legacy (`type` field with `"streams"`/`"satellites"`) and new (`type_label` field with `"tidal_features"`/`"satellites"`/`"inner_galaxy"`) row shapes. Falls back to `inst.get("type") or inst.get("type_label", "unknown")`.
- The `streams` and `satellites` buckets are still emitted unconditionally for backward compatibility with consumers that index by hardcoded class name.

## Status

Experimental, partially prepared — only a subset of inputs has been generated so far and whether a full verifier loop runs end-to-end is still being decided. The module ships the schemas, render/asset pipeline, silver labeller, example/ETL builders, and correction write-back paths; consumers should treat it as exploratory and not depend on it from the main training/eval flows.
