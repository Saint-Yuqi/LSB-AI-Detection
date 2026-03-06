# Module: src/pipelines

## Responsibilities
- Contain the core logic for multi-phase data preparation pipelines.
- Provide testable, importable functions that scripts/ entry points delegate to.

## Sub-packages

### `unified_dataset/`
4-Phase pipeline for preparing the LSB training dataset:
- `keys.py` — `BaseKey`, `VariantKey` dataclasses
- `paths.py` — `PathResolver` for config-driven I/O paths
- `config.py` — `load_config`, `generate_base_keys`
- `preprocessor_factory.py` — `create_preprocessor` (single source of truth)
- `fs_utils.py` — `sha1_file` and shared filesystem helpers
- `render.py` — Phase 1: FITS -> RGB renders
- `gt.py` — Phase 2: SB mask -> streams instance map
- `inference.py` — Phase 3 dispatcher (routes to SAM2 or SAM3)
- `inference_sam2.py` — SAM2 AutoMask -> filter -> merge -> cache
- `inference_sam3.py` — SAM3 text-prompt -> filter -> evaluate
- `artifacts.py` — Prediction JSON, satellites cache, instance merging
- `export.py` — Phase 4: SAM2 symlinks + SAM3 COCO annotations
- `split.py` — Galaxy-level COCO train/val split (hash-stable)
- `noise_aug.py` — Noise augmentation (train-only): symlinks noisy renders, copies GT annotations
- `compose.py` — Compose training dataset from multiple annotation sources (symlink or merge)

### Layered Dataset Pipeline (post-export)

Canonical ordering: `export.py` → `split.py` → `noise_aug.py` → `compose.py`

```
annotations.json (canonical clean)
  ├─ split → annotations_train.json + annotations_val.json
  │           + split_manifest.json
  ├─ noise_aug → annotations_train_noise_augmented.json
  │               + noise_aug_manifest.json
  └─ compose → annotations_train_active.json (training target)
                + compose_manifest.json
```

Shared dataset root contract: all annotation files use `file_name` paths
relative to the same dataset root (`sam3_prepared/`). The compose layer
only merges JSON — image symlinks are created by upstream producers.

## Non-goals
- No CLI argument parsing (that lives in `scripts/`).
- No direct I/O to stdout (use logging).
