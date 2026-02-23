# Module: utils

## Responsibilities
- Provide standardized COCO format conversion utilities (RLE encoding, bounding box extraction).
- Supply a centralized, thread-safe, rotating file logging configuration.

## Non-goals
- **No Domain Logic:** Does not run astrophysics physical equations or instance filtering algorithms.
- **No Orchestration:** Does not write JSON files to disk directly.

## Inputs / Outputs

### `mask_to_rle(binary_mask)`
- **Input:** `binary_mask: np.ndarray` — shape `(H, W)`, dtype `np.uint8`, C-contiguous.
- **Output:**
  - `size: List[int]` — `[H, W]`
  - `counts: str` — RLE-encoded run-length string

### `get_bbox_from_mask(binary_mask)`
- **Input:** `binary_mask: np.ndarray` — shape `(H, W)`, dtype `np.uint8`.
- **Output:** `List[float]` of length 4 — `[x_min, y_min, width, height]`

### `create_categories(thresholds)`
- **Input:** `thresholds: List[float]`
- **Output:** `Tuple` of:
  - `List` of category dicts, each containing:
    - `id: int`
    - `name: str`
    - `supercategory: str`
  - `Dict` mapping `Tuple[str, float]` → `int` (lookup index)

### `process_mask_to_annotations(mask_data, image_id, category_id, ann_id, min_area)`
- **Input:**
  - `mask_data: np.ndarray` — shape `(H, W)`, dtype `int`
  - `image_id: int`
  - `category_id: int`
  - `ann_id: int`
  - `min_area: int`
- **Output:** `Tuple` of:
  - `List` of annotation dicts (one per instance with `area >= min_area`), each containing:
    - `id: int`
    - `image_id: int`
    - `category_id: int`
    - `segmentation`:
      - `size: List[int]` — `[H, W]`
      - `counts: str`
    - `bbox: List[float]` — `[x_min, y_min, width, height]`
    - `area: int`
    - `iscrowd: int` — always `0`
  - `int` — next available annotation ID after this call

### `setup_logger(name, log_dir, level, max_bytes, backup_count)`
- **Input:** `name: str`, `log_dir: Optional[Path]`, `level: int`, `max_bytes: int`, `backup_count: int`
- **Output:** `logging.Logger`

## Invariants
- **COCO Encoding Standard:** Serialized annotation objects comply with the COCO JSON schema so downstream tools can validate them without modification.
- **Input Array Format:** `binary_mask` passed to `mask_to_rle` must be dtype `np.uint8` and C-contiguous; any other layout is rejected before encoding.

## Produced Artifacts
- Disk logging traces under bounded rotating capacities.
- In-memory annotation lists abiding by the COCO JSON schema.

## Failure Modes
- `TypeError`: Raised by `mask_to_rle` if `binary_mask` dtype is not `np.uint8`.
- `ValueError`: Raised by `mask_to_rle` if `binary_mask` is not C-contiguous (non-standard memory layout).
- `ValueError`: Raised by `get_bbox_from_mask` if `binary_mask` contains no foreground pixels (empty mask returns no valid bounding box).
 
