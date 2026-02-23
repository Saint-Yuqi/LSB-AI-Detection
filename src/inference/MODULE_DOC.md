# Module: inference

## Responsibilities
- Provide an interface for SAM2 Automatic Mask Generation inference.
- Handle model initialization with specified configuration and weights.
- Measure execution time on the GPU for sweep evaluations.

## Non-goals
- **No Evaluation:** Does not compare masks against ground truth or compute precision/recall.
- **No Post-processing:** Returns raw generated masks. It does not filter out unstable masks or apply shape thresholds.

## Type Aliases

### `MaskDict` (TypedDict, STRICTLY CLOSED)
- `segmentation: np.ndarray` (boolean, HxW)
- `area: int`
- `bbox: List[int]` ([x0, y0, w, h])
- `predicted_iou: float`
- `point_coords: List[List[float]]`
- `stability_score: float`
- `crop_box: List[int]`

### `AutoMaskConfig` (TypedDict, all keys optional)
- `points_per_side: int`
- `points_per_batch: int`
- `pred_iou_thresh: float`
- `stability_score_thresh: float`
- `box_nms_thresh: float`
- `crop_n_layers: int`
- `crop_nms_thresh: float`
- `min_mask_region_area: int`

## Inputs / Outputs

### `AutoMaskRunner.__init__(checkpoint, model_cfg, device, use_bf16)`
- **Input:**
  - `checkpoint: Union[str, Path]` — path to model weight file
  - `model_cfg: str` — model architecture identifier string
  - `device: str` — e.g. `"cuda"`, `"cpu"`
  - `use_bf16: bool`

### `AutoMaskRunner.run(image, config, warmup)`
- **Input:**
  - `image: np.ndarray` — shape `(H, W, 3)`, dtype `np.uint8`
  - `config: Optional[AutoMaskConfig]`
  - `warmup: bool`
- **Output:** `Tuple[List[MaskDict], float]`
  - Element 0: list of mask dicts (one per detected region)
  - Element 1: GPU runtime in milliseconds

### `AutoMaskRunner.warmup(image, config, n)`
- **Input:**
  - `image: np.ndarray` — shape `(H, W, 3)`, dtype `np.uint8`
  - `config: Optional[AutoMaskConfig]`
  - `n: int` — number of warmup passes
- **Output:** `None`

## Invariants
- **Precision Contract:** The model executes in the numeric precision set by `use_bf16` at construction; callers cannot override precision per-call.
- **Benchmarking:** The returned runtime scalar covers the full GPU execution span from image ingestion to final mask list retrieval.
- **Zero-Detection Boundary:** When the model produces zero candidate regions for the given image (e.g., uniform/blank input), `run` returns `([], float)`.

## Produced Artifacts
- In-memory `List[MaskDict]` reflecting exactly the SAM2 framework's native output keys.

## Failure Modes
- `FileNotFoundError`: Raised by `__init__` when `checkpoint` path does not exist on the filesystem.
- `RuntimeError`: Raised by `run` when GPU memory is exhausted (OOM) due to image resolution or grid density.
