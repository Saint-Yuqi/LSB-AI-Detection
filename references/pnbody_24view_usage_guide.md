# PNbody 24-View Pipeline 使用说明

## 整体工作流概览

```
Step 0: 生成 24 视角 clean FITS              (generate_pnbody_fits.py)
Step 1: 渲染 + 跳过 GT + 伪标签推理 + 导出   (prepare_unified_dataset.py)
Step 2: Galaxy-level 分割                     (split_annotations.py)
Step 3: 合并到训练集                           (build_training_dataset.py)
```

---

## Step 0: 从 HDF5 生成 24 视角 clean FITS

每个 halo 通过 `mockimgs_sb_compute_images` 生成 24 个 LOS 视角的 FITS 文件。

```bash
# 全量运行（184 halo × 24 LOS = 4416 FITS）
python scripts/generate_pnbody_fits.py \
  --config configs/pnbody/firebox_pnbody_24los.yaml

# Smoke test（仅 2 个 halo）
python scripts/generate_pnbody_fits.py \
  --config configs/pnbody/firebox_pnbody_24los.yaml \
  --galaxies 11,13

# Dry run（只打印命令，不执行）
python scripts/generate_pnbody_fits.py \
  --config configs/pnbody/firebox_pnbody_24los.yaml \
  --galaxies 11 \
  --dry-run
```

**输出**:
- `data/01_raw/LSB_and_Satellites/FIREbox_PNbody/SB_maps/magnitudes-Fbox-{gid}-los{00..23}-VIS2.fits.gz`
- `data/01_raw/LSB_and_Satellites/FIREbox_PNbody/metadata/views.csv` — view_id 到 LOS 向量的映射
- `data/01_raw/LSB_and_Satellites/FIREbox_PNbody/metadata/generation_manifest.json` — 完整参数快照

---

## Step 1: 统一数据准备管线（PNbody 配置）

使用专门的 PNbody 配置运行 4-phase 管线。关键区别：
- **GT phase 自动跳过**（`gt_phase.enabled: false`）
- **SAM3 以 `pseudo_label` 模式运行**（不需要真值 mask）
- **导出到独立注释文件** `annotations_pnbody_pseudo.json`

```bash
# 全量运行 4 个阶段（render → [gt=skip] → inference → export）
python scripts/prepare_unified_dataset.py \
  --config configs/unified_data_prep_pnbody.yaml

# 只跑推理阶段（已有 render 产物的情况下）
python scripts/prepare_unified_dataset.py \
  --config configs/unified_data_prep_pnbody.yaml \
  --phase inference

# Smoke test：仅处理 2 个 galaxy
python scripts/prepare_unified_dataset.py \
  --config configs/unified_data_prep_pnbody.yaml \
  --galaxies 11,13

# 强制重跑全部变体
python scripts/prepare_unified_dataset.py \
  --config configs/unified_data_prep_pnbody.yaml \
  --force
```

**pseudo_label 模式输出**（每个 `gt_canonical/current/{gid}_{view}/`）:

| 文件 | 说明 |
|------|------|
| `sam3_predictions_raw.json` | SAM3 原始预测（RLE 编码） |
| `sam3_predictions_post.json` | 后过滤预测 |
| `instance_map_uint8.png` | 伪标签实例图（uint8，ID 1..N） |
| `instances.json` | 实例元数据列表 `[{"id":1,"type":"streams"}, ...]` |
| `sam3_pseudo_label_overlay.png` | **无 GT 的 QA overlay**（仅 pred 填充+轮廓+分数） |
| `manifest.json` | 运行元数据（`run_mode: "pseudo_label"`, `gt_source: "none"`） |

**导出输出**:
- `data/02_processed/sam3_prepared/annotations_pnbody_pseudo.json` — PNbody 专用 COCO 注释

> **注意**: 旧 DR1 数据管线完全不受影响，继续使用 `configs/unified_data_prep.yaml`。

---

## Step 2: Galaxy-level Train/Val 分割

```bash
# 标准分割（DR1 gold 数据）
python scripts/split_annotations.py \
  --annotations data/02_processed/sam3_prepared/annotations.json

# PNbody 伪标签分割（使用 --output-prefix 和 --reuse-manifest）
python scripts/split_annotations.py \
  --annotations data/02_processed/sam3_prepared/annotations_pnbody_pseudo.json \
  --reuse-manifest data/02_processed/sam3_prepared/split_manifest.json \
  --output-prefix annotations_pnbody_pseudo
```

**`--output-prefix` 新参数**（默认 `"annotations"`）:

| 参数值 | 输出文件 |
|--------|---------|
| `annotations`（默认） | `annotations_train.json`, `annotations_val.json` |
| `annotations_pnbody_pseudo` | `annotations_pnbody_pseudo_train.json`, `annotations_pnbody_pseudo_val.json` |

**`--reuse-manifest`**: 复用已有 `split_manifest.json`，保证相同 galaxy 的 train/val 分配一致。PNbody 新增 galaxy 会按相同 seed 扩展分配。

---

## Step 3: 合并到训练集

```bash
# Gold + PNbody pseudo → annotations_train_active.json
python scripts/build_training_dataset.py \
  --include gold:data/02_processed/sam3_prepared/annotations_train_noise_augmented.json \
  --include pnbody_pseudo:data/02_processed/sam3_prepared/annotations_pnbody_pseudo_train.json \
  --output data/02_processed/sam3_prepared/annotations_train_active.json \
  --allow-duplicate-filenames \
  --force
```

合并后每个 image 会被标记 `"dataset_source": "gold"` 或 `"dataset_source": "pnbody_pseudo"`。

---

## 配置变更说明

### `data_selection.views` 替代 `data_selection.orientations`

```yaml
# 新写法（推荐）
data_selection:
  views: ["eo", "fo"]

# 旧写法（仍兼容，会产生 deprecation warning）
data_selection:
  orientations: ["eo", "fo"]
```

### 路径模板双格式支持

```yaml
# 新 PNbody 配置使用 {view_id}
image_pattern: "magnitudes-Fbox-{galaxy_id}-{view_id}-VIS2.fits.gz"

# 旧 DR1 配置使用 {orientation}（仍然有效，无需修改）
image_pattern: "magnitudes-Fbox-{galaxy_id}-{orientation}-VIS2.fits.gz"
```

### GT 阶段显式禁用

```yaml
# 在 PNbody 配置中
gt_phase:
  enabled: false    # --phase all 会自动跳过 GT 阶段
```

不设置此项时默认 `enabled: true`（DR1 行为不变）。

### 导出文件名可配置

```yaml
export_phase:
  annotations_filename: "annotations_pnbody_pseudo.json"  # 默认 "annotations.json"
```

---

## COCO 注释 Schema 变更

每个 `images[]` entry 新增 `view_id` 字段：

```json
{
  "id": 1,
  "file_name": "images/00011_los00_linear_magnitude.png",
  "galaxy_id": 11,
  "view_id": "los00",
  "orientation": "los00",
  "variant": "linear_magnitude",
  "base_key": "00011_los00"
}
```

- `view_id` — 新主字段（`eo`, `fo`, `los00`..`los23`）
- `orientation` — 兼容镜像字段，值与 `view_id` 相同

---

## API 变更（Python 代码）

### `BaseKey`

```python
# 旧
key = BaseKey(galaxy_id=11, orientation="eo")

# 新
key = BaseKey(galaxy_id=11, view_id="eo")
key = BaseKey(galaxy_id=11, view_id="los00")
str(key)  # "00011_eo" 或 "00011_los00"
```

### 新增函数

| 模块 | 函数 | 说明 |
|------|------|------|
| `artifacts.py` | `rasterize_pseudo_gt(masks, H, W)` | 将 post-filtered masks 栅格化为 instance_map + instances_list |
| `artifacts.py` | `save_pseudo_gt(gt_dir, masks, H, W)` | 写入 `instance_map_uint8.png` + `instances.json` |
| `overlay.py` | `save_pseudo_label_overlay(path, image, predictions)` | 无 GT 的 QA overlay（仅预测填充+轮廓+分数） |
| `inference_sam3.py` | `_pseudo_label_complete(gt_dir)` | 检查 6 个伪标签产物是否齐全 |

---

## 测试

```bash
# 运行全部 91 个测试（75 原有 + 16 smoke）
conda run -n sam3 python -m pytest tests/ -v

# 仅跑 PNbody smoke tests
conda run -n sam3 python -m pytest tests/test_smoke_pnbody.py -v

# 仅跑原有单元测试
conda run -n sam3 python -m pytest tests/test_dataset_keys.py tests/test_artifacts.py \
  tests/test_cli_compat.py tests/test_compose.py tests/test_noise_aug.py \
  tests/test_galaxy_split.py tests/test_eval_type_aware.py -v
```

---

## 完整端到端流程示例（Smoke Test）

```bash
# 0. 生成 FITS（dry-run 验证）
python scripts/generate_pnbody_fits.py \
  --config configs/pnbody/firebox_pnbody_24los.yaml \
  --galaxies 11 --dry-run

# 1. 准备数据（render + pseudo-label + export）
python scripts/prepare_unified_dataset.py \
  --config configs/unified_data_prep_pnbody.yaml \
  --galaxies 11

# 2. 分割伪标签（复用已有 manifest）
python scripts/split_annotations.py \
  --annotations data/02_processed/sam3_prepared/annotations_pnbody_pseudo.json \
  --reuse-manifest data/02_processed/sam3_prepared/split_manifest.json \
  --output-prefix annotations_pnbody_pseudo

# 3. 合并训练集
python scripts/build_training_dataset.py \
  --include gold:data/02_processed/sam3_prepared/annotations_train_noise_augmented.json \
  --include pnbody_pseudo:data/02_processed/sam3_prepared/annotations_pnbody_pseudo_train.json \
  --output data/02_processed/sam3_prepared/annotations_train_active.json \
  --allow-duplicate-filenames --force
```
