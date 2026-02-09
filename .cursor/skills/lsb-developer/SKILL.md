---
name: lsb-developer
description: Expert guide for developing the LSB-AI-Detection project. Use this skill when modifying code, adding features, or fixing bugs to ensure architectural integrity and physical correctness.
---

# LSB Developer Guide

You are the lead engineer for the LSB-AI-Detection project. Your goal is to maintain a modular architecture and ensure astrophysical accuracy.

## 🧠 Decision Tree: Where does my code go?

Before writing a single line, locate your task in this logic tree:

1.  **Are you defining NEW logic/math/classes?**
    * Yes, Image Processing/Physics -> Go to `src/data/preprocessing.py`
    * Yes, File Loading/Saving -> Go to `src/data/io.py`
    * Yes, Dataset/Matching Logic -> Go to `src/data/builder.py`
    * Yes, Evaluation Metrics -> Go to `src/evaluation/metrics.py`

2.  **Are you EXECUTING existing logic?**
    * Yes, running a dataset build -> Go to `scripts/build_dataset.py`
    * Yes, evaluating a model -> Go to `scripts/eval_model.py`
    * *Constraint*: Scripts should be thin wrappers (<100 lines). Logic belongs in `src/`.

3.  **Are you modifying data structures?**
    * Yes -> Check `src/data/io.py` first to see how `SatelliteInstance` is defined.

---

## ⚡️ Physics Safety Protocols (Critical)

**Trigger**: If your task involves pixel values, intensity, or magnitude.

* **Protocol 1: Mag vs Flux**
    * Neural Networks require **Linear Flux**.
    * Raw FITS files are **Magnitude**.
    * **Action**: ALWAYS apply `Flux = 10 ** ((ZP - Mag) / 2.5)` (ZP=22.5) immediately after loading.

* **Protocol 2: Mask Integrity**
    * **Streams**: NEVER binary threshold (`mask > 0`). Use `int` IDs to distinguish instances.
    * **Satellites**: Preserve PKL instance IDs.

---

## 📝 Documentation Workflow

**Trigger**: You have finished writing/modifying code.

**Step 1: Update `CHANGELOG.md`**
* Follow **Keep a Changelog** format.
* Record changes under `### Added`, `### Changed`, or `### Fixed`.

**Step 2: Update `project_context.md`**
* If you added a file -> Update Directory Structure.
* If you changed a Class signature -> Update Code Skeleton.
* *Note*: Do not duplicate file contents here; just keep the map accurate.

---

## 🚫 Anti-Patterns (Do Not Do)

* **Script Sprawl**: Creating `test.py` or `temp_script.py`. (Use Jupyter for scratchpad).
* **Physics Violation**: Subtracting background from Magnitude data.
* **Context Duplication**: Copying full file contents into chat unnecessarily.
