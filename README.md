# Whole-Tumor Segmentation from Multimodal MRI

A PyTorch pipeline for binary whole-tumor segmentation using the Medical Segmentation Decathlon (MSD) Task01_BrainTumour dataset. Developed for ENG2440 Medical Imaging & AI in Healthcare, the project covers volumetric preprocessing, patch-based training, full-volume reconstruction, and case-level evaluation.

The main experiment compares uniform and tumor-aware patch sampling under the same training conditions. Additional experiments evaluate missing MRI modalities and sliding-window overlap.

## Project structure

```text
├── ENG2440_A2_BrainTumour.ipynb   # Pipeline, experiments, figures, and analysis
├── requirements.txt             # Pinned Python dependencies
├── sanity_checks.py             # Synthetic software checks
├── VERIFICATION.txt             # Verification scope and results
├── AI_USE_DECLARATION.md         # AI assistance disclosure
├── .gitignore                   # Exclusions for data and generated artifacts
└── README.md
```

## Pipeline

1. Validate the supplied patient-level split and inspect NIfTI geometry.
2. Resample all MRI channels and labels onto a common grid for each case.
3. Normalize MRI intensities and extract paired image/mask patches.
4. Train a lightweight 3D U-Net with uniform sampling, then with tumor-aware sampling.
5. Select the model checkpoint and postprocessing configuration using validation results.
6. Reconstruct full-volume predictions and evaluate the held-out test cases.
7. Generate axial, coronal, and sagittal error overlays.
8. Evaluate missing-modality robustness and sliding-window seams without retraining.

## Dataset

The input consists of four co-registered MRI channels: FLAIR, T1, contrast-enhanced T1 (T1ce), and T2. Channel definitions are read from `dataset.json` and reordered explicitly. Every non-zero reference label is mapped to whole tumor; the original multiclass label files remain unchanged.

Expected directory structure:

```text
Task01_BrainTumour/
├── dataset.json
├── assignment2_split.csv        # Also accepts: assignment2 split.csv
├── imagesTr/
│   ├── BRATS_001.nii.gz
│   └── ...
└── labelsTr/
    ├── BRATS_001.nii.gz
    └── ...
```

Case filenames above are illustrative. The dataset and supplied split CSV are external dependencies and are not included in the repository.

The split loader preserves the supplied case membership and row order. It expects `case_id` and `split` columns, with the following supported alternatives:

| Field | Supported columns or values |
|---|---|
| Case identifier | `case_id`, `case`, or `patient_id` |
| Split | `train`, `validation`, and `test`; `val` and `valid` are accepted aliases |
| Optional image path | `image` or `image_path` |
| Optional label path | `label` or `label_path` |

Without explicit paths, images and labels are resolved under `imagesTr/` and `labelsTr/`. Relative paths are resolved against `DATA_ROOT`; absolute paths are also supported. Case IDs can be derived from image filenames when no identifier column is present. Image and label filename stems must match the case identifiers.

Missing files, duplicate case identifiers or paths, ambiguous columns, and missing split groups stop execution. NIfTI spatial units must be millimetres. Evaluation requires labelled test cases from the supplied CSV; the unlabelled MSD challenge `imagesTs` set is not used.

## Installation

The project uses Python 3.11 or 3.12. From the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --prefix .venv --name eng2440-a2 --display-name "ENG2440 A2"
```

Windows activation:

```powershell
.venv\Scripts\Activate.ps1
```

The notebook uses CUDA when available and otherwise falls back to CPU. CUDA execution requires a compatible PyTorch installation and NVIDIA driver. Apple MPS is not enabled in this implementation.

## Running the notebook

Start JupyterLab from the project directory:

```bash
jupyter lab
```

Open `ENG2440_A2_BrainTumour.ipynb` with the **ENG2440 A2** kernel. The required path setting is in the setup cell:

```python
DATA_ROOT = Path('/path/to/Task01_BrainTumour')
```

**Restart Kernel and Run All Cells** executes the workflow in this order:

```text
Setup and split validation
→ Geometry inspection and preprocessing
→ Uniform-sampling training
→ Tumor-aware training
→ Validation comparison and configuration selection
→ Final test evaluation
→ Three-plane error analysis
→ Missing-modality experiment
→ Sliding-window seam experiment
→ Run record
```

Generated artifacts are written to `local_run/` in the working directory. A new run overwrites artifacts at the same paths. Input NIfTI files and the split CSV are never modified. The notebook has no automatic resume mechanism; a complete reproducible run starts from the first cell.

## Default configuration

| Component | Setting |
|---|---|
| Target grid | Per-case axis-aligned RAS+, 1 mm isotropic spacing |
| Interpolation | Linear for MRI; nearest neighbour for labels |
| Normalization | Per-channel z-score over non-zero voxels on the target grid |
| Spatial handling | Full field of view; high-end padding removed after inference |
| Augmentation | Training-only paired left–right flips and intensity scaling in [0.9, 1.1] |
| Model | 3D U-Net, two downsampling stages, base width 8, 85,985 parameters |
| Input / output | Four MRI channels / one whole-tumor logit channel |
| Patch size / batch size | 64 × 64 × 64 / 2 |
| Training budget | 10 epochs × 100 optimizer steps per experiment |
| Optimizer / learning rate | Adam / 0.001 |
| Loss | 0.5 binary cross-entropy with logits + 0.5 soft Dice loss |
| Random seed | 42 |
| Tumor-aware sampling | 50% tumor-centred draws, 50% uniform draws |
| Inference | 50% overlap, stride 32, equal-weight probability averaging |
| Binary threshold | 0.5 |
| Candidate postprocessing | Remove 6-connected components smaller than 0.10 mL |

Both sampling experiments use the same initial weights, case draws, augmentation seed stream, architecture, optimizer, loss, and update count. The observed fraction of patches containing tumor is measured separately from the forced-positive sampling probability.

Each experiment retains the checkpoint with the highest mean full-volume validation Dice, choosing the earliest epoch on ties. The final sampler is selected by validation Dice, with uniform sampling winning an exact tie. The component-removal rule is retained only when it improves validation mean Dice without reducing mean sensitivity by more than 0.01.

## Reproducibility

Python, NumPy, and PyTorch are seeded with `42`. Deterministic PyTorch algorithms are enabled, cuDNN benchmarking is disabled, and data loading uses zero worker processes. Reproducibility across different hardware or library versions is not guaranteed.

Patient-level separation is enforced before preprocessing. Test paths are checked initially, but test arrays are opened only after the selected configuration has been saved. The test reference-volume distribution is therefore computed during final evaluation. Model selection and postprocessing decisions use validation data only.

Run metadata is recorded automatically:

| File | Contents |
|---|---|
| `local_run/hardware.json` | Python and package versions, operating system, device, GPU name, and thread count |
| `local_run/frozen_config.json` | Selected configuration, seed, training settings, and split-file SHA256 |
| `local_run/run_record.json` | Hardware, configuration, experiment runtimes, and evaluation counts |
| `local_run/tables/*_history.csv` | Per-epoch losses, validation Dice, patch statistics, and elapsed time |

An exact installed-package snapshot can be exported with:

```bash
python -m pip freeze > environment_actual.txt
```

Training times include per-epoch validation. The total notebook timer includes preprocessing, experiments, evaluation, and any manual pauses. Seam-test timings measure the warmed-up inference loop, including transfers and probability aggregation, with CUDA synchronization. They exclude disk loading, preprocessing, and metric computation. Each case and overlap condition is timed once.

## Evaluation

Metrics are calculated per case on the target grid:

- Dice and intersection over union (IoU).
- Tumor sensitivity and false-positive volume.
- HD95 in millimetres.
- Reference and predicted volume in millilitres.
- Signed, absolute, and absolute percentage volume error.
- Number of empty predictions.

Tumor volume is the foreground voxel count multiplied by voxel volume in mm³, divided by 1,000. HD95 is the larger of the two directed 95th-percentile distances between 6-connected surface voxels.

When both masks are empty, Dice and IoU are 1 and HD95 is 0. When exactly one mask is empty, HD95 is infinite. Sensitivity and percentage volume error are undefined for an empty reference. Summaries include mean, standard deviation, median, counts of undefined/infinite values, and explicitly labelled finite-only statistics. Evaluation SD uses `ddof=0`; split-volume descriptive SD uses `ddof=1`.

Raw and candidate-postprocessed test results are both reported. The final method follows the validation decision. Missing-modality and overlap results are descriptive experiments and do not change the selected configuration.

## Generated outputs

```text
local_run/
├── cache/                       # Preprocessed image and label arrays
├── figures/                     # Geometry, learning curves, overlays, and bonus figures
├── tables/                      # Case-wise metrics, summaries, and experiment histories
├── probabilities/               # Reconstructed probability volumes
├── uniform_best.pt
├── tumor_aware_best.pt
├── hardware.json
├── frozen_config.json
├── run_record.json
├── bonus1_interpretation.txt
└── bonus2_interpretation.txt
```

Preprocessed arrays are memory-mapped to avoid loading the entire cohort into RAM. The float32 image cache requires approximately 143 MB per 240 × 240 × 155 four-channel case, excluding labels and probability volumes. Full-cohort storage can reach tens of GB. Full-volume validation and the additional experiments also contribute substantially to runtime; full-data runtime and GPU memory requirements have not been measured.

Dataset files, extracted arrays, caches, checkpoints, and virtual environments are excluded by `.gitignore`.

## Software verification

Function-level checks:

```bash
python sanity_checks.py
```

Complete synthetic workflow:

```bash
python sanity_checks.py --workflow
```

The workflow check executes every notebook code cell on eight synthetic NIfTI cases using 16³ patches, base width 4, batch size 1, two epochs, and two steps per epoch. Temporary synthetic data is removed afterward, and the notebook's default configuration is unchanged.

Checks cover geometry handling, normalization, split validation, metric edge cases, physical distances and volumes, patch extraction, paired augmentation, model backpropagation, reconstruction, and both additional experiments. The recorded synthetic workflow passed. Verification details are in `VERIFICATION.txt`.

The supplied notebook is unexecuted on the official dataset. Synthetic verification establishes software behavior, not segmentation performance on real MRI data.

## Limitations

The small model and training budget limit the experiment's scope. A single split and seed do not quantify variability across repeated runs, and external-site generalization has not been evaluated. Whole-tumor binarization discards subregion distinctions, while component filtering can remove genuine small lesions. The project is an educational research implementation and has not been clinically validated.

## References

- [Medical Segmentation Decathlon](https://medicaldecathlon.com/) and [Antonelli et al. (2022)](https://doi.org/10.1038/s41467-022-30695-9)
- [NiBabel coordinate systems](https://nipy.org/nibabel/coordinate_systems.html) and [resampling](https://nipy.org/nibabel/reference/nibabel.processing.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [SciPy Euclidean distance transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)
