# Tumor Segmentation from Multimodal MRI

A PyTorch implementation of binary whole-tumor segmentation using the Medical Segmentation Decathlon `Task01_BrainTumour` dataset. The project was developed for ENG2440 Medical Imaging & AI in Healthcare.

The notebook compares uniform and tumor-aware patch sampling with the same lightweight 3D U-Net and training budget. It includes geometry inspection, preprocessing, full-volume evaluation, three-plane error analysis, a missing-modality stress test, and a sliding-window overlap experiment.

## Repository files

```text
Tumor-Segmentation-MRI/
├── Assignment2_Indira_Duisembayeva.ipynb  # Main implementation, outputs and discussion
├── requirements.txt                     # Pinned Python dependencies
└── README.md
```

The dataset, supplied split CSV, trained checkpoints and generated run directory are not tracked in this repository. There is no separate training script or standalone test suite; the implementation and inline checks are in the notebook.

## Dataset setup

Obtain [MSD Task01_BrainTumour](https://medicaldecathlon.com/) and the course-supplied `assignment2_split.csv` separately. Place them as follows:

```text
Tumor-Segmentation-MRI/
├── Assignment2_Indira_Duisembayeva.ipynb
├── requirements.txt
├── README.md
└── Task01_BrainTumour/
    ├── dataset.json
    ├── assignment2_split.csv
    ├── imagesTr/
    │   └── BRATS_*.nii.gz
    └── labelsTr/
        └── BRATS_*.nii.gz
```

The notebook sets `DATA_ROOT = Path("Task01_BrainTumour")`. The split CSV must be inside that directory, not only beside the notebook. Its required columns are:

```text
case_id,split,image_relpath,label_relpath
```

Image and label paths are resolved relative to `DATA_ROOT`. The supplied split contains 484 cases: 338 training, 73 validation and 73 test cases. The code uses these assignments without generating a new split and checks file existence, unique case identifiers and unique image/label paths. Test cases use the labelled files specified by this CSV, not the unlabelled challenge `imagesTs` directory.

Each image contains four MRI channels: FLAIR, T1, contrast-enhanced T1 (T1ce) and T2. The file channel order is read from `dataset.json` and mapped to that model input order. All non-zero reference labels are combined into one whole-tumor mask. Spatial units are required to be millimetres.

## Installation

Use Python 3.11 or 3.12. From a terminal:

```bash
git clone https://github.com/indiradu/Tumor-Segmentation-MRI.git
cd Tumor-Segmentation-MRI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` instead. Create the environment on the machine where computation will run; a virtual environment copied from macOS cannot be reused on a Linux HPC node.

The notebook selects CUDA when available and otherwise uses the CPU. Check the active notebook kernel before GPU training:

```python
import sys
import torch

print(sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

On an HPC cluster, start Jupyter within an allocated GPU job and connect the notebook to that server. An SSH connection to a login node alone does not give the notebook a GPU. The runtime must have a compatible NVIDIA driver and CUDA-enabled PyTorch installation.

## Experiment configuration

| Component | Default |
| --- | --- |
| Model | `SmallUNet3D`, base width 8, 85,985 parameters |
| Input / output channels | 4 MRI channels / 1 binary-mask logit |
| Target grid | Per-case axis-aligned RAS+, 1 mm isotropic spacing |
| Interpolation | Linear MRI; nearest-neighbour labels |
| Normalization | Per-channel z-score using non-zero voxels after resampling |
| Padding | High-end padding to at least patch size and a multiple of 4 |
| Training augmentation | Paired left–right flips; image intensity scaling from 0.9 to 1.1 |
| Patch / batch size | 64 × 64 × 64 / 2 |
| Budget per sampler | 10 epochs × 100 updates = 1,000 optimizer updates |
| Optimizer / learning rate | Adam / 0.001 |
| Loss | 0.5 BCE with logits + 0.5 soft Dice loss |
| Seed / DataLoader workers | 42 / 0 |
| Tumor-aware sampling | 50% forced tumor-centred draws; remaining draws uniform |
| Default inference | 50% overlap, stride 32, equal-weight probability averaging |
| Prediction threshold | 0.5, applied after probability averaging |
| Candidate postprocessing | Remove 6-connected components smaller than 0.10 mL |

The measured fraction of patches containing tumor can exceed the forced tumor-sampling probability because uniform draws can also contain tumor.

## Reproducibility and model selection

Both sampling experiments reset the seed and load identical initial model weights. They use the same architecture, split, loss, optimizer, update budget, case-draw sequence and augmentation random streams. The patch-location sampling rule is the experimental difference.

The checkpoint with the highest mean full-volume validation Dice is retained for each sampler; the earliest epoch wins a tie. Final sampler selection uses validation Dice, with uniform sampling winning an exact tie. The component-removal rule is retained only if validation mean Dice improves and mean sensitivity decreases by no more than 0.01. These decisions are saved before final test evaluation. The bonus test experiments describe robustness and do not select a new configuration using test labels.

### Environment and run records

`requirements.txt` pins dependencies, including `torch==2.6.0`; it is not an automatically captured record of every environment used to execute saved notebook outputs. To record the actual environment, run these commands from the activated environment used for the notebook:

```bash
python --version > environment_python.txt
python -m pip freeze > environment_actual.txt
```

For a GPU run, also record:

```bash
nvidia-smi > environment_gpu.txt
```

The notebook computes the split CSV's SHA256 and stores it in `selected_config.json` and `frozen_config.json`, together with selected model settings. Preserve these files, the notebook, checkpoint files and result tables together. The notebook does not automatically generate `hardware.json` or `run_record.json`.

Results and timings can vary across hardware, driver versions and dependency versions. The overlap benchmark uses a warm-up pass and CUDA synchronization, measuring inference and probability reconstruction but excluding file loading, preprocessing and metric calculation. Each overlap condition is measured once per case, so runtime differences are descriptive rather than repeated benchmark estimates.

## Evaluation and generated files

Evaluation uses complete volumes on the preprocessing target grid, with artificial padding removed. Reported metrics include Dice, IoU, sensitivity, HD95 in millimetres, false-positive volume, reference/predicted volumes, absolute volume error and absolute percentage volume error.

Volume in millilitres is voxel count × voxel volume in mm³ / 1,000. HD95 is the larger of the two directed 95th-percentile surface distances. When exactly one mask is empty, HD95 is infinite; when both are empty, Dice and IoU are 1 and HD95 is 0. Sensitivity and percentage volume error are undefined for an empty reference. Summaries report mean, population standard deviation, median and undefined/infinite counts. Infinite values are not silently removed: infinite HD95 makes its mean infinite and its standard deviation undefined.

```text
local_run/
├── cache/<case_id>/
│   ├── image.npy
│   ├── mask.npy
│   ├── tumor_indices.npy
│   └── geometry.json
├── figures/                  # Saved PNG plots and overlays
├── tables/                   # CSV histories, comparisons and case-wise metrics
├── probabilities/            # Test probability arrays, one .npy per case
├── uniform_best.pt
├── tumor_aware_best.pt
├── selected_config.json
└── frozen_config.json
```

The full-volume float32 cache can require tens of gigabytes. Data, virtual environments and generated artifacts should remain outside Git; the repository's three tracked source/documentation files are sufficient to distribute the implementation, but reproducing an experiment also requires the original dataset and supplied split.

## Limitations

The model uses a small training budget, limited patch context, one split and one random seed. External-site validation has not been performed. Binary whole-tumor masks discard tumor-subregion distinctions, and component filtering may remove genuine small lesions. The missing-modality experiment measures sensitivity to zero-filled inputs rather than all consequences of an unavailable acquisition. Overlap comparisons on test data are descriptive and must not be used for repeated test-set tuning.
