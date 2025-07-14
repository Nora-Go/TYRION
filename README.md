# TYRION: Self-Supervised Learning for Glacier Calving Front Extraction from SAR Imagery

This repository contains the official implementation of the paper "SSL4SAR: Self-Supervised Learning for Glacier Calving Front Extraction from SAR Imagery" published in IEEE Transactions on Geoscience and Remote Sensing.

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TGRS-blue)](https://ieeexplore.ieee.org/abstract/document/11039838)

## Overview

TYRION is a deep learning framework designed for glacier calving front extraction from Synthetic Aperture Radar (SAR) imagery. The project introduces two novel self-supervised learning approaches:

1. **OptSimMIM**: A cross-modal masked image modeling approach for SAR imagery
2. **OptTranslator**: A cross-modal translation approach between SAR and optical imagery

These self-supervised learning methods enable the model to learn meaningful representations from unlabeled SAR data, improving performance on the downstream task of glacier calving front extraction.

## Architecture

TYRION is based on the Swin Transformer V2 architecture with the following key components:

- **Backbone**: Modified Swin Transformer V2 for feature extraction
- **Decoder**: Multi-scale feature fusion CNN decoder for precise segmentation
- **Self-Supervised Learning Modules**:
  - OptSimMIM: Cross-modal masked image modeling for SAR imagery
  - OptTranslator: Cross-modal translation between SAR and optical imagery

The model performs semantic segmentation of SAR images into four classes:
1. Not Available (NA) areas
2. Rock
3. Glacier
4. Ocean and ice mélange

The glacier calving front is then extracted from the boundaries between glacier and ocean/ice mélange regions.

## Installation

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/Nora-Go/TYRION.git
cd TYRION

# Create and activate conda environment
conda env create -f environment.yml
conda activate mocca2
```

### Requirements
Download swinv2_tiny_patch4_window16_256.pth from [GitHub page](https://github.com/microsoft/Swin-Transformer/tree/main) and put it in the project folder directly.

The project requires:
- PyTorch (>=2.1.1)
- PyTorch Lightning (>=2.1.0)
- CUDA-enabled GPU
- Additional dependencies listed in environment.yml

## Usage

### Data Preparation

Download the CaFFe dataset here: [CaFFe Dataset](https://doi.pangaea.de/10.1594/PANGAEA.940950) and the pretraining dataset here: [SSL4SAR Dataset](https://zenodo.org/records/14748506).

If you are training locally, fill in the entries "parent_dir" and "parent_dir_optical" in "work.yaml," "work_OptSimMIM.yaml," and "work_OptTranslator.yaml."

Before running pretraining or training on a cluster, you need to prepare your data in the expected directory structure:

```
/scratch/
├── dataCaFFeBig/              # For training/evaluation data
│   └── data_raw/
│       ├── sar_images/
│       ├── zones/
│       ├── fronts/
│       └── bounding_boxes/
├── data_unlabeled/            # For pretraining data
│   ├── unlabeled_data_raw_train/
│   └── unlabeled_data_raw_val/
└── data_optical/              # For optical imagery (used in pretraining)
    └── data_optical/
```

The shell scripts include commands to extract data from tar archives to these locations.

### Pretraining

TYRION uses two self-supervised pretraining approaches that can be run independently:

#### OptSimMIM Pretraining

This approach uses cross-modal masked image modeling to learn representations from SAR imagery:

```bash
# For A100 GPUs
bash A100_pretraining_OptSimMIM.sh

# Or run directly
python pretraining_OptSimMIM.py
```

**Key Configuration Options** (in `Config/model/OptSimMIM.yaml`):
- `mask_ratio`: Proportion of image patches to mask (default: 0.3)
- `lr`: Learning rate (default: 0.001)
- `weight_decay`: Weight decay for regularization (default: 1e-4)
- `img_size`: Input image size (default: 512)
- `patch_size`: Patch size for tokenization (default: 4)
- `embed_dim`: Embedding dimension (default: 96)
- `depths`: Depth of each Swin Transformer layer (default: [2, 2, 6, 2])
- `window_size`: Window size for attention (default: 16)

#### OptTranslator Pretraining

This approach learns to translate SAR images to optical images:

```bash
# For A100 GPUs
bash A100_pretraining_SAR2OPT.sh

# Or run directly
python pretraining_OptTranslator.py
```

**Key Configuration Options** (in `Config/model/OptTranslator.yaml`):
- `lr`: Learning rate (default: 0.005)
- `weight_decay`: Weight decay for regularization (default: 1e-4)
- `img_size`: Input image size (default: 512)
- `patch_size`: Patch size for tokenization (default: 4)
- `embed_dim`: Embedding dimension (default: 96)
- `depths`: Depth of each Swin Transformer layer (default: [2, 2, 6, 2])
- `window_size`: Window size for attention (default: 16)

Both pretraining methods use PyTorch Lightning for training with features like:
- Automatic checkpointing of best models
- Early stopping
- Learning rate monitoring
- TensorBoard logging
- Multi-GPU training with DDP strategy

### Training

After pretraining, you can train the TYRION model for glacier calving front extraction:

```bash
# Training with pretraining
bash A100_train_after_pretraining.sh

# Training without pretraining
bash A100_train_without_sentinel1_pretraining.sh

# Or run directly
python train_Tyrion.py
```

**Key Training Parameters** (in `train_Tyrion.py`):
- `batch_size`: Batch size per GPU (default: 128, divided by number of GPUs and gradient accumulation steps)
- `devices`: Number of GPUs to use (default: 4)
- `accumulated_grad_batches`: Gradient accumulation steps (default: 2)
- `max_epochs`: Maximum number of training epochs (default: 150)

To use a pretrained model, update the `ckpt` path in the model configuration file (`Config/model/Tyrion.yaml`).

### Evaluation

Evaluate your trained model on test data:

```bash
# Standard evaluation
python Evaluation.py --mode test --data_raw_parent_dir /path/to/data --model_config Tyrion.yaml

# Evaluation with ensembling and test-time augmentation (improves results)
python Evaluation_test_time_aug_ensemble.py --mode test --data_raw_parent_dir /path/to/data
```

**Evaluation Options**:
- `--mode`: Evaluation mode, either "test" or "val" (default: "val")
- `--data_raw_parent_dir`: Path to the directory containing the data_raw folder (default: "..")
- `--with_overlap`: Use overlapping patches for evaluation (flag)
- `--experiment_name`: Name for the experiment in results (default: "testtest")
- `--model_config`: Model configuration file to use (default: "Tyrion.yaml")

Update the `ckpt` path in the model configuration file (`Config/model/Tyrion.yaml` or `Config/model/Ensemble_0.yaml`, `Config/model/Ensemble_1.yaml`, `Config/model/Ensemble_2.yaml`, `Config/model/Ensemble_3.yaml`, and `Config/model/Ensemble_4.yaml`) to point to the trained model checkpoint.

The evaluation script produces:
- Segmentation metrics (IoU, Precision, Recall, F1-Score)
- Front delineation metrics (Mean Distance Error)
- Visualizations of predicted fronts
- CSV files with detailed results

## Project Structure

```
TYRION/
├── A100_*.sh                  # Shell scripts for running on A100 GPUs
├── environment.yml            # Conda environment file
├── pretraining_*.py           # Self-supervised pretraining scripts
├── train_Tyrion.py            # Main training script
├── Evaluation.py              # Evaluation script
├── Evaluation_test_time_aug_ensemble.py  # Ensemble test-time augmentation evaluation
├── CaFFe/                     # Data handling modules
│   ├── data.py                # Data processing utilities
│   ├── fetcher.py             # Data fetching utilities
│   ├── augmentations/         # Data augmentation modules
│   └── ...
├── Config/                    # Configuration files
│   ├── Data/                  # Data configuration
│   └── model/                 # Model configuration
├── model/                     # Model implementation
│   ├── modules/               # Model architecture modules
│   │   ├── Tyrion.py          # Main TYRION model
│   │   ├── SwinV2.py          # Swin Transformer V2 implementation
│   │   └── ...
│   ├── pretraining/           # Self-supervised pretraining modules
│   │   ├── OptSimMIM.py       # Masked image modeling
│   │   ├── OptTranslator.py   # SAR to optical translation
│   │   └── ...
│   └── loss/                  # Loss functions
└── Evaluation_modules/        # Evaluation utilities
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Gourmelon.2025,
  author={Gourmelon, Nora and Dreier, Marcel and Mayr, Martin and Seehaus, Thorsten and Pyles, Dakota and Braun, Matthias and Maier, Andreas and Christlein, Vincent},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SSL4SAR: Self-Supervised Learning for Glacier Calving Front Extraction From SAR Imagery}, 
  year={2025},
  volume={63},
  pages={1-12},
  doi={10.1109/TGRS.2025.3580945}
}

```


## Contributing

Contributions to TYRION are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- This research was supported by the Bayerisches Staatsministerium für Wissenschaft und Kunst within the Elite Network Bavaria through the International Doctorate Program “Measuring and Modelling Mountain Glaciers and Ice Caps in a Changing Climate” (IDP M3OCCA); in part by German Research Foundation (DFG) through the Project “Large-Scale Automatic Calving Front Segmentation and Frontal Ablation Analysis of Arctic Glaciers Using Synthetic-Aperture Radar Image Sequences (LASSI)” and the Project “PAGE” within the DFG Emmy Noether Programme
- We thank the providers of the satellite data under various AOs from respective space agencies (DLR, ESA, JAXA, and CSA)
- We acknowledge the computational resources provided by Erlangen National High Performance Computing Center (NHR@FAU), Friedrich–Alexander-Universität Erlangen–Nürnberg (FAU)

## License

This project is licensed under the terms of the LICENSE file included in the repository.
