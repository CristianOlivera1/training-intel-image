# Image Scene Classification

A deep learning project that classifies natural-scene images into 6 categories and exposes the trained models through a small desktop GUI (`app.py`).

## Dataset

- **Source:** Intel Image Classification (Kaggle). 6 balanced classes: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.
- **Training set:** 14,034 images. **Test set:** 3,000 images.
- **Input:** 150×150 RGB, batch size 32, with data augmentation.

## Objective

Train and compare CNN approaches on the same dataset:

1. **From scratch** - each architecture trained with random initialization.
2. **Transfer learning** - ImageNet pretrained weights with a frozen base (feature extraction).
3. **Fine-tuning** - same pretrained base, unfrozen for further training.

## Models

VGG16, ResNet50, InceptionV3, MobileNetV2 and DenseNet121, each trained in the 3 modes above, plus a small custom CNN (4 conv blocks) defined and compiled in the notebook.

## Results

### Models trained from scratch

| Model | Learning rate | Epochs (best) | Optimizer | Accuracy (Train) | Accuracy (Val) | Accuracy (Test) | Precision (macro) | Recall (macro) | F1-score (macro) |
|---|---|---|---|---|---|---|---|---|---|
| VGG16 | 1e-4 | 17 (17) | Adam | 0.8770 | 0.8717 | 0.8687 | 0.8704 | 0.8723 | 0.8700 |
| ResNet50 | 1e-4 | 14 (10) | Adam | 0.8254 | 0.7950 | 0.7877 | 0.8000 | 0.7935 | 0.7894 |
| MobileNetV2 | 1e-4 | 6 (1) | Adam | 0.3966 | 0.1700 | 0.1700 | 0.0283 | 0.1667 | 0.0484 |
| InceptionV3 | 1e-4 | 20 (16) | Adam | 0.8519 | 0.8480 | 0.8480 | 0.8537 | 0.8496 | 0.8492 |
| DenseNet121 | 1e-4 | 13 (8) | Adam | 0.8454 | 0.8207 | 0.8207 | 0.8398 | 0.8240 | 0.8197 |

### Models with fine-tuning

| Model | Learning rate | Epochs (best) | Optimizer | Accuracy (Train) | Accuracy (Val) | Accuracy (Test) | Precision (macro) | Recall (macro) | F1-score (macro) |
|---|---|---|---|---|---|---|---|---|---|
| VGG16 | 1e-5 | 15 (12) | Adam | 0.9266 | 0.9200 | 0.9157 | 0.9168 | 0.9180 | 0.9171 |
| ResNet50 | 1e-5 | 15 (11) | Adam | 0.7425 | 0.7730 | 0.7730 | 0.7746 | 0.7779 | 0.7742 |
| MobileNetV2 | 1e-5 | 15 (15) | Adam | 0.8985 | 0.9090 | 0.9090 | 0.9103 | 0.9121 | 0.9110 |
| InceptionV3 | 1e-5 | 15 (6) | Adam | 0.8602 | 0.8967 | 0.8957 | 0.8978 | 0.8987 | 0.8979 |
| DenseNet121 | 1e-5 | 15 (15) | Adam | 0.9091 | 0.9227 | 0.9227 | 0.9236 | 0.9249 | 0.9241 |

- **Best model overall: DenseNet121 fine-tuning** - test accuracy **0.9227**, macro F1 **0.9241**.
- Fine-tuning clearly outperforms training from scratch for most models.
- MobileNetV2 from scratch failed to converge (validation stuck at ~0.17; early stopping at epoch 6).
- ResNet50 is the only case where fine-tuning scored below scratch (overfits early).

## Environment & Hardware

- **GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (compute capability 8.6, ~3.6 GB available).
- Training ran on the **GPU from VS Code via WSL** (`wsl_env_312`, Python 3.12).
- **Stack:** TensorFlow 2.21.0, CUDA 12.5 / cuDNN 9.25, Keras ImageNet weights, scikit-learn for metrics.
- Adam optimizer: scratch at `lr=1e-4`, fine-tuning at `lr=1e-5`.
- `ModelCheckpoint` saved the best weights (by validation loss); `EarlyStopping` (patience 5) stopped runs that stalled.

## Application (app.py)

A Tkinter desktop app that loads any saved `.keras` model and predicts on a dropped/uploaded image:

- Drag & drop (via `tkinterdnd2`) or click to upload an image.
- Model selector that only shows models present in the folder.
- Top-3 predictions with confidence bars.
- Runs on **CPU** (`CUDA_VISIBLE_DEVICES=-1`) to keep the GPU free.
- Note: the GUI labels "VGG16 (fine-tuning)" as best, but the recorded results show **DenseNet121 fine-tuning** is the actual best.

## Files

- `script.ipynb` - full training pipeline with outputs (exploratory analysis, augmentation, training, metrics, confusion matrices, learning curves).
- `app.py` - desktop inference app.
- `*.keras` - trained model checkpoints for every architecture/mode.

## Usage

```bash
python app.py
```

Requires `tensorflow`, `Pillow`, `numpy` and optional `tkinterdnd2` for drag & drop.
