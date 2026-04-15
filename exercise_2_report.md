# Problem 6.2: CNN Shelf Inspector

## 1. Introduction

Describe the task: classify synthetic shelf images into three classes (`normal`, `damaged`, `overloaded`) using a CNN and compare to a fully connected baseline.

## 2. Dataset Generation

- `src/shelf_cnn.py` generates `shelf_images.npz` if missing.
- Each image is 64x64 grayscale.
- Shelf position is jittered by ±4 pixels.
- Boxes are drawn with random width, height, and brightness.
- Damaged shelves include a thin crack through the shelf bar.
- Overloaded shelves use more boxes and a taller stack.
- Brightness jitter and Gaussian noise are added.

## 3. Data Split

- 70% training, 15% validation, 15% test.
- The dataset is shuffled before split.
- The validation set is used for early stopping and model selection.
- The test set is held out until final evaluation.

## 4. Model Architectures

### 4.1 CNN

- Three convolutional blocks:
  - Conv(1→16), ReLU, MaxPool
  - Conv(16→32), ReLU, MaxPool
  - Conv(32→64), ReLU, MaxPool
- Fully connected head: `128 → 3`
- Dropout after the hidden FC layer.
- Batch normalization is enabled by default.

### 4.2 Fully Connected Baseline

- Flatten 64×64 image into 4096 features.
- Two hidden layers: 512 and 256 units.
- Final fully connected layer outputs 3 classes.

### 4.3 Transfer Learning (Optional)

- Pretrained ResNet-18 adapted for 1-channel grayscale input.
- Final fully connected layer replaced with 3 outputs.
- Optionally freeze the backbone.

## 5. Regularization Toolkit

- Dropout after the dense layer.
- Weight decay in the Adam optimizer (`1e-4`).
- Data augmentation on the training set only:
  - random horizontal flips
  - small random rotations
  - brightness jitter
- Early stopping monitored on validation loss.

## 6. Training and Evaluation

### 6.1 Commands

```powershell
& .\.venv\Scripts\python.exe .\src\shelf_cnn.py --model cnn --epochs 40 --batch-size 64
```

```powershell
& .\.venv\Scripts\python.exe .\src\shelf_cnn.py --model fc --epochs 40 --batch-size 64
```

```powershell
& .\.venv\Scripts\python.exe .\src\shelf_cnn.py --model transfer --epochs 20 --batch-size 32
```

### 6.2 Outputs

- `outputs/training_history_{model}.png`
- `outputs/confusion_matrix_{model}.png`
- `outputs/example_predictions_{model}.png`
- `outputs/first_layer_filters.png` (CNN only)
- `outputs/best_model_{model}.pth`

## 7. Results

Include:
- Training and validation loss curves.
- Training and validation accuracy curves.
- Validation accuracy for each architecture.
- Test accuracy, precision, recall, and F1-score.
- Confusion matrix.
- Correct and incorrect sample predictions.
- First-layer filter visualizations.

## 8. Discussion

- Compare CNN vs FC baseline.
- Describe the effect of regularization.
- Explain any overfitting or underfitting behavior.
- Comment on whether transfer learning improves performance.

## 9. Conclusion

Summarize the final model choice and the main takeaways from the shelf inspector assignment.
