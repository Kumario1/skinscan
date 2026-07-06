# SkinScan

SkinScan is a computer-vision learning project for acne analysis.

The repo has three separate jobs. Keep them mentally separate:

```text
1. Detector / locator
   Input: full face image
   Output: boxes around acne spots

2. Classifier
   Input: cropped acne spot
   Output: acne type: Blackheads, Cyst, Papules, Pustules, Whiteheads

3. Recommender
   Input: acne concerns
   Output: ingredient/product routine rules
```

This is not medical software. It uses cosmetic concern language only.

Latest tracked progress:

```text
Stage 1 lesion locator: YOLOv8m ACNE04 checkpoint, F1=0.722 at conf=0.07 / IoU=0.2
Stage 2 type classifier: Colab T4 EfficientNetB0 checkpoint, test accuracy=91.18%
End-to-end check: detector boxes -> saved lesion crops -> acne type probabilities
```

## Current Pipeline

```mermaid
flowchart LR
    A["full face image"] --> B["YOLO acne detector"]
    B --> C["acne location boxes"]
    C --> D["crop each box"]
    D --> E["EfficientNet acne type classifier"]
    E --> F["raw acne type probabilities"]
    F --> G["rules-based recommender"]
```

## What Is Where

| Thing | Path | Git? | Purpose |
|---|---|---:|---|
| Detector code | `src/detection/` | yes | ACNE04 conversion, visualization, detector checks |
| Detector weights | `models/detection/acne04_yolov8m_best.pt` | no | Stage 1 acne spot locator |
| Detector data | `data/raw/acne04/` | no | ACNE04 images + box labels |
| Classifier code | `src/classification/` | yes | Crop helper, EfficientNet classifier, classifier trainer |
| Classifier weights | `models/classification/acne_model.keras` | no | Stage 2 acne type classifier |
| Classifier data | `data/raw/typeclassification/AcneDataset/` | no | Kaggle acne type dataset |
| Recommendation code | `src/recommendation/` | yes | Concern-to-ingredient rules |
| Config | `configs/default.yaml` | yes | Current model paths and thresholds |
| Generated checks | `runs/` | no | Rendered sheets, predictions, threshold sweeps |

Raw data and model weights are intentionally ignored. The repo tracks code and
documentation; your machine keeps datasets, weights, and generated outputs.

## Stage 1: Detector / Lesion Locator

This is the lesion localization step.

Goal: given a full ACNE04 face image, draw boxes around acne spots.

Current detector:

```text
weights: models/detection/acne04_yolov8m_best.pt
data:    data/raw/acne04/
conf:    0.07
iou:     0.2
imgsz:   1024
```

Run the detector location check:

```bash
.venv/bin/python -m src.detection.check_acne04_detector
```

Fast smoke test:

```bash
.venv/bin/python -m src.detection.check_acne04_detector --limit 5 --render-limit 5
```

Outputs:

```text
runs/detection_check/gt_green_pred_red_sheet.jpg
runs/detection_check/threshold_sweep.json
```

On ACNE04 validation, the current best saved operating point is:

```text
conf=0.07, iou=0.2, imgsz=1024
precision=0.697
recall=0.750
F1=0.722
```

Green boxes are ACNE04 labels. Red boxes are the detector predictions.

![Detector check](assets/acne04_detector_gt_pred_sheet.jpg)

## Stage 2: Classifier

This model is separate from the detector. The current checkpoint was trained in
Colab on a T4 GPU with EfficientNetB0 transfer learning, class weights,
Adam(1e-5), ReduceLROnPlateau, and best-checkpoint saving on validation
accuracy.

Goal: given one cropped acne spot, predict the type.

Raw output classes:

```python
["Blackheads", "Cyst", "Papules", "Pustules", "Whiteheads"]
```

Training data:

```text
data/raw/typeclassification/AcneDataset/
  train/
  valid/
  test/
```

Latest T4 run:

```text
TensorFlow: 2.20.0
train/valid/test: 2778 / 921 / 918 images
best validation accuracy: 0.8979
test loss: 0.4999
test accuracy: 0.9118
macro F1: 0.92
weighted F1: 0.91
```

Per-class test report:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Blackheads | 0.94 | 0.95 | 0.95 | 265 |
| Cyst | 0.92 | 0.93 | 0.92 | 189 |
| Papules | 0.88 | 0.87 | 0.88 | 202 |
| Pustules | 0.88 | 0.87 | 0.88 | 205 |
| Whiteheads | 0.98 | 0.96 | 0.97 | 57 |

Training curves:

![T4 training curves](assets/stage2_t4_training_curves.png)

Confusion matrix:

![T4 confusion matrix](assets/stage2_t4_confusion_matrix.png)

Training code mirrors the Colab T4 notebook and the original Kaggle reference:
https://www.kaggle.com/code/dadydada/miniproject-ai-6610210284

It saves the best validation checkpoint:

Train:

```bash
.venv/bin/python -m src.classification.train_type_classifier
```

Inspect dataset counts:

```bash
.venv/bin/python -m src.classification.train_type_classifier --inspect
```

Expected classifier output:

```text
models/classification/acne_model.keras
models/classification/acne_model.keras.labels.json
```

The classifier has not replaced the detector. It only runs after the detector
has produced acne crops.

Inference expects raw RGB crops resized to 224x224 with pixel values 0-255.
Do not divide by 255 before calling the Keras model; EfficientNetB0 handles its
own normalization internally.

## Detector To Classifier

Once the classifier weights exist:

```bash
.venv/bin/python -m src.classification.run_acne04_pipeline
```

Run one uploaded/full-face image:

```bash
.venv/bin/python -m src.classification.run_acne04_pipeline --image path/to/image.jpg
```

That does:

```text
ACNE04 image -> detector boxes -> crop boxes -> classify crops
```

Outputs:

```text
runs/acne04_pipeline_check/predictions.json
runs/acne04_pipeline_check/*_crop_*.jpg
runs/acne04_pipeline_check/*_input_collage.jpg
runs/acne04_pipeline_check/*_crops.jpg
```

The pipeline check now keeps the lesion crop inputs and summarizes predicted
acne-type counts per image, which makes detector-to-classifier review easier.

Detector crop inputs:

![Pipeline detector crop inputs](assets/stage2_pipeline_input_collage_overview.jpg)

End-to-end crop predictions:

![Pipeline crop predictions](assets/stage2_pipeline_single_crop_overview.jpg)

## Stage 3: Recommender

The recommender is rules-based, not ML.

Code:

```text
src/recommendation/
```

It maps concerns to ingredients, for example:

```text
comedonal acne     -> salicylic acid / adapalene / azelaic acid
inflammatory acne  -> benzoyl peroxide / azelaic acid / niacinamide
cystic acne        -> soothing support + professional-care flag
```

## Setup

Install dependencies into the local venv:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The raw ACNE04 archives should be stored locally at:

```text
data/raw/acne04/archives/Detection.tar
data/raw/acne04/archives/Classification.tar
```

Extracted ACNE04 should look like:

```text
data/raw/acne04/Detection/VOC2007/Annotations/
data/raw/acne04/Detection/VOC2007/ImageSets/
data/raw/acne04/Classification/JPEGImages/
```

## Do Not Commit

These are intentionally local only:

```text
data/raw/
data/processed/
data/self_collected/
models/
runs/
*.tar
*.pt
*.keras
```

## Short Version

Right now, the useful checks are:

```bash
.venv/bin/python -m src.detection.check_acne04_detector
.venv/bin/python -m src.classification.run_acne04_pipeline
```
