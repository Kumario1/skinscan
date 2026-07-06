# Data

Raw datasets stay here locally and are ignored by Git.

Current local training data:

```text
data/raw/typeclassification/AcneDataset/
  train/
  valid/
  test/
```

Use it with:

```bash
.venv/bin/python -m src.classification.train_type_classifier --inspect
.venv/bin/python -m src.classification.train_type_classifier
```

The trainer replicates:

https://www.kaggle.com/code/dadydada/miniproject-ai-6610210284

Detector-to-classifier smoke test expects:

```text
models/detection/acne04_yolov8m_best.pt
models/classification/acne_model.keras
data/raw/acne04/Classification/JPEGImages/
```

Run:

```bash
.venv/bin/python -m src.classification.run_acne04_pipeline
```

Detector-only location check expects:

```text
Detection.tar + Classification.tar extracted under data/raw/acne04/
models/detection/acne04_yolov8m_best.pt
```

Run:

```bash
.venv/bin/python -m src.detection.check_acne04_detector
```
