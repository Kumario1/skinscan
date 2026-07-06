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
python -m src.classification.train_type_classifier --inspect
python -m src.classification.train_type_classifier
```

The trainer replicates:

https://www.kaggle.com/code/dadydada/miniproject-ai-6610210284
