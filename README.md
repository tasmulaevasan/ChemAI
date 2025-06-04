# ChemAI

This repository provides a minimal Transformer-based classifier for chemical reactions expressed as SMILES strings. The code is organized as a small Python package located in `chemai/`.

## Modules

- `tokenizer.py` – simple SMILES tokenizer.
- `dataset.py` – dataset loader that reads a CSV file with `smiles` and `label` columns and prepares batches.
- `model.py` – PyTorch implementation of the Transformer encoder and classification head.
- `train.py` – script for training the classifier.

## Example Usage

```
python -m chemai.train --data reactions.csv --num-classes 20 --epochs 10 --output model.pt
```

The input CSV should have two columns:

```
smiles,label
CCO.CC>>CCOC,0
...
```

The script will train a simple model and save weights to `model.pt`.
