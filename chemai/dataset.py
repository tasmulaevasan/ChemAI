import csv
from typing import List, Tuple

from torch.utils.data import Dataset

from .tokenizer import SmilesTokenizer


class ReactionDataset(Dataset):
    """Dataset reading a CSV file with 'smiles' and 'label' columns."""

    def __init__(self, path: str, tokenizer: SmilesTokenizer, max_length: int = 200):
        self.samples: List[Tuple[List[int], int]] = []
        smiles_list = []
        labels = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles_list.append(row["smiles"].strip())
                labels.append(int(row["label"]))
        tokenizer.fit(smiles_list)
        for sm, lab in zip(smiles_list, labels):
            ids = tokenizer.encode(sm)[:max_length]
            self.samples.append((ids, lab))
        self.pad_id = tokenizer.vocab["<pad>"]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def collate_fn(self, batch):
        max_len = max(len(ids) for ids, _ in batch)
        input_ids = []
        labels = []
        attention_mask = []
        for ids, lab in batch:
            padded = ids + [self.pad_id] * (max_len - len(ids))
            mask = [1] * len(ids) + [0] * (max_len - len(ids))
            input_ids.append(padded)
            attention_mask.append(mask)
            labels.append(lab)
        import torch

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
