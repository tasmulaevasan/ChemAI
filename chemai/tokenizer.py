import re
from typing import List

class SmilesTokenizer:
    """A simple tokenizer for SMILES strings."""

    atom_regex = re.compile(r"Cl|Br")

    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2, "<sep>": 3}
        self.inv_vocab = {0: "<pad>", 1: "<unk>", 2: "<cls>", 3: "<sep>"}

    def fit(self, smiles_list: List[str]) -> None:
        for sm in smiles_list:
            for tok in self._tokenize(sm):
                if tok not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[tok] = idx
                    self.inv_vocab[idx] = tok

    def encode(self, smiles: str) -> List[int]:
        tokens = ["<cls>"] + self._tokenize(smiles) + ["<sep>"]
        return [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.inv_vocab.get(i, "<unk>") for i in ids]
        # remove special tokens
        tokens = [t for t in tokens if t not in {"<cls>", "<sep>", "<pad>"}]
        return "".join(tokens)

    @classmethod
    def _tokenize(cls, smiles: str) -> List[str]:
        tokens = []
        i = 0
        while i < len(smiles):
            if smiles[i:i+2] in {"Cl", "Br"}:
                tokens.append(smiles[i:i+2])
                i += 2
            else:
                tokens.append(smiles[i])
                i += 1
        return tokens
