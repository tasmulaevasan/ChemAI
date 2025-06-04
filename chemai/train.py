import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import ReactionDataset
from .model import ReactionClassifier
from .tokenizer import SmilesTokenizer


def train(args):
    tokenizer = SmilesTokenizer()
    dataset = ReactionDataset(args.data, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    model = ReactionClassifier(
        vocab_size=len(tokenizer.vocab),
        num_classes=args.num_classes,
        emb_dim=args.emb_dim,
        nhead=args.nhead,
        num_layers=args.layers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output)
        print(f"Model saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Train reaction classifier")
    parser.add_argument("--data", required=True, help="CSV file with smiles,label")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--output", help="Path to save trained model")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":  # pragma: no cover
    main()
