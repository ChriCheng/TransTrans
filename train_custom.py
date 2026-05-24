"""
Train an English-to-Chinese Transformer from scratch with plain PyTorch.

This script is intentionally separate from train.py, which fine-tunes a
pretrained MarianMT model.  It uses the assignment vocabulary files when
available and trains a randomly initialized encoder-decoder Transformer.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from dataload import load_assignment_dataset
from sacrebleu import corpus_bleu, corpus_chrf
from torch import nn
from torch.utils.data import DataLoader, Dataset


OUT_DIR = "out_custom"
MODEL_PATH = os.path.join(OUT_DIR, "custom_transformer.pt")
CONFIG_PATH = os.path.join(OUT_DIR, "run_config.json")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")
PREDICTIONS_PATH = os.path.join(OUT_DIR, "predictions.json")
TEST_SAMPLES_PATH = os.path.join(OUT_DIR, "test_samples.json")

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"


@dataclass
class CustomTrainConfig:
    data_dir: str = "data"
    train_size: int = 18000
    val_size: int = 500
    test_size: int = 2636
    max_source_length: int = 40
    max_target_length: int = 50
    d_model: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    batch_size: int = 64
    learning_rate: float = 3e-4
    epochs: int = 20
    clip_grad_norm: float = 1.0
    eval_every: int = 1
    max_train_steps: int = 0
    generation_max_length: int = 50
    seed: int = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def find_vocab_dir(data_dir):
    root = Path(data_dir)
    if not root.exists():
        return None
    for candidate in [root, *sorted(path for path in root.rglob("*") if path.is_dir())]:
        if (
            (candidate / "word2int_en.json").exists()
            and (candidate / "word2int_cn.json").exists()
            and (candidate / "int2word_cn.json").exists()
        ):
            return candidate
    return None


def load_vocab(data_dir):
    vocab_dir = find_vocab_dir(data_dir)
    if vocab_dir is None:
        raise FileNotFoundError(
            f"未在 {os.path.abspath(data_dir)} 下找到 word2int_en/cn.json 词表文件"
        )

    with open(vocab_dir / "word2int_en.json", "r", encoding="utf-8") as f:
        src_stoi = json.load(f)
    with open(vocab_dir / "word2int_cn.json", "r", encoding="utf-8") as f:
        tgt_stoi = json.load(f)
    with open(vocab_dir / "int2word_cn.json", "r", encoding="utf-8") as f:
        tgt_itos_raw = json.load(f)

    tgt_itos = {int(idx): token for idx, token in tgt_itos_raw.items()}
    return src_stoi, tgt_stoi, tgt_itos, str(vocab_dir)


def encode_tokens(text, stoi, max_length, add_bos=False, add_eos=True):
    tokens = text.strip().split()
    ids = []
    if add_bos:
        ids.append(stoi[BOS])
    ids.extend(stoi.get(token, stoi[UNK]) for token in tokens[:max_length])
    if add_eos:
        ids.append(stoi[EOS])
    return ids


def decode_target(ids, itos):
    tokens = []
    for idx in ids:
        token = itos.get(int(idx), UNK)
        if token == EOS:
            break
        if token in {PAD, BOS}:
            continue
        tokens.append(token)
    return " ".join(tokens).replace(" ,", "，").replace(" .", "。").strip()


class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_stoi, tgt_stoi, config):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_stoi = src_stoi
        self.tgt_stoi = tgt_stoi
        self.config = config

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, index):
        src_ids = encode_tokens(
            self.src_texts[index],
            self.src_stoi,
            self.config.max_source_length - 1,
            add_bos=False,
            add_eos=True,
        )
        tgt_ids = encode_tokens(
            self.tgt_texts[index],
            self.tgt_stoi,
            self.config.max_target_length - 2,
            add_bos=True,
            add_eos=True,
        )
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_batch(batch, src_pad_id, tgt_pad_id):
    src_batch, tgt_batch = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=src_pad_id)
    tgt = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_id)
    return src, tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class ScratchTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout,
        src_pad_id,
        tgt_pad_id,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_id)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def causal_mask(size, device):
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, src, tgt_input):
        src_key_padding_mask = src.eq(self.src_pad_id)
        tgt_key_padding_mask = tgt_input.eq(self.tgt_pad_id)
        tgt_mask = self.causal_mask(tgt_input.size(1), tgt_input.device)

        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt_input) * math.sqrt(self.d_model))

        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_projection(output)

    def greedy_decode(self, src, bos_id, eos_id, max_length):
        self.eval()
        src_key_padding_mask = src.eq(self.src_pad_id)
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        ys = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_length - 1):
            tgt_mask = self.causal_mask(ys.size(1), src.device)
            tgt_emb = self.positional_encoding(self.tgt_embedding(ys) * math.sqrt(self.d_model))
            out = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=ys.eq(self.tgt_pad_id),
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_token = self.output_projection(out[:, -1]).argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(eos_id)
            if finished.all():
                break
        return ys


def build_loaders(config, src_stoi, tgt_stoi):
    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = load_assignment_dataset(
        data_dir=config.data_dir,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
    )

    src_pad_id = src_stoi[PAD]
    tgt_pad_id = tgt_stoi[PAD]
    collate_fn = lambda batch: collate_batch(batch, src_pad_id, tgt_pad_id)
    train_dataset = TranslationDataset(train_src, train_tgt, src_stoi, tgt_stoi, config)
    val_dataset = TranslationDataset(val_src, val_tgt, src_stoi, tgt_stoi, config)
    test_dataset = TranslationDataset(test_src, test_tgt, src_stoi, tgt_stoi, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader, (train_src, train_tgt, val_src, val_tgt, test_src, test_tgt)


def train_one_epoch(model, loader, optimizer, criterion, config, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start = time.monotonic()

    for step, (src, tgt) in enumerate(loader, start=1):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
        optimizer.step()

        token_count = tgt_output.ne(model.tgt_pad_id).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count

        if step == 1 or step % 50 == 0 or step == len(loader):
            elapsed = time.monotonic() - start
            print(
                f"[train] epoch={epoch}/{total_epochs} step={step}/{len(loader)} "
                f"loss={total_loss / max(1, total_tokens):.4f} "
                f"elapsed={format_duration(elapsed)}",
                flush=True,
            )

        if config.max_train_steps and step >= config.max_train_steps:
            break

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        token_count = tgt_output.ne(model.tgt_pad_id).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def evaluate_generation(model, loader, tgt_stoi, tgt_itos, config, device, limit=None):
    model.eval()
    hypotheses = []
    references = []
    sources = []
    bos_id = tgt_stoi[BOS]
    eos_id = tgt_stoi[EOS]

    for src, tgt in loader:
        src = src.to(device)
        generated = model.greedy_decode(src, bos_id, eos_id, config.generation_max_length).cpu()
        for gen_ids, tgt_ids in zip(generated.tolist(), tgt.tolist()):
            hypotheses.append(decode_target(gen_ids, tgt_itos))
            references.append(decode_target(tgt_ids[1:], tgt_itos))
            sources.append(src.size(1))
            if limit and len(hypotheses) >= limit:
                break
        if limit and len(hypotheses) >= limit:
            break

    bleu = corpus_bleu(hypotheses, [references], tokenize="zh").score
    chrf = corpus_chrf(hypotheses, [references]).score
    exact_match = float(np.mean([hyp == ref for hyp, ref in zip(hypotheses, references)]))
    return {
        "bleu": bleu,
        "chrf": chrf,
        "exact_match": exact_match,
        "num_samples": len(hypotheses),
    }, hypotheses, references


def run_training(config):
    set_seed(config.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    src_stoi, tgt_stoi, tgt_itos, vocab_dir = load_vocab(config.data_dir)
    train_loader, val_loader, test_loader, raw_splits = build_loaders(config, src_stoi, tgt_stoi)
    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = raw_splits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"vocab_dir={vocab_dir}")
    print(f"train/val/test={len(train_src)}/{len(val_src)}/{len(test_src)}")

    model = ScratchTransformer(
        src_vocab_size=len(src_stoi),
        tgt_vocab_size=len(tgt_stoi),
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        src_pad_id=src_stoi[PAD],
        tgt_pad_id=tgt_stoi[PAD],
    ).to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={parameter_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_stoi[PAD])

    save_json(
        {
            "config": asdict(config),
            "vocab_dir": vocab_dir,
            "src_vocab_size": len(src_stoi),
            "tgt_vocab_size": len(tgt_stoi),
            "parameter_count": parameter_count,
            "model_type": "scratch_pytorch_transformer",
        },
        CONFIG_PATH,
    )
    save_json({"test_src_texts": test_src, "test_tgt_texts": test_tgt}, TEST_SAMPLES_PATH)

    best_val_loss = float("inf")
    history = []
    start = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config, device, epoch, config.epochs
        )
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        print(
            f"[eval] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"total_elapsed={format_duration(time.monotonic() - start)}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "src_vocab_size": len(src_stoi),
                    "tgt_vocab_size": len(tgt_stoi),
                    "src_pad_id": src_stoi[PAD],
                    "tgt_pad_id": tgt_stoi[PAD],
                },
                MODEL_PATH,
            )
            print(f"[save] best checkpoint -> {MODEL_PATH}", flush=True)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, hypotheses, references = evaluate_generation(
        model, test_loader, tgt_stoi, tgt_itos, config, device
    )
    metrics = {
        "best_val_loss": best_val_loss,
        "history": history,
        "test": test_metrics,
    }
    save_json(metrics, METRICS_PATH)
    save_json(
        [
            {"source": src, "reference": ref, "prediction": hyp}
            for src, ref, hyp in zip(test_src, references, hypotheses)
        ],
        PREDICTIONS_PATH,
    )

    print("[test] " + json.dumps(test_metrics, ensure_ascii=False), flush=True)
    print(f"结果已保存到 {OUT_DIR}/")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a scratch PyTorch Transformer.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-size", type=int, default=18000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=2636)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-source-length", type=int, default=40)
    parser.add_argument("--max-target-length", type=int, default=50)
    parser.add_argument("--generation-max-length", type=int, default=50)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return CustomTrainConfig(**vars(args))


if __name__ == "__main__":
    run_training(parse_args())
