"""
RoBERTa embedding extraction via PyTorch.
Embeddings are extracted ONCE and cached to disk — never re-extracted per epoch.
"""

import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs.config import (
    ROBERTA_CHECKPOINT,
    MAX_LEN,
    ROBERTA_DIM,
    EMBED_BATCH_SIZE,
    EMBEDDINGS_CACHE,
    SEED,
)


class TextDataset(Dataset):
    """Minimal Dataset wrapper for tokenized text."""

    def __init__(self, texts: list[str], tokenizer, max_len: int) -> None:
        self.encodings = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.encodings.items()}


def _mean_pool(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Attention-mask-aware mean pooling.
    Padding tokens (mask=0) contribute nothing to the average.
    """
    mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class EmbeddingExtractor:
    """
    Wraps the frozen RoBERTa model for batch embedding extraction.
    Designed to be instantiated once and reused across train/val/test splits.
    """

    def __init__(self, checkpoint: str = ROBERTA_CHECKPOINT) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[EmbeddingExtractor] Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(self.device)

        # Freeze all parameters — we never update RoBERTa weights
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract(
        self,
        texts: list[str],
        batch_size: int = EMBED_BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract mean-pooled embeddings for a list of cleaned texts.
        Returns: np.ndarray of shape (N, ROBERTA_DIM)
        """
        dataset = TextDataset(texts, self.tokenizer, MAX_LEN)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_embeddings: list[np.ndarray] = []

        for batch in tqdm(loader, desc="Extracting embeddings", disable=not show_progress):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = _mean_pool(outputs.last_hidden_state, attention_mask)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def release(self) -> None:
        """
        Explicitly free GPU memory after extraction is complete.
        Critical when handing off to TensorFlow on the same GPU.
        """
        self.model.cpu()
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[EmbeddingExtractor] GPU memory released.")


def extract_and_cache(
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
    cache_path: Path = EMBEDDINGS_CACHE,
    force_recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract embeddings for all splits, caching to disk.
    On subsequent runs, loads from cache unless force_recompute=True.
    This is the key optimization: RoBERTa runs exactly once per dataset.
    """
    if cache_path.exists() and not force_recompute:
        print(f"[features] Loading embeddings from cache: {cache_path}")
        data = np.load(cache_path)
        return data["train"], data["val"], data["test"]

    print("[features] Cache not found — extracting embeddings...")
    extractor = EmbeddingExtractor()

    train_emb = extractor.extract(train_texts)
    val_emb   = extractor.extract(val_texts)
    test_emb  = extractor.extract(test_texts)

    # Release GPU before TensorFlow takes over
    extractor.release()

    np.savez_compressed(cache_path, train=train_emb, val=val_emb, test=test_emb)
    print(f"[features] Embeddings cached to: {cache_path}")

    return train_emb, val_emb, test_emb