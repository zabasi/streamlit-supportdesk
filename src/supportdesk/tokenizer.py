"""The project's own tokenizer, used for chunk sizes, the evidence budget and keyword search. It splits
text into words and keeps identifiers such as st.session_state and 1.32.0 whole. It is not the
model's tokenizer: costs come from the token counts the provider reports."""
from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+(?:['\-.,:][A-Za-z0-9_]+)*")

STOPWORDS = frozenset(
    ["a", "an", "and", "any", "are", "as", "at", "be", "been", "before", "being", "but", "by", "can", "cannot", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "may", "me", "my", "no", "not", "of", "on", "one", "only", "or", "other", "our", "out", "over", "own", "same", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to", "under", "until", "up", "upon", "was", "we", "were", "what", "when", "where", "which", "while", "who", "why", "will", "with", "within", "would", "you", "your"]
)


def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def count_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def normalize_token(token: str) -> str:
    return token.lower().strip("'-.,:")


def lexical_tokens(text: str) -> list[str]:
    """Normalised, stopword-free tokens: the representation BM25 and the offline embedder use."""
    out = []
    for token in tokenize(text):
        norm = normalize_token(token)
        if norm and norm not in STOPWORDS:
            out.append(norm)
    return out
