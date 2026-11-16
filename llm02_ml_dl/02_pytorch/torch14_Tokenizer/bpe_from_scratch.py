"""A small, dependency-free BPE tokenizer for learning purposes.

This implementation uses characters plus an end-of-word marker as the initial
symbols. It trains merge rules from a whitespace-tokenized corpus, then applies
the learned rules in rank order during encoding.

It is intentionally not a production tokenizer: it omits byte-level handling,
Unicode normalization, fast data structures, offset mapping, and serialization.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

Pair = tuple[str, str]
Word = tuple[str, ...]


def _merge_pair(pair: Pair, word: Word) -> Word:
    """Merge every non-overlapping occurrence of pair in one symbol sequence."""
    merged: list[str] = []
    index = 0
    while index < len(word):
        if index + 1 < len(word) and (word[index], word[index + 1]) == pair:
            merged.append(word[index] + word[index + 1])
            index += 2
        else:
            merged.append(word[index])
            index += 1
    return tuple(merged)


@dataclass
class SimpleBPE:
    """Educational character-level BPE tokenizer."""

    num_merges: int = 100
    end_of_word: str = "</w>"
    unk_token: str = "[UNK]"
    merges: list[Pair] = field(default_factory=list, init=False)
    merge_ranks: dict[Pair, int] = field(default_factory=dict, init=False)
    vocab: set[str] = field(default_factory=set, init=False)
    is_trained: bool = field(default=False, init=False)

    def _word_to_symbols(self, word: str) -> Word:
        if not word:
            return ()
        return tuple(word) + (self.end_of_word,)

    @staticmethod
    def _pair_frequencies(word_counts: Counter[Word]) -> Counter[Pair]:
        frequencies: Counter[Pair] = Counter()
        for word, count in word_counts.items():
            for pair in zip(word, word[1:]):
                frequencies[pair] += count
        return frequencies

    def fit(self, corpus: Iterable[str]) -> "SimpleBPE":
        """Learn frequent-pair merges from a whitespace-tokenized corpus."""
        text_counts: Counter[str] = Counter()
        for text in corpus:
            text_counts.update(text.split())
        if not text_counts:
            raise ValueError("corpus must contain at least one non-empty word")

        word_counts: Counter[Word] = Counter(
            {self._word_to_symbols(word): count for word, count in text_counts.items()}
        )
        self.merges = []
        for _ in range(self.num_merges):
            frequencies = self._pair_frequencies(word_counts)
            if not frequencies:
                break
            best_pair, _ = max(
                frequencies.items(), key=lambda item: (item[1], item[0])
            )
            self.merges.append(best_pair)
            word_counts = Counter(
                {
                    _merge_pair(best_pair, word): count
                    for word, count in word_counts.items()
                }
            )

        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab = {self.unk_token, self.end_of_word}
        for word in word_counts:
            self.vocab.update(word)
        self.is_trained = True
        return self

    def encode_word(self, word: str) -> list[str]:
        """Apply the lowest-rank available merge until no rule is usable."""
        if not self.is_trained:
            raise RuntimeError("call fit() before encode_word()")

        symbols = list(self._word_to_symbols(word))
        while len(symbols) > 1:
            candidates = [
                (self.merge_ranks[pair], pair)
                for pair in zip(symbols, symbols[1:])
                if pair in self.merge_ranks
            ]
            if not candidates:
                break
            _, best_pair = min(candidates)
            symbols = list(_merge_pair(best_pair, tuple(symbols)))
        return [
            symbol if symbol in self.vocab else self.unk_token
            for symbol in symbols
        ]

    def encode(self, text: str) -> list[str]:
        """Whitespace pre-tokenize text, then encode every word."""
        tokens: list[str] = []
        for word in text.split():
            tokens.extend(self.encode_word(word))
        return tokens

    def decode(self, tokens: Iterable[str]) -> str:
        """Recover words from end-of-word markers; UNK remains visible."""
        words: list[str] = []
        current = ""
        for token in tokens:
            if token == self.end_of_word:
                if current:
                    words.append(current)
                current = ""
            elif token.endswith(self.end_of_word):
                current += token[: -len(self.end_of_word)]
                words.append(current)
                current = ""
            else:
                current += token
        if current:
            words.append(current)
        return " ".join(words)


if __name__ == "__main__":
    training_corpus = [
        "low lower newest widest",
        "low lowest newer wider",
        "low lower lower",
    ]
    bpe = SimpleBPE(num_merges=20).fit(training_corpus)
    print("learned merges:", bpe.merges)

    text = "lowest newer"
    tokens = bpe.encode(text)
    print("input:", text)
    print("tokens:", tokens)
    print("decode:", bpe.decode(tokens))

    assert bpe.decode(bpe.encode("low lower")) == "low lower"
    assert bpe.encode_word("not_seen")
