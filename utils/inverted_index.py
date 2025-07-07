from collections import defaultdict
import re
from typing import Iterable, Set

class InvertedIndex:
    """Simple in-memory inverted index."""

    def __init__(self) -> None:
        self.index = defaultdict(set)

    def add_document(
        self, text: str, doc_id: int, extra: Iterable[str] | None = None
    ) -> None:
        tokens = self._tokenize(text)
        if extra:
            for value in extra:
                if value and isinstance(value, str):
                    tokens.extend(self._tokenize(value))
        for token in tokens:
            self.index[token].add(doc_id)

    def query(self, text: str) -> Set[int]:
        tokens = self._tokenize(text)
        results = None
        for token in tokens:
            docs = self.index.get(token)
            if docs:
                results = docs if results is None else results.intersection(docs)
        return results if results else set()

    def query_any(self, text: str) -> Set[int]:
        """Return documents containing any of the tokens in the text."""
        tokens = self._tokenize(text)
        results: Set[int] = set()
        for token in tokens:
            docs = self.index.get(token)
            if docs:
                results.update(docs)
        return results
    
    @staticmethod
    def _tokenize(text: str) -> list:
        return re.findall(r"\b\w{3,}\b", text.lower())