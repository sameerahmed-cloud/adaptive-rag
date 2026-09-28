import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List
from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore

@dataclass
class QueryProfile:
    """Lightweight query analysis used to select an adaptive retrieval path."""

    mode: str
    reason: str
    needs_exact_match: bool = False
    is_broad: bool = False
    is_complex: bool = False


class LexicalIndex:
    """
    Small dependency-free BM25-style lexical index.

    It indexes node text plus useful metadata so exact identifiers, filenames,
    fields, error codes, SKUs, and similar lexical signals can be retrieved
    even when dense similarity is not the strongest signal.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.nodes = []
        self.term_frequencies = []
        self.document_frequencies = Counter()
        self.document_lengths = []
        self.average_document_length = 0.0
        self.built = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    @staticmethod
    def searchable_text(node) -> str:
        metadata = getattr(node, "metadata", {}) or {}
        metadata_parts = [
            str(metadata.get(key, ""))
            for key in (
                "file_name",
                "file_path",
                "sheet_name",
                "language",
                "document_type",
                "chunk_strategy",
            )
        ]
        return " ".join(
            [node.get_content()] + metadata_parts
        )

    def build(self, nodes: List) -> None:
        self.nodes = list(nodes)
        self.term_frequencies = []
        self.document_frequencies = Counter()
        self.document_lengths = []

        for node in self.nodes:
            terms = self.tokenize(self.searchable_text(node))
            frequencies = Counter(terms)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(terms))
            self.document_frequencies.update(frequencies.keys())

        total_length = sum(self.document_lengths)
        self.average_document_length = (
            total_length / len(self.nodes)
            if self.nodes
            else 0.0
        )
        self.built = True

    def retrieve(self, query: str, top_k: int = 8) -> List[NodeWithScore]:
        if not self.built or not self.nodes:
            return []

        query_terms = self.tokenize(query)
        if not query_terms:
            return []

        query_counter = Counter(query_terms)
        total_documents = len(self.nodes)
        average_length = self.average_document_length or 1.0

        scored = []
        for index, node in enumerate(self.nodes):
            frequencies = self.term_frequencies[index]
            document_length = self.document_lengths[index] or 1
            score = 0.0

            for term in query_counter:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue

                document_frequency = self.document_frequencies.get(term, 0)
                idf = math.log(
                    1.0
                    + (
                        (total_documents - document_frequency + 0.5)
                        / (document_frequency + 0.5)
                    )
                )

                denominator = (
                    frequency
                    + self.k1
                    * (
                        1.0
                        - self.b
                        + self.b
                        * document_length
                        / average_length
                    )
                )
                score += (
                    idf
                    * (
                        frequency
                        * (self.k1 + 1.0)
                        / denominator
                    )
                )

            if score <= 0:
                continue

            searchable = self.searchable_text(node).lower()
            query_text = query.lower().strip()
            if query_text and query_text in searchable:
                score += 1.0

            scored.append(
                NodeWithScore(
                    node=node,
                    score=float(score),
                )
            )

        scored.sort(
            key=lambda item: item.score or 0.0,
            reverse=True,
        )
        return scored[:top_k]
