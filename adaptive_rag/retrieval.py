import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from llama_index.core.schema import NodeWithScore

_WORD_RE = re.compile(r"\w+", re.UNICODE)

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "what", "when", "where",
    "who", "why", "how", "did", "does", "do", "and", "or", "to", "of", "in",
    "on", "for", "from", "with", "about", "this", "that", "these", "those",
    "it", "its", "be", "by", "as", "at", "which", "than", "can", "could",
    "should", "would", "will", "me", "my", "you", "your", "tell", "give",
    "please", "there", "their", "them", "has", "have", "had",
})

# Metadata worth matching lexically. Deliberately NOT included:
#   file_path       absolute path pieces ("users", "home", "data") would match
#                   unrelated queries for every chunk of every file
#   document_type / chunk_strategy
#                   internal labels; a query containing the word "code" or
#                   "markdown" would boost every chunk carrying that label
SEARCHABLE_METADATA_KEYS = ("file_name", "sheet_name", "header_path", "language")

# Re-rank this many times top_k candidates with the exact-phrase boost.
PHRASE_BOOST_POOL_FACTOR = 3


@dataclass
class QueryProfile:
    """Lightweight query analysis used to select an adaptive retrieval path."""

    mode: str
    reason: str
    needs_exact_match: bool = False
    is_broad: bool = False
    is_complex: bool = False


@dataclass
class _IndexData:
    """Everything retrieve() needs, swapped in as ONE object after a rebuild."""

    nodes: list
    document_lengths: List[int]
    postings: Dict[str, List[Tuple[int, int]]]  # term -> [(node_index, term_freq)]
    average_document_length: float


class LexicalIndex:
    """
    Small dependency-free BM25-style lexical index.

    It indexes node text plus useful metadata (file name, sheet, section path,
    language) so exact identifiers, filenames, error codes, SKUs and similar
    lexical signals can be retrieved even when dense similarity is not the
    strongest signal.

    Scores are raw BM25 values (unbounded). They are NOT comparable with
    dense cosine scores, so merge the two result lists by rank (for example
    reciprocal rank fusion) rather than by adding scores.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._data = _IndexData([], [], {}, 0.0)
        # Kept for callers that inspect them.
        self.nodes = []
        self.built = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercase word tokens; snake_case identifiers also emit their parts.

        "q3_report.xlsx" -> q3_report, q3, report, xlsx
        so both "q3_report" (exact identifier) and "q3 report" (words) match.
        The same function is used for indexing and for queries.
        """
        tokens: List[str] = []
        for token in _WORD_RE.findall(text.lower()):
            tokens.append(token)
            if "_" in token:
                parts = [part for part in token.split("_") if part]
                if len(parts) > 1:
                    tokens.extend(parts)
        return tokens

    @staticmethod
    def searchable_text(node) -> str:
        metadata = getattr(node, "metadata", {}) or {}
        metadata_parts = [
            str(metadata[key])
            for key in SEARCHABLE_METADATA_KEYS
            if metadata.get(key)
        ]
        return " ".join([node.get_content()] + metadata_parts)

    def build(self, nodes: List) -> None:
        # Build into locals and publish once at the end. Writing into self.*
        # while a query is running lets retrieve() see new
        # nodes with half-built statistics and raise IndexError.
        nodes = list(nodes)
        postings: Dict[str, List[Tuple[int, int]]] = {}
        document_lengths: List[int] = []

        for index, node in enumerate(nodes):
            terms = self.tokenize(self.searchable_text(node))
            document_lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                postings.setdefault(term, []).append((index, frequency))

        average = sum(document_lengths) / len(nodes) if nodes else 0.0

        self._data = _IndexData(nodes, document_lengths, postings, average)
        self.nodes = nodes
        self.built = True

    def retrieve(self, query: str, top_k: int = 8) -> List[NodeWithScore]:
        if top_k <= 0:
            return []

        data = self._data  # one consistent snapshot for the whole query
        if not data.nodes:
            return []

        query_terms = self.tokenize(query)
        # Words like "what", "does", "do" occur in almost every chunk and made
        # unrelated documents match. Keep them only if nothing else is left.
        query_terms = [t for t in query_terms if t not in STOP_WORDS] or query_terms
        if not query_terms:
            return []

        total_documents = len(data.nodes)
        average_length = data.average_document_length or 1.0

        # Only nodes that contain at least one query term are ever touched.
        scores: Dict[int, float] = {}
        for term in set(query_terms):
            postings = data.postings.get(term)
            if not postings:
                continue

            document_frequency = len(postings)
            idf = math.log(
                1.0
                + (total_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

            for index, frequency in postings:
                length = data.document_lengths[index] or 1
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / average_length
                )
                scores[index] = scores.get(index, 0.0) + (
                    idf * frequency * (self.k1 + 1.0) / denominator
                )

        if not scores:
            return []

        # Deterministic order: score desc, then node position.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        pool = ranked[: top_k * PHRASE_BOOST_POOL_FACTOR]

        # Exact-phrase boost, only for multi-term queries and only for the
        # candidate pool (it needs the node text, which is expensive to build).
        query_text = query.lower().strip()
        use_phrase_boost = bool(query_text) and len(query_terms) > 1

        results = []
        for index, score in pool:
            node = data.nodes[index]
            if use_phrase_boost and query_text in self.searchable_text(node).lower():
                score += 1.0
            results.append(NodeWithScore(node=node, score=float(score)))

        results.sort(key=lambda item: item.score or 0.0, reverse=True)
        return results[:top_k]