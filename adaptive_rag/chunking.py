import logging
import hashlib
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from llama_index.core import Document
from llama_index.core.node_parser import (
    CodeSplitter,
    MarkdownNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import NodeRelationship, TextNode, MetadataMode
from llama_index.core.utils import get_tokenizer

from .config import (
    CHARS_PER_TOKEN_CODE,
    CHUNK_OVERLAP_RATIO,
    CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_TIERS,
    CODE_LINE_DEFAULT,
    CODE_LINE_TIERS,
    EMBED_METADATA_KEYS,
    EMBEDDING_METADATA_RESERVE,
    EMBEDDING_MODEL,
    EMBEDDING_SPECIAL_TOKENS,
    FALLBACK_TOKEN_MARGIN,
    LLM_METADATA_KEYS,
    MAX_EMBEDDING_TOKENS,
    MIN_CHUNK_OVERLAP,
    MIN_CHUNK_TOKENS,
    SHORT_TEXT_MAX_WORDS, 
    )
from .documents import DocumentProfile, profile_document

logger = logging.getLogger(__name__)

_LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "cs": "c_sharp",
    "csharp": "c_sharp",
    "c#": "c_sharp",
    "c++": "cpp",
    "cc": "cpp",
    "h": "c",
    "sh": "bash",
    "shell": "bash",
    "rb": "ruby",
    "rs": "rust",
    "kt": "kotlin",
    "golang": "go",
}

# (text, extra_metadata)
Segment = Tuple[str, Dict]

def _pick(tiers, value, default):
    """First tier whose upper bound is above `value`, else `default`."""
    for upper_bound, result in tiers:
        if value < upper_bound:
            return result
    return default

def _load_encoder() -> Tuple[Callable[[str], list], bool]:
    """Return (encode_fn, is_exact).
 
    Preferred: the embedding model's own tokenizer, so "N tokens" here means
    exactly what it means to the embedder. Fallback: tiktoken (GPT), which
    undercounts, so the caller applies FALLBACK_TOKEN_MARGIN.
    """
    try:
        from transformers import AutoTokenizer
 
        hf_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        # We count long texts on purpose; silence the "longer than max" warning.
        hf_tokenizer.model_max_length = int(1e9)
 
        def encode(text: str) -> list:
            return hf_tokenizer.encode(text, add_special_tokens=False)
 
        return encode, True
    except Exception as error:
        logger.warning(
            "Could not load tokenizer for %s (%s). Falling back to the GPT "
            "tokenizer with a %.0f%% safety margin.",
            EMBEDDING_MODEL,
            error,
            FALLBACK_TOKEN_MARGIN * 100,
        )
        return get_tokenizer(), False

class AdaptiveChunker:

    def __init__(
        self,
        encode: Optional[Callable[[str], int]] = None):
        if encode is not None:
            exact = True
        else:
            encode, exact = _load_encoder()
 
        self._encode = encode
        self._count = lambda text: len(encode(text))
 
        usable = MAX_EMBEDDING_TOKENS - EMBEDDING_SPECIAL_TOKENS
        # Limit for the FULL embedded text (chunk text + embedded metadata).
        self.embed_limit = usable if exact else int(usable * FALLBACK_TOKEN_MARGIN)
        # Limit for the chunk text alone.
        self.max_tokens = max(64, self.embed_limit - EMBEDDING_METADATA_RESERVE)
 
        self._limit_splitter = self._sentence_splitter(self.max_tokens)

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    def process(self, documents: List[Document]) -> List[TextNode]:
        nodes: List[TextNode] = []

        for document in documents:
            # Empty parser outputs should never become embedded nodes.
            if not document.text or not document.text.strip():
                continue

            try:
                nodes.extend(self._process_one(document))
            except Exception:
                # One bad document must not kill the whole ingestion batch.
                logger.exception(
                    "Chunking failed for %s; skipping",
                    document.metadata.get("file_path", "unknown"),
                )

        return nodes

    def _process_one(self, document: Document) -> List[TextNode]:
        path = Path(document.metadata.get("file_path", "unknown"))

        profile = profile_document(path, extracted_text=document.text)

        # Copy instead of mutating the caller's Document.
        metadata = dict(document.metadata)
        metadata.update(
            {
                "word_count": profile.word_count,
                "line_count": profile.line_count,
                "has_headers": profile.has_headers,
                "has_tables": profile.has_tables,
                "has_code": profile.has_code,
                "structure_depth": profile.structure_depth,
            }
        )

        strategy = self.choose_strategy(profile)

        logger.info(
            "[PROFILE] %s | type=%s | words=%s | strategy=%s",
            path.name,
            profile.document_type,
            profile.word_count,
            strategy,
        )

        try:
            label, segments = self.segment_document(document, profile, strategy)
        except Exception:
            logger.warning(
                "Strategy '%s' failed for %s; falling back to text splitter",
                strategy,
                path.name,
                exc_info=True,
            )
            label = f"{strategy}_fallback"
            segments = self._text_segments(document.text, profile)

        if not segments:
            segments = [(document.text.strip(), {})]

        # Final guarantee: nothing larger than the embedder window leaves here,
        # regardless of which strategy produced it.
        segments = self._enforce_limit(segments)

        return self._build_nodes(document, metadata, segments, label)

    # ------------------------------------------------------------------ #
    # Strategy selection
    # ------------------------------------------------------------------ #

    def choose_strategy(self, profile: DocumentProfile) -> str:

        if profile.document_type == "code":
            return "code"

        if profile.document_type == "markdown":
            return "markdown"

        if profile.document_type == "document":
            if profile.has_headers or profile.has_tables:
                return "structured_document"
            return "document"

        if profile.document_type in {"json", "tabular", "spreadsheet"}:
            return "structured_data"

        if profile.document_type == "html":
            return "html"

        if profile.word_count <= SHORT_TEXT_MAX_WORDS:
            return "short_text"

        return "adaptive_text"

    def segment_document(
        self,
        document: Document,
        profile: DocumentProfile,
        strategy: str,
    ) -> Tuple[str, List[Segment]]:
        """Return (strategy_label, segments)."""

        text = document.text

        if strategy == "code":
            return self._code_segments(text, profile)

        if strategy in {"markdown", "structured_document"}:
            return "markdown_structure", self._markdown_segments(document)

        if strategy == "structured_data":
            return self._structured_segments(text, profile)

        if strategy == "short_text":
            return "minimal", [(text.strip(), {})]

        # "html", "document", "adaptive_text": the label reflects the selected
        # strategy so routing and observability agree.
        return strategy, self._text_segments(text, profile)

    # ------------------------------------------------------------------ #
    # Per-strategy segmenters
    # ------------------------------------------------------------------ #

    def _code_segments(
        self, text: str, profile: DocumentProfile
    ) -> Tuple[str, List[Segment]]:

        language = self._normalize_language(getattr(profile, "language", None))

        if language:
            try:
                chunk_lines = self.dynamic_code_lines(profile.line_count)
                splitter = CodeSplitter(
                    language=language,
                    chunk_lines=chunk_lines,
                    chunk_lines_overlap=min(15, chunk_lines // 4),
                    # Default is 1500 chars regardless of the embedder; tie it
                    # to the token budget instead.
                    max_chars=self.max_tokens * CHARS_PER_TOKEN_CODE,
                )
                chunks = splitter.split_text(text)
                return "code", [
                    (c, {"language": language}) for c in chunks if c.strip()
                ]
            except Exception as error:
                logger.warning("Code splitter failed (%s): %s", language, error)

        
        # Split on line boundaries; SentenceSplitter would cut code on '.' and ','.
        extra = {"language": language} if language else {}
        chunks = self._split_by_lines(text, self.max_tokens, overlap_lines=3)
        return "code_fallback", [(c, dict(extra)) for c in chunks]

    def _markdown_segments(self, document: Document) -> List[Segment]:

        parsed = MarkdownNodeParser().get_nodes_from_documents([document])

        segments: List[Segment] = []
        for node in parsed:
            content = node.get_content().strip()
            if not content:
                continue
            # Keep only what the parser added (header path), not the copy of
            # the document metadata it also propagates.
            extra = {
                k: v
                for k, v in (node.metadata or {}).items()
                if k not in document.metadata
            }
            # The header key name differs between LlamaIndex versions
            # ("header_path" / "Header_Path"). Normalize so the allowlist in
            # config only needs one spelling.
            for key in list(extra):
                if key.lower() == "header_path":
                    extra["header_path"] = extra.pop(key)
            segments.append((content, extra))

        return self._merge_small(segments)

    def _merge_small(self, segments: List[Segment]) -> List[Segment]:
        """Fold tiny sections (header-only, one-liners) into their successor."""

        merged: List[Segment] = []
        for text, meta in segments:
            if merged:
                prev_text, prev_meta = merged[-1]
                prev_tokens = self._count(prev_text)
                if (
                    prev_tokens < MIN_CHUNK_TOKENS
                    and prev_tokens + self._count(text) <= self.max_tokens
                ):
                    merged[-1] = (f"{prev_text}\n\n{text}", prev_meta)
                    continue
            merged.append((text, meta))

        return merged

    def _structured_segments(
        self, text: str, profile: DocumentProfile
    ) -> Tuple[str, List[Segment]]:

        budget = self.dynamic_chunk_size(profile.word_count)

        if self._count(text) <= budget:
            return "structured_data", [(text.strip(), {})]

        # SentenceSplitter breaks on commas/periods, which shreds CSV rows and
        # JSON. Split on line boundaries instead, and for tabular data repeat
        # the header row in every chunk so each chunk is self-describing.
        if profile.document_type == "tabular":
            lines = text.splitlines()
            header = lines[0] if lines else None
            body = "\n".join(lines[1:])
            chunks = self._split_by_lines(body, budget, header=header)
        else:
            chunks = self._split_by_lines(text, budget)

        return "structured_text", [(c, {}) for c in chunks]

    def _text_segments(self, text: str, profile: DocumentProfile) -> List[Segment]:

        size = self.dynamic_chunk_size(profile.word_count)
        splitter = _sentence_splitter(size)
        return [(c, {}) for c in splitter.split_text(text) if c.strip()]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

     def _sentence_splitter(self, size: int) -> SentenceSplitter:
        # Passing `tokenizer` makes the splitter count in the embedder's own
        # tokens instead of its default GPT tokenizer.
        return SentenceSplitter(
            chunk_size=size,
            chunk_overlap=self.dynamic_overlap(size),
            tokenizer=self._encode,
        )

    def _enforce_limit(self, segments: List[Segment]) -> List[Segment]:
        """Split any segment that exceeds the embedding window.
        """

        safe: List[Segment] = []
        for text, meta in segments:
            if not text.strip():
                continue
            if self._count(text) <= self.max_tokens:
                safe.append((text, meta))
                continue
            for piece in self._limit_splitter.split_text(text):
                if piece.strip():
                    safe.append((piece, meta))

        return safe

    def _split_by_lines(
        self,
        text: str,
        budget: int,
        header: Optional[str] = None,
        overlap_lines: int = 0,
    ) -> List[str]:

        header_cost = 0
        if header is not None:
            header_cost = self._count(header) + 1
            if header_cost > budget // 2:
                header, header_cost = None, 0

        prefix = f"{header}\n" if header else ""
        body_budget = max(budget - header_cost, 16)

        chunks: List[str] = []
        current: List[str] = []
        current_cost = 0

        def emit():
            body = "\n".join(current)
            if body.strip():
                chunks.append(prefix + body)

        for line in text.splitlines():
            cost = self._count(line) + 1

            if cost > body_budget:
                # A single line bigger than the budget (minified JSON, long
                # CSV row): last resort, split inside the line.
                emit()
                current.clear()
                current_cost = 0
                line_splitter = SentenceSplitter(
                    chunk_size=body_budget, chunk_overlap=0, tokenizer=self._encode,
                )
                for piece in line_splitter.split_text(line):
                    if piece.strip():
                        chunks.append(prefix + piece)
                continue

            if current and current_cost + cost > body_budget:
                emit()
                carry = current[-overlap_lines:] if overlap_lines else []
                carry_cost = sum(self._count(item) + 1 for item in carry)
                if carry_cost + cost > body_budget:
                    carry, carry_cost = [], 0
                current[:] = carry
                current_cost = carry_cost

            current.append(line)
            current_cost += cost

        emit()
        return chunks

    def _build_nodes(
        self,
        document: Document,
        metadata: Dict,
        segments: List[Segment],
        label: str,
    ) -> List[TextNode]:
        """Create nodes uniformly so every strategy gets the same guarantees:
        source link, prev/next links, stable ids, and metadata hygiene."""

        source = document.as_related_node_info()
        # document_id is set by attach_document_identity (hash of the relative
        # path) and stored as doc_id. part_index separates the several
        # Documents one file can produce (Excel sheets, PDF pages).
        document_id = document.doc_id
        part_index = int(metadata.get("part_index", 0) or 0)

        nodes: List[TextNode] = []
        total = len(segments)

        for index, (text, extra) in enumerate(segments):
            node_metadata = {
                **metadata,
                **extra,
                "chunk_strategy": label,
                "chunk_index": index,
                "chunk_total": total,
            }

            # Allowlists: only these keys are pasted into the embedded text /
            # LLM prompt. All keys remain stored on the node.
            embed_excluded = [k for k in node_metadata if k not in EMBED_METADATA_KEYS]
            llm_excluded = [k for k in node_metadata if k not in LLM_METADATA_KEYS]
            # Deterministic UUID: same file + part + position + content => same id.
            # (UUID format keeps stores like Qdrant happy.)
            content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
            node_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{document_id}|{part_index}|{index}|{content_hash}",
                )
            )
            node = TextNode(
                    id_=node_id,
                    text=text,
                    metadata=node_metadata,
                    excluded_embed_metadata_keys=embed_excluded,
                    excluded_llm_metadata_keys=llm_excluded,
                    relationships={NodeRelationship.SOURCE: source},
                )
            

            # Safety net: if text + embedded metadata still exceeds the model
            # window (very long header_path or file_name), stop embedding the
            # metadata for this node rather than let the text get truncated.
            embedded = node.get_content(metadata_mode=MetadataMode.EMBED)
            if self._count(embedded) > self.embed_limit:
                node.excluded_embed_metadata_keys = list(node_metadata.keys())
 
            nodes.append(node)

        for prev, nxt in zip(nodes, nodes[1:]):
            prev.relationships[NodeRelationship.NEXT] = nxt.as_related_node_info()
            nxt.relationships[NodeRelationship.PREVIOUS] = prev.as_related_node_info()

        return nodes

    @staticmethod
    def _normalize_language(language: Optional[str]) -> Optional[str]:
        if not language:
            return None
        key = str(language).lower().strip().lstrip(".")
        return _LANGUAGE_ALIASES.get(key, key)

    # ------------------------------------------------------------------ #
    # Size heuristics
    # ------------------------------------------------------------------ #

    def dynamic_code_lines(self, line_count: int) -> int:

        return _pick(CODE_LINE_TIERS, line_count, CODE_LINE_DEFAULT)

    def dynamic_chunk_size(self, word_count: int) -> int:
        """Target chunk size in *tokens*, capped to the chunk budget."""

        target = _pick(CHUNK_SIZE_TIERS, word_count, CHUNK_SIZE_DEFAULT)
        return min(target, self.max_tokens)

    def dynamic_overlap(self, chunk_size: int) -> int:
        overlap = max(MIN_CHUNK_OVERLAP, int(chunk_size * CHUNK_OVERLAP_RATIO))
        return min(overlap, chunk_size // 2)