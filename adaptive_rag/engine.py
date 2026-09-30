import json
import re
import shutil
import time
import uuid
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import qdrant_client
from json_repair import repair_json
from llama_index.core import (
    Settings, SummaryIndex, StorageContext, VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, NodeWithScore
from llama_index.vector_stores.qdrant import QdrantVectorStore

from .chunking import AdaptiveChunker
from .config import STORAGE_DIR
from .documents import (
    calculate_document_id, calculate_file_hash, discover_files, load_manifest,
    load_single_file, save_manifest,
)
from .observability import API_TRACKER
from .models import configure_models
from .retrieval import LexicalIndex, QueryProfile

class AdaptiveRAG:

    VECTOR_INDEX_ID = "vector_idx"

    SUMMARY_INDEX_ID = "summary_idx"

    def __init__(
        self,
        data_dir: str = "./data",
    ):

        configure_models()

        self.data_dir = Path(
            data_dir
        )

        self.manifest = (
            load_manifest()
        )

        self.chunker = (
            AdaptiveChunker()
        )

        self.engine = None

        self.vector_index = None

        self.summary_index = None

        self.db_client = qdrant_client.QdrantClient(url="http://localhost:6333")

        # Adaptive retrieval configuration.
        self.lexical_index = LexicalIndex()
        self.retrieval_top_k = 5
        self.candidate_top_k = 12
        self.default_rag_mode = "auto"

        # This can skip a repeated vector/lexical search while remaining
        # bounded and invalidated whenever the knowledge base changes.
        self._retrieval_cache = OrderedDict()
        self._retrieval_cache_size = 32
        self._summary_query_engine = None

    def _invalidate_query_caches(self):
        """Invalidate query objects whenever the underlying index changes."""
        self._retrieval_cache.clear()
        self._summary_query_engine = None

    def indexes_exist(self) -> bool:

        try:
            collections = self.db_client.get_collections().collections
            exists = any(c.name == "pipeline_collection" for c in collections)
            return exists and (STORAGE_DIR / "docstore.json").exists()
        except Exception:
            return False

    def load_indexes(self):

        if not self.indexes_exist():
            return False

        print(
            "--> [LOAD] Connecting to existing Local VectorDB..."
        )

        vector_store = QdrantVectorStore(
            client=self.db_client, 
            collection_name="pipeline_collection"
        )

        storage_context = (
            StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(
                    STORAGE_DIR
                )
            )
        )

        self.vector_index = (
            load_index_from_storage(
                storage_context,
                index_id=self.VECTOR_INDEX_ID,
            )
        )

        self.summary_index = (
            load_index_from_storage(
                storage_context,
                index_id=self.SUMMARY_INDEX_ID,
            )
        )

        self._invalidate_query_caches()

        return True

    def persist_indexes(self):
        """Persist the shared StorageContext once.

        VectorIndex and SummaryIndex intentionally share the same
        StorageContext, so persisting both independently duplicates work.
        """
        if self.vector_index is None and self.summary_index is None:
            return

        storage_context = None
        if self.vector_index is not None:
            storage_context = self.vector_index.storage_context
        elif self.summary_index is not None:
            storage_context = self.summary_index.storage_context

        if storage_context is None:
            return

        print("--> [PERSIST] Saving index state...")
        persist_start = time.perf_counter()
        storage_context.persist(
            persist_dir=str(STORAGE_DIR)
        )
        API_TRACKER.record_persist(time.perf_counter() - persist_start)
        print("--> [PERSIST] Index state saved.")

    def create_indexes(
        self,
        nodes,
    ):

        print(
            "--> [INDEX] Creating production VectorDB collection..."
        )

        index_start = time.perf_counter()
        embedding_before = API_TRACKER.embedding_snapshot()

        vector_store = QdrantVectorStore(
            client=self.db_client,
            collection_name="pipeline_collection"
        )

        storage_context = (
            StorageContext.from_defaults(vector_store=vector_store)
        )

        self.vector_index = (
            VectorStoreIndex(
                nodes,
                storage_context=storage_context,
            )
        )

        self.vector_index.set_index_id(
            self.VECTOR_INDEX_ID
        )

        self.summary_index = (
            SummaryIndex(
                nodes,
                storage_context=storage_context,
            )
        )

        self.summary_index.set_index_id(
            self.SUMMARY_INDEX_ID
        )

        self._invalidate_query_caches()
        self.persist_indexes()

        API_TRACKER.record_indexing(
            time.perf_counter() - index_start,
            len(nodes),
        )
        API_TRACKER.record_embedding_delta(embedding_before)
        self.rebuild_lexical_index()

    def delete_document_from_indexes(
        self,
        document_id: str,
        persist: bool = True,
        rebuild_lexical: bool = True,
    ):

        print(
            f"--> [DELETE INDEX] "
            f"Removing document "
            f"{document_id[:12]}..."
        )

        if self.vector_index is None:
            raise RuntimeError(
            "Vector index is not loaded."
        )

        try:

            self.vector_index.delete_ref_doc(
                ref_doc_id=document_id,
                delete_from_docstore=True,
            )

            print(
                "--> [DELETE INDEX] "
                "Vector index cleaned."
            )

        except Exception as error:
            raise RuntimeError(
                f"Vector index deletion failed for "
                f"{document_id}: {error}"
            ) from error


        if self.summary_index:

            try:

                self.summary_index.delete_ref_doc(
                    ref_doc_id=document_id,
                    delete_from_docstore=True,
                )

                print(
                    "--> [DELETE INDEX] "
                    "Summary index cleaned."
                )

            except Exception as error:
                raise RuntimeError(
                    f"Summary index deletion failed for "
                    f"{document_id}: {error}"
                ) from error
            
        self._invalidate_query_caches()
        if persist:
            self.persist_indexes()
        if rebuild_lexical:
            self.rebuild_lexical_index()


    def ingest_file(
        self,
        path: Path,
        relative_path: str,
        document_id: str,
        file_hash: str,
    ):

        print(
            f"--> [INGEST] {relative_path}"
        )

        documents = load_single_file(
            path=path,
            document_id=document_id,
            file_hash=file_hash,
        )

        if not documents:

            print(
                "--> [WARNING] "
                f"No readable content: "
                f"{relative_path}"
            )

            return []

        chunk_start = time.perf_counter()
        nodes = self.chunker.process(
            documents
        )

        for node in nodes:
            node.id_ = str(uuid.uuid4())
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=document_id)

        chunk_seconds = time.perf_counter() - chunk_start
        API_TRACKER.record_chunking(chunk_seconds)

        print(
            f"--> [INGEST] "
            f"{relative_path} → "
            f"{len(nodes)} nodes"
        )

        return nodes

    def sync(
        self,
        force_rebuild: bool = False,
    ):

        sync_start = time.perf_counter()
        API_TRACKER.sync_had_ingestion = False
        sync_had_mutations = False

        print()
        print("=" * 70)
        print("              ADAPTIVE RAG SYNC")
        print("=" * 70)

        if force_rebuild:

            print(
                "--> [REBUILD] "
                "Complete rebuild requested."
            )

            self.clear_storage()

            self.manifest = {}

            self.vector_index = None

            self.summary_index = None

        else:

            self.load_indexes()


        files = discover_files(
            self.data_dir
        )

        current_files = {}

        for path in files:

            relative_path = str(
                path.relative_to(
                    self.data_dir
                )
            )

            document_id = (
                calculate_document_id(
                    relative_path
                )
            )

            file_hash = (
                calculate_file_hash(
                    path
                )
            )

            current_files[
                relative_path
            ] = {
                "document_id": document_id,
                "hash": file_hash,
                "extension": path.suffix.lower(),
                "size": path.stat().st_size,
            }

        if not self.indexes_exist():

            print(
                "--> [INIT] "
                "No indexes found. "
                "Building initial RAG."
            )

            all_nodes = []

            for path in files:

                relative_path = str(
                    path.relative_to(
                        self.data_dir
                    )
                )

                info = current_files[
                    relative_path
                ]

                API_TRACKER.sync_had_ingestion = True
                nodes = self.ingest_file(
                    path=path,
                    relative_path=relative_path,
                    document_id=info[
                        "document_id"
                    ],
                    file_hash=info[
                        "hash"
                    ],
                )

                all_nodes.extend(
                    nodes
                )

            if all_nodes:

                self.create_indexes(
                    all_nodes
                )

            self.manifest = (
                current_files
            )

            save_manifest(
                self.manifest
            )

            print(
                "--> [DONE] "
                "Initial RAG created."
            )

            self.build_router()
            API_TRACKER.record_sync(time.perf_counter() - sync_start)

            return

        old_paths = set(
            self.manifest.keys()
        )

        current_paths = set(
            current_files.keys()
        )

        deleted_paths = (
            old_paths - current_paths
        )

        for relative_path in deleted_paths:

            old_record = (
                self.manifest[
                    relative_path
                ]
            )

            document_id = (
                old_record[
                    "document_id"
                ]
            )

            print(
                f"--> [DELETED FILE] "
                f"{relative_path}"
            )

            self.delete_document_from_indexes(
                document_id,
                persist=False,
                rebuild_lexical=False,
            )
            sync_had_mutations = True

            del self.manifest[
                relative_path
            ]

        for relative_path, info in (
            current_files.items()
        ):

            old_record = (
                self.manifest.get(
                    relative_path
                )
            )

            path = (
                self.data_dir
                / relative_path
            )

            if old_record is None:

                print(
                    f"--> [NEW FILE] "
                    f"{relative_path}"
                )

                API_TRACKER.sync_had_ingestion = True
                nodes = self.ingest_file(
                    path=path,
                    relative_path=relative_path,
                    document_id=info[
                        "document_id"
                    ],
                    file_hash=info[
                        "hash"
                    ],
                )

                self.insert_nodes(
                    nodes,
                    persist=False,
                    rebuild_lexical=False,
                )
                sync_had_mutations = True

                self.manifest[
                    relative_path
                ] = info

                continue

            old_hash = (
                old_record.get(
                    "hash"
                )
            )

            new_hash = (
                info["hash"]
            )

            if old_hash != new_hash:

                print(
                    f"--> [CHANGED FILE] "
                    f"{relative_path}"
                )

                document_id = (
                    old_record[
                        "document_id"
                    ]
                )

                self.delete_document_from_indexes(
                    document_id,
                    persist=False,
                    rebuild_lexical=False,
                )
                sync_had_mutations = True

                API_TRACKER.sync_had_ingestion = True
                nodes = self.ingest_file(
                    path=path,
                    relative_path=relative_path,
                    document_id=document_id,
                    file_hash=new_hash,
                )

                self.insert_nodes(
                    nodes,
                    persist=False,
                    rebuild_lexical=False,
                )
                sync_had_mutations = True

                self.manifest[
                    relative_path
                ] = {
                    **info,
                    "document_id": document_id,
                }

                continue

            print(
                f"--> [UNCHANGED] "
                f"{relative_path}"
            )

        save_manifest(
            self.manifest
        )

        if sync_had_mutations:
            self.persist_indexes()
            self.rebuild_lexical_index()
            self.build_router()
        else:
            # No index mutation means cached query objects remain valid.
            self.build_router()

        print()
        print(
            "--> [DONE] "
            "RAG synchronized successfully."
        )

        API_TRACKER.record_sync(time.perf_counter() - sync_start)

    def insert_nodes(
        self,
        nodes,
        persist: bool = True,
        rebuild_lexical: bool = True,
    ):

        if not nodes:
            return

        if self.vector_index is None:

            print(
                "--> [INDEX] "
                "Vector index doesn't exist."
            )

            self.create_indexes(
                nodes
            )

            return

        print(
            f"--> [INDEX] "
            f"Inserting {len(nodes)} nodes..."
        )

        try:

            index_start = time.perf_counter()
            embedding_before = API_TRACKER.embedding_snapshot()

            self.vector_index.insert_nodes(
                nodes
            )

            if self.summary_index is not None:
                self.summary_index.insert_nodes(
                    nodes
                )

            self._invalidate_query_caches()
            if persist:
                self.persist_indexes()
            API_TRACKER.record_indexing(
            time.perf_counter() - index_start,
            len(nodes),
            )
            API_TRACKER.record_embedding_delta(embedding_before)
            if rebuild_lexical:
                self.rebuild_lexical_index()

        except Exception as error:

            raise RuntimeError(
                f"Failed to insert nodes into indexes: "
                f"{error}"
            ) from error

    def rebuild_lexical_index(self):
        """Rebuild the local lexical index from the current node docstore."""
        rebuild_start = time.perf_counter()

        if self.vector_index is None:
            self.lexical_index = LexicalIndex()
            API_TRACKER.record_lexical_rebuild(
                time.perf_counter() - rebuild_start
            )
            return

        nodes = list(self.vector_index.docstore.docs.values())
        self.lexical_index.build(nodes)
        API_TRACKER.record_lexical_rebuild(
            time.perf_counter() - rebuild_start
        )

        print(
            f"--> [RETRIEVAL] Lexical index rebuilt: "
            f"{len(nodes)} nodes"
        )

    def profile_query(
        self,
        question: str,
        rag_mode: str = "auto",
    ) -> QueryProfile:
        """
        Select a retrieval strategy.

        User-selected modes are honored explicitly. In auto mode the system
        uses lightweight lexical signals to choose between semantic, keyword,
        hybrid, and summary retrieval.
        """
        requested_mode = (rag_mode or self.default_rag_mode).strip().lower()

        aliases = {
            "vector": "semantic",
            "dense": "semantic",
            "bm25": "keyword",
            "lexical": "keyword",
            "hybrid_search": "hybrid",
            "global": "summary",
        }
        requested_mode = aliases.get(
            requested_mode,
            requested_mode,
        )

        valid_modes = {
            "auto",
            "semantic",
            "keyword",
            "hybrid",
            "summary",
        }
        if requested_mode not in valid_modes:
            raise ValueError(
                f"Unsupported RAG mode '{rag_mode}'. "
                f"Choose from: {', '.join(sorted(valid_modes))}."
            )

        if requested_mode != "auto":
            return QueryProfile(
                mode=requested_mode,
                reason="user selected",
            )

        normalized = question.lower().strip()

        exact_patterns = [
            r"`[^`]+`",
            r"\b(error|exception|traceback|id|code|filename|"
            r"function|method|variable|class|field|column|sku|part)\b",
            r"\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b",
            r"\b[a-zA-Z0-9]+-[a-zA-Z0-9-]+\b",
        ]
        needs_exact_match = any(
            re.search(pattern, question, flags=re.IGNORECASE)
            for pattern in exact_patterns
        ) or len(re.findall(r"\b\d+\b", question)) >= 2

        broad_terms = {
            "summarize",
            "summary",
            "overview",
            "overall",
            "main themes",
            "key themes",
            "entire document",
            "whole document",
            "all documents",
            "big picture",
        }
        is_broad = any(term in normalized for term in broad_terms)

        complex_terms = {
            "compare",
            "comparison",
            "contrast",
            "difference",
            "differences",
            "versus",
            "vs",
            "across",
            "multiple",
            "both",
            "and explain",
            "why and how",
        }
        is_complex = any(
            term in normalized
            for term in complex_terms
        ) or question.count("?") > 1

        if is_broad:
            return QueryProfile(
                mode="summary",
                reason="broad/global question",
                needs_exact_match=needs_exact_match,
                is_broad=True,
                is_complex=is_complex,
            )

        if needs_exact_match and is_complex:
            return QueryProfile(
                mode="hybrid",
                reason="exact-match signals + complex comparison",
                needs_exact_match=True,
                is_complex=True,
            )

        if needs_exact_match:
            return QueryProfile(
                mode="keyword",
                reason="exact identifier/fact signals",
                needs_exact_match=True,
            )

        if is_complex:
            return QueryProfile(
                mode="hybrid",
                reason="multi-part/comparison question",
                is_complex=True,
            )

        return QueryProfile(
            mode="semantic",
            reason="conceptual/semantic question",
        )

    def _vector_retrieve(
        self,
        question: str,
        top_k: int,
    ) -> List[NodeWithScore]:
        if self.vector_index is None:
            return []

        # Retriever construction is cheap. Cache the retrieval results below,
        # because that is the operation that actually avoids a repeated search.
        retriever = self.vector_index.as_retriever(
            similarity_top_k=top_k
        )
        return retriever.retrieve(question)

    @staticmethod
    def _normalize_cache_question(question: str) -> str:
        return " ".join(question.lower().split())

    def _retrieval_cache_key(self, question: str, profile: QueryProfile):
        return (
            self._normalize_cache_question(question),
            profile.mode,
            self.candidate_top_k,
            self.retrieval_top_k,
        )

    def _get_cached_retrieval(self, key):
        cached = self._retrieval_cache.get(key)
        if cached is None:
            API_TRACKER.record_retriever_cache(False)
            return None

        self._retrieval_cache.move_to_end(key)
        API_TRACKER.record_retriever_cache(True)
        return [
            NodeWithScore(node=item.node, score=item.score)
            for item in cached
        ]

    def _cache_retrieval(self, key, nodes):
        self._retrieval_cache[key] = [
            NodeWithScore(node=item.node, score=item.score)
            for item in nodes
        ]
        self._retrieval_cache.move_to_end(key)
        while len(self._retrieval_cache) > self._retrieval_cache_size:
            self._retrieval_cache.popitem(last=False)

    def _rerank(
        self,
        question: str,
        candidates: List[NodeWithScore],
        top_k: int,
    ) -> List[NodeWithScore]:
        """
        Lightweight local reranker.

        It preserves the retriever score while rewarding query-term coverage
        and exact phrase matches. A cross-encoder can be evaluated later if testing shows it is needed.
        """
        query_terms = set(
            LexicalIndex.tokenize(question)
        )
        query_phrase = question.lower().strip()

        if not candidates:
            return []

        source_scores = [
            float(candidate.score or 0.0)
            for candidate in candidates
        ]
        max_score = max(source_scores) or 1.0

        reranked = []
        for candidate in candidates:
            content = candidate.node.get_content().lower()
            candidate_terms = set(
                LexicalIndex.tokenize(content)
            )

            coverage = (
                len(query_terms & candidate_terms)
                / len(query_terms)
                if query_terms
                else 0.0
            )
            exact_phrase = (
                1.0
                if query_phrase and query_phrase in content
                else 0.0
            )
            normalized_source = (
                float(candidate.score or 0.0) / max_score
            )

            final_score = (
                0.55 * normalized_source
                + 0.35 * coverage
                + 0.10 * exact_phrase
            )

            reranked.append(
                NodeWithScore(
                    node=candidate.node,
                    score=final_score,
                )
            )

        reranked.sort(
            key=lambda item: item.score or 0.0,
            reverse=True,
        )
        return reranked[:top_k]

    def _merge_hybrid(
        self,
        vector_results: List[NodeWithScore],
        keyword_results: List[NodeWithScore],
        top_k: int,
    ) -> List[NodeWithScore]:
        """Fuse dense and lexical candidates using reciprocal rank fusion."""
        fused_scores = defaultdict(float)
        nodes_by_id = {}

        for results in (vector_results, keyword_results):
            for rank, result in enumerate(results, start=1):
                node_id = result.node.node_id
                nodes_by_id[node_id] = result.node
                fused_scores[node_id] += 1.0 / (60.0 + rank)

        fused = [
            NodeWithScore(
                node=nodes_by_id[node_id],
                score=score,
            )
            for node_id, score in fused_scores.items()
        ]
        fused.sort(
            key=lambda item: item.score or 0.0,
            reverse=True,
        )
        return fused[:top_k]

    def _assess_context_boundary(
        self,
        question: str,
        retrieved_nodes: List[NodeWithScore],
    ) -> Tuple[bool, str]:
        """
        Determine whether the selected nodes appear likely to need adjacent
        chunks.

        This does NOT inspect words such as 'before' or 'after' in the query.
        The decision is based on the actual retrieved context.

        Expansion is considered useful when:
        - relevant query terms occur very close to a node boundary, or
        - the selected node looks structurally incomplete, or
        - the selected context is unusually small for the query.

        The method remains local and does not create another Gemini call.
        """

        if not retrieved_nodes:
            return False, "no retrieved nodes"

        # Normalize meaningful query terms.
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were",
            "what", "when", "where", "who", "why", "how",
            "did", "does", "do", "and", "or", "to", "of",
            "in", "on", "for", "from", "with", "about",
            "this", "that", "these", "those", "it", "its",
            "be", "by", "as", "at", "which", "than",
        }

        query_terms = [
            term
            for term in re.findall(
                r"[A-Za-z0-9_]+",
                question.lower(),
            )
            if len(term) >= 3 and term not in stop_words
        ]

        if not query_terms:
            return False, "no meaningful query terms"

        boundary_hits = []

        for item in retrieved_nodes:
            text = item.node.get_content().strip()

            if not text:
                continue

            normalized_text = text.lower()
            text_length = len(normalized_text)

            # Only examine terms that actually occur in this selected node.
            relevant_positions = []

            for term in query_terms:
                position = normalized_text.find(term)

                if position >= 0:
                    relevant_positions.append(
                        position / max(text_length, 1)
                    )

            if not relevant_positions:
                continue

            # A relevant term very near either edge suggests that the answer
            # may continue into the adjacent chunk.
            near_start = any(position <= 0.15 for position in relevant_positions)
            near_end = any(position >= 0.85 for position in relevant_positions)

            if near_start or near_end:
                boundary_hits.append(
                    {
                        "node": item.node,
                        "near_start": near_start,
                        "near_end": near_end,
                    }
                )

        if boundary_hits:
            # Only expand if the corresponding neighboring node actually exists.
            docstore_docs = self.vector_index.docstore.docs

            for hit in boundary_hits:
                node = hit["node"]

                if hit["near_start"]:
                    related = node.relationships.get(
                        NodeRelationship.PREVIOUS
                    )

                    if related and related.node_id in docstore_docs:
                        return True, "relevant content begins near chunk boundary"

                if hit["near_end"]:
                    related = node.relationships.get(
                        NodeRelationship.NEXT
                    )

                    if related and related.node_id in docstore_docs:
                        return True, "relevant content ends near chunk boundary"

        # Structural incompleteness is a secondary signal.
        for item in retrieved_nodes:
            text = item.node.get_content().strip()

            if not text:
                continue

            # These are only weak signals. We require them to occur together
            # with an actual neighboring node.
            looks_incomplete = (
                text.endswith("...")
                or text.endswith(":")
                or text.endswith(";")
                or text.endswith(",")
            )

            if not looks_incomplete:
                continue

            has_neighbor = (
                (
                    item.node.relationships.get(NodeRelationship.PREVIOUS)
                    and item.node.relationships[
                        NodeRelationship.PREVIOUS
                    ].node_id in self.vector_index.docstore.docs
                )
                or
                (
                    item.node.relationships.get(NodeRelationship.NEXT)
                    and item.node.relationships[
                        NodeRelationship.NEXT
                    ].node_id in self.vector_index.docstore.docs
                )
            )

            if has_neighbor:
                return True, "retrieved node appears structurally incomplete"

        return False, "selected context appears self-contained"    

    def retrieve_adaptively(
        self,
        question: str,
        profile: QueryProfile,
    ):
        """Run only the retrieval path selected for the current query."""
        if profile.mode == "summary":
            return None, "summary"

        cache_key = self._retrieval_cache_key(question, profile)
        cached = self._get_cached_retrieval(cache_key)
        if cached is not None:
            print("[RETRIEVAL CACHE] HIT | reused previous retrieval results")
            return cached, profile.mode

        print("[RETRIEVAL CACHE] MISS | running retrieval")

        if profile.mode == "semantic":
            candidates = self._vector_retrieve(
                question,
                self.candidate_top_k,
            )

        elif profile.mode == "keyword":
            candidates = self.lexical_index.retrieve(
                question,
                self.candidate_top_k,
            )

        elif profile.mode == "hybrid":
            vector_results = self._vector_retrieve(
                question,
                self.candidate_top_k,
            )
            keyword_results = self.lexical_index.retrieve(
                question,
                self.candidate_top_k,
            )
            candidates = self._merge_hybrid(
                vector_results,
                keyword_results,
                self.candidate_top_k,
            )

        else:
            raise ValueError(
                f"Unsupported retrieval mode: {profile.mode}"
            )

        reranked = self._rerank(
            question,
            candidates,
            self.retrieval_top_k,
        )

        should_expand, reason = self._assess_context_boundary(
            question,
            reranked,
        )

        if not should_expand:
            print(
                f"[CONTEXT EXPANSION] SKIPPED | {reason}"
            )
            self._cache_retrieval(cache_key, reranked)
            return reranked, profile.mode

        print(
            f"[CONTEXT EXPANSION] CHECK PASSED | {reason}"
        )

        expanded = []
        seen_ids = set()
        missing_neighbors = 0

        docstore_docs = self.vector_index.docstore.docs

        for node_with_score in reranked:

            current_id = node_with_score.node.node_id

            if current_id not in seen_ids:
                seen_ids.add(current_id)
                expanded.append(node_with_score)

            for relationship in (
                NodeRelationship.PREVIOUS,
                NodeRelationship.NEXT,
            ):
                related = node_with_score.node.relationships.get(
                    relationship
                )

                if related is None:
                    continue

                neighbor_id = related.node_id

                # Safe lookup. A stale/missing relationship can never
                # crash retrieval.
                neighbor = docstore_docs.get(neighbor_id)

                if neighbor is None:
                    missing_neighbors += 1
                    continue

                if neighbor_id in seen_ids:
                    continue

                seen_ids.add(neighbor_id)

                expanded.append(
                    NodeWithScore(
                        node=neighbor,
                        score=node_with_score.score,
                    )
                )

        print(
            f"[CONTEXT EXPANSION] USED | "
            f"{len(reranked)} → {len(expanded)} nodes"
        )

        if missing_neighbors:
            print(
                f"[CONTEXT EXPANSION] "
                f"Skipped {missing_neighbors} missing neighbor references"
            )

        self._cache_retrieval(cache_key, expanded)
        return expanded, profile.mode

    def synthesize_answer(
        self,
        question: str,
        retrieved_nodes: List[NodeWithScore],
    ) -> str:
        """Generate an answer strictly from the selected retrieval context."""
        if not retrieved_nodes:
            return (
                "The provided documentation does not contain "
                "enough information to answer this question."
            )

        contexts = []
        for index, item in enumerate(retrieved_nodes, start=1):
            node = item.node
            metadata = getattr(node, "metadata", {}) or {}
            source = (
                metadata.get("file_name")
                or metadata.get("file_path")
                or "unknown source"
            )
            contexts.append(
                f"[Context {index} | Source: {source}]\n"
                f"{node.get_content()}"
            )

        unified_context = "\n\n---\n\n".join(contexts)

        prompt = f"""
            You are a retrieval-augmented assistant.

            Answer the user's question using ONLY the supplied CONTEXT.
            Do not use outside knowledge.
            Do not invent facts, values, identifiers, filenames, or relationships.
            If the context does not contain enough information, say:
            "The provided documentation does not contain this information."

            QUESTION:
            {question}

            CONTEXT:
            \"\"\"{unified_context}\"\"\"

            Provide a concise, factual answer.
            """

        return Settings.llm.complete(prompt).text.strip()

    def build_router(self):
        """
        Prepare retrieval state.

        Query profiling  selects semantic, keyword,
        hybrid, or summary retrieval explicitly or automatically.
        """
        if self.vector_index is None:
            if not self.load_indexes():
                return None

        if not self.lexical_index.built:
            self.rebuild_lexical_index()

        if self.summary_index is not None:
            if self._summary_query_engine is None:
                self._summary_query_engine = self.summary_index.as_query_engine(
                    response_mode="tree_summarize",
                )
            self.engine = self._summary_query_engine
        else:
            self.engine = None

        return self.engine

    def evaluate_rag_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> Dict:
        """
        Evaluate whether a generated answer is grounded in retrieved evidence.

        Uses a single structured LLM judge for three complementary
        checks: context relevance, answer faithfulness, and completeness.
        The evaluator fails closed if the judge cannot return a valid result.
        """
        if not contexts:
            API_TRACKER.record_evaluation(failed=True)
            return {
                "relevant": False,
                "faithful": False,
                "complete": False,
                "verdict": "FAIL",
                "evaluation_error": False,
                "issues": ["No retrieval context was available."],
            }

        unified_context = "\n---\n".join(contexts)

        eval_prompt = f"""
            You are a strict RAG quality evaluator.
            Evaluate the ANSWER only against the supplied CONTEXT and the QUESTION.
            Do not use outside knowledge.

            Evaluate three dimensions:
            1. relevant: Does the CONTEXT contain evidence needed to answer the question?
            2. faithful: Is every factual claim in the ANSWER directly supported by the CONTEXT?
            3. complete: Does the ANSWER address all material parts of the QUESTION that the CONTEXT supports?

            Important:
            - Do not require the exact wording of the answer to appear in the context.
            - Simple arithmetic is allowed only when every input value is explicitly present in the context.
            - Do not treat plausible inference, outside knowledge, or unstated assumptions as supported.
            - If the context does not contain enough evidence, relevant should be false.
            - If the answer correctly says the documentation does not contain the requested information when the context lacks it, faithful and complete may be true, but relevant remains false.

            Return ONLY valid JSON matching exactly this schema:
            {{
            "relevant": true,
            "faithful": true,
            "complete": true,
            "verdict": "PASS",
            "issues": []
            }}

            Use verdict "PASS" only when relevant, faithful, and complete are all true.
            Otherwise use "FAIL" and list concise issues explaining the failed dimensions.

            QUESTION:
            {question}

            ANSWER:
            {answer}

            CONTEXT:
            [BEGIN CONTEXT]
            {unified_context}
            [END CONTEXT]
"""

        try:
            llm_before = API_TRACKER.llm_snapshot()
            result = Settings.llm.complete(eval_prompt).text.strip()
            API_TRACKER.record_llm_operation(
                "rag_quality_evaluation",
                llm_before,
            )

            evaluation = json.loads(repair_json(result))

            required = {"relevant", "faithful", "complete", "verdict", "issues"}
            if not required.issubset(evaluation):
                raise ValueError("Evaluator response is missing required fields.")

            evaluation["relevant"] = bool(evaluation["relevant"])
            evaluation["faithful"] = bool(evaluation["faithful"])
            evaluation["complete"] = bool(evaluation["complete"])
            evaluation["verdict"] = str(evaluation["verdict"]).upper()
            evaluation["issues"] = list(evaluation["issues"] or [])
            evaluation["evaluation_error"] = False

            evaluation["verdict"] = (
                "PASS"
                if evaluation["relevant"]
                and evaluation["faithful"]
                and evaluation["complete"]
                else "FAIL"
            )

            API_TRACKER.record_evaluation(
                failed=evaluation["verdict"] != "PASS"
            )
            return evaluation

        except Exception as error:
            API_TRACKER.record_evaluation(failed=True)
            return {
                "relevant": False,
                "faithful": False,
                "complete": False,
                "verdict": "FAIL",
                "evaluation_error": True,
                "issues": [
                    f"RAG quality evaluation failed: {error}"
                ],
            }

    def print_evaluation(self, evaluation: Dict):
        print("\n[RAG QUALITY EVALUATION]")
        print(f"  Retrieval relevance: {'PASS' if evaluation['relevant'] else 'FAIL'}")
        print(f"  Faithfulness:        {'PASS' if evaluation['faithful'] else 'FAIL'}")
        print(f"  Completeness:        {'PASS' if evaluation['complete'] else 'FAIL'}")
        print(f"  Verdict:             {evaluation['verdict']}")
        if evaluation["issues"]:
            for issue in evaluation["issues"]:
                print(f"  Issue:               {issue}")

    def ask(
        self,
        question: str,
        rag_mode: str = "auto",
        max_retries: int = 3,
    ) -> str:
        if self.vector_index is None:
            if not self.load_indexes():
                return (
                    "The knowledge base is not initialized. "
                    "Add documents and synchronize first."
                )

        if not self.lexical_index.built:
            self.rebuild_lexical_index()

        print(f"\n[QUESTION]\n{question}")
        query_start = time.perf_counter()

        profile = self.profile_query(
            question,
            rag_mode=rag_mode,
        )

        print(
            f"[RETRIEVAL STRATEGY] "
            f"{profile.mode} "
            f"({profile.reason})"
        )

        llm_before = API_TRACKER.llm_snapshot()
        retrieval_start = time.perf_counter()

        if profile.mode == "summary":
            generation_start = time.perf_counter()
            if self.engine is None:
                self.build_router()

            if self.engine is None:
                generated_answer = (
                    "The summary retrieval engine is unavailable."
                )
                retrieved_contexts = []
            else:
                response = self.engine.query(question)
                generated_answer = response.response
                retrieved_contexts = [
                    node.node.get_content()
                    for node in response.source_nodes
                ]
        else:
            retrieved_nodes, _ = self.retrieve_adaptively(
                question,
                profile,
            )
            retrieved_nodes = retrieved_nodes or []

            generation_start = time.perf_counter()
            generated_answer = self.synthesize_answer(
                question,
                retrieved_nodes,
            )
            retrieved_contexts = [
                node_with_score.node.get_content()
                for node_with_score in retrieved_nodes
            ]

        retrieval_seconds = time.perf_counter() - retrieval_start
        generation_seconds = time.perf_counter() - generation_start

        API_TRACKER.record_llm_operation(
            "answer_generation",
            llm_before,
        )

        unified_context = "\n---\n".join(
            retrieved_contexts
        )

        attempt = 0
        evaluation = None

        while attempt < max_retries:
            attempt += 1

            evaluation_start = time.perf_counter()
            evaluation = self.evaluate_rag_response(
                question=question,
                answer=generated_answer,
                contexts=retrieved_contexts,
            )
            evaluation_seconds = time.perf_counter() - evaluation_start
            API_TRACKER.record_query_stages(
                retrieval=retrieval_seconds if attempt == 1 else 0.0,
                generation=generation_seconds if attempt == 1 else 0.0,
                evaluation=evaluation_seconds,
            )

            self.print_evaluation(evaluation)

            if evaluation["verdict"] == "PASS":
                print(
                    f"\n[ANSWER] (Verified on Attempt {attempt})"
                )
                print(generated_answer)
                print("=" * 70)
                query_seconds = time.perf_counter() - query_start
                API_TRACKER.record_query(
                    1,
                    query_seconds,
                    len(retrieved_contexts),
                )
                API_TRACKER.print_query_usage(
                    llm_before,
                    strategy=profile.mode,
                    retrieved_nodes=len(retrieved_contexts),
                    query_seconds=query_seconds,
                )
                return generated_answer

            if attempt >= max_retries:
                break

            if evaluation.get("evaluation_error"):
                print(
                    "[GUARDRAIL] RAG evaluator failed. "
                    "Skipping self-correction because the failure is not a generation-quality signal."
                )
                break

            if not evaluation["relevant"]:
                print(
                    "[GUARDRAIL] Required evidence was not found in the retrieved context. "
                    "Skipping self-correction."
                )
                break

            print(
                f"[AUDIT WARNING] Attempt {attempt} failed RAG quality "
                f"evaluation. Running self-correction..."
            )

            correction_prompt = f"""
                You are correcting a RAG answer that failed a strict quality evaluation.
                Rewrite the answer using ONLY the supplied VERIFIED CONTEXT.

                Rules:
                1. Every factual claim must be directly supported by the context.
                2. Do not use outside knowledge.
                3. Do not invent or approximate numbers, dates, names, identifiers, or relationships.
                4. Simple arithmetic is allowed only when every input value is explicitly present in the context.
                5. Address every material part of the user's question that the context supports.
                6. If the context does not contain enough information, say exactly:
                "The provided documentation does not contain this information."
                7. Do not mention the evaluation process.

                EVALUATION ISSUES:
                {json.dumps(evaluation["issues"], ensure_ascii=False)}

                QUESTION:
                {question}

                VERIFIED CONTEXT:
                [BEGIN CONTEXT]
                {unified_context}
                [END CONTEXT]

                PREVIOUS ANSWER:
                {generated_answer}

                Provide only the corrected answer.
"""

            try:
                correction_before = API_TRACKER.llm_snapshot()
                generated_answer = Settings.llm.complete(
                    correction_prompt
                ).text.strip()
                API_TRACKER.record_llm_operation(
                    "self_correction",
                    correction_before,
                )
                API_TRACKER.record_self_correction()
            except Exception as error:
                print(
                    f"--> [ERROR] Network drop during "
                    f"correction retry: {error}"
                )
                break

        query_seconds = (
            time.perf_counter() - query_start
        )
        API_TRACKER.record_query(
            1,
            query_seconds,
            len(retrieved_contexts),
        )
        API_TRACKER.print_query_usage(
            llm_before,
            strategy=profile.mode,
            retrieved_nodes=len(retrieved_contexts),
            query_seconds=query_seconds,
        )

        print(
            "[GUARDRAIL BLOCK] Maximum self-correction "
            "retries reached. Output completely masked."
        )
        return (
            "I apologize, but I am unable to verify or prove "
            "that claim using the uploaded documentation."
        )

    def clear_storage(self):

        print(
            "--> [WIPE] "
            "Deleting RAG storage and container volumes..."
        )

        try:
            collection_name = "pipeline_collection"
            collections = self.db_client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                self.db_client.delete_collection(collection_name=collection_name)
                print("--> [WIPE] Qdrant collection dropped from Docker bubble.")
        except Exception as e:
            print(f"--> [WARNING] Failed to drop Qdrant collection: {e}")


        if STORAGE_DIR.exists():

            for item in (
                STORAGE_DIR.iterdir()
            ):

                if item.is_file():

                    item.unlink()

                elif item.is_dir():

                    shutil.rmtree(
                        item
                    )

        STORAGE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
