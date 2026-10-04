import threading
import time
from typing import Optional

from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler


class APIUsageTracker:
    """Lightweight, real-time observability for the current application session."""

    def __init__(self):
        self.started_at = time.perf_counter()
        # FastAPI requests can read/write metrics from different worker
        # threads. Protect counters and per-operation dictionaries so /metrics
        # sees a consistent snapshot rather than partially updated state.
        self._lock = threading.RLock()
        self.llm_token_counter = TokenCountingHandler()
        self.callback_manager = CallbackManager([self.llm_token_counter])

        # Running totals. TokenCountingHandler keeps EVERY event (including the
        # full prompt and completion text) in a list forever and recomputes its
        # totals by summing that list. In a long-running server that is an
        # unbounded memory leak and an O(n) cost per /metrics call. We fold the
        # events into these integers and let the handler's lists be dropped
        # (see _fold_token_events).
        self._llm_events = 0
        self._llm_prompt_tokens = 0
        self._llm_completion_tokens = 0
        self._embedding_events = 0
        self._embedding_tokens = 0

        # External API / model activity
        self.llm_calls = 0
        self.llm_operations = {}
        self.llama_parse_calls = 0
        self.llama_parse_duration = 0.0

        # Local pipeline activity
        self.embedding_calls = 0  # NOTE: counts texts/chunks embedded, not API calls
        self.embedding_tokens = 0
        self.embedding_duration = 0.0
        self.query_embedding_duration = 0.0
        self.embedding_batches = 0
        self.nodes_indexed = 0
        self.indexing_duration = 0.0
        self.chunking_duration = 0.0

        # Query/retrieval activity
        self.query_count = 0
        self.retrieval_nodes = 0
        self.query_duration = 0.0
        self.sync_duration = 0.0
        self.sync_had_ingestion = False

        # RAG quality evaluation
        self.evaluation_count = 0
        self.evaluation_failures = 0
        self.self_corrections = 0

        # Performance / efficiency observability
        self.persist_count = 0
        self.persist_duration = 0.0
        self.lexical_rebuild_count = 0
        self.lexical_rebuild_duration = 0.0
        self.retriever_cache_hits = 0
        self.retriever_cache_misses = 0
        self.query_retrieval_duration = 0.0
        self.query_generation_duration = 0.0
        self.query_evaluation_duration = 0.0

    def attach(self):
        Settings.callback_manager = self.callback_manager

    # ------------------------------------------------------------------ #
    # Token accounting
    # ------------------------------------------------------------------ #

    def _fold_token_events(self):
        """Move events out of the handler into integer totals. Caller holds the lock.

        The list is swapped for an empty one in a single statement, so events
        recorded afterwards land in the new list and are picked up next time.
        """
        handler = self.llm_token_counter

        llm_events, handler.llm_token_counts = handler.llm_token_counts, []
        for event in llm_events:
            self._llm_events += 1
            self._llm_prompt_tokens += getattr(event, "prompt_token_count", 0)
            self._llm_completion_tokens += getattr(event, "completion_token_count", 0)

        if hasattr(handler, "embedding_token_counts"):
            embedding_events, handler.embedding_token_counts = (
                handler.embedding_token_counts,
                [],
            )
            for event in embedding_events:
                self._embedding_events += 1
                self._embedding_tokens += getattr(event, "prompt_token_count", 0)

    def llm_snapshot(self):
        """(calls, total_tokens, prompt_tokens, completion_tokens). Same shape as before."""
        with self._lock:
            self._fold_token_events()
            return (
                self._llm_events,
                self._llm_prompt_tokens + self._llm_completion_tokens,
                self._llm_prompt_tokens,
                self._llm_completion_tokens,
            )

    def embedding_snapshot(self):
        """(texts_embedded, tokens). Same shape as before."""
        with self._lock:
            self._fold_token_events()
            return (self._embedding_events, self._embedding_tokens)

    def snapshot(self):
        """Return a JSON-serializable point-in-time metrics snapshot."""
        with self._lock:
            llm_calls, llm_total, llm_input, llm_output = self.llm_snapshot()
            return {
                "uptime_seconds": time.perf_counter() - self.started_at,
                "llm": {
                    # Calls and tokens both come from the handler, so they
                    # always agree. Token values are ESTIMATES (see summary).
                    "calls": llm_calls,
                    "input_tokens": llm_input,
                    "output_tokens": llm_output,
                    "total_tokens": llm_total,
                    "operations": {
                        name: dict(values)
                        for name, values in self.llm_operations.items()
                    },
                },
                "llama_parse": {
                    "calls": self.llama_parse_calls,
                    "duration_seconds": self.llama_parse_duration,
                },
                "embeddings": {
                    "calls": self.embedding_calls,
                    "tokens": self.embedding_tokens,
                    "duration_seconds": self.embedding_duration,
                    "batches": self.embedding_batches,
                    "query_duration_seconds": self.query_embedding_duration,
                },
                "indexing": {
                    "nodes": self.nodes_indexed,
                    "duration_seconds": self.indexing_duration,
                    "chunking_duration_seconds": self.chunking_duration,
                    "persist_count": self.persist_count,
                    "persist_duration_seconds": self.persist_duration,
                    "lexical_rebuild_count": self.lexical_rebuild_count,
                    "lexical_rebuild_duration_seconds": self.lexical_rebuild_duration,
                },
                "retrieval": {
                    "queries": self.query_count,
                    "nodes": self.retrieval_nodes,
                    "duration_seconds": self.query_duration,
                    "cache_hits": self.retriever_cache_hits,
                    "cache_misses": self.retriever_cache_misses,
                    "retrieval_duration_seconds": self.query_retrieval_duration,
                    "generation_duration_seconds": self.query_generation_duration,
                    "evaluation_duration_seconds": self.query_evaluation_duration,
                },
                "evaluation": {
                    "count": self.evaluation_count,
                    "failures": self.evaluation_failures,
                    "self_corrections": self.self_corrections,
                },
                "sync": {
                    "duration_seconds": self.sync_duration,
                    "last_sync_had_ingestion": self.sync_had_ingestion,
                },
            }

    # ------------------------------------------------------------------ #
    # Recorders
    # ------------------------------------------------------------------ #

    def record_llm_operation(self, name, before_snapshot):
        """Attribute the LLM usage since `before_snapshot` to operation `name`.

        Limitation: the counters are process-wide, so if two requests call the
        LLM at the same time, each one's delta can include the other's calls.
        Totals stay correct; per-operation attribution is approximate under
        concurrency.
        """
        with self._lock:
            after = self.llm_snapshot()
            calls = max(0, after[0] - before_snapshot[0])
            total = max(0, after[1] - before_snapshot[1])
            input_tokens = max(0, after[2] - before_snapshot[2])
            output_tokens = max(0, after[3] - before_snapshot[3])

            operation = self.llm_operations.setdefault(
                name,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            operation["calls"] += calls
            operation["input_tokens"] += input_tokens
            operation["output_tokens"] += output_tokens
            operation["total_tokens"] += total
            self.llm_calls += calls

    def record_llama_parse(self, duration):
        with self._lock:
            self.llama_parse_calls += 1
            self.llama_parse_duration += duration

    def record_embedding_delta(self, before_snapshot):
        with self._lock:
            after = self.embedding_snapshot()
            calls = max(0, after[0] - before_snapshot[0])
            tokens = max(0, after[1] - before_snapshot[1])
            self.embedding_calls += calls
            self.embedding_tokens += tokens

    def record_embedding_time(self, duration):
        with self._lock:
            self.embedding_duration += duration
            self.embedding_batches += 1

    def record_query_embedding_time(self, duration):
        with self._lock:
            self.query_embedding_duration += duration

    def record_indexing(self, duration, nodes):
        with self._lock:
            self.indexing_duration += duration
            self.nodes_indexed += nodes

    def record_chunking(self, duration):
        with self._lock:
            self.chunking_duration += duration

    def record_sync(self, duration, had_ingestion: Optional[bool] = None):
        with self._lock:
            self.sync_duration += duration
            if had_ingestion is not None:
                self.sync_had_ingestion = had_ingestion

    def record_query(self, query_count, duration, retrieved_nodes):
        # `query_count` is ignored: the count is incremented here. The
        # parameter is kept so existing callers do not break.
        with self._lock:
            self.query_count += 1
            self.query_duration += duration
            self.retrieval_nodes += retrieved_nodes

    def record_evaluation(self, failed: bool = False):
        with self._lock:
            self.evaluation_count += 1
            if failed:
                self.evaluation_failures += 1

    def record_self_correction(self):
        with self._lock:
            self.self_corrections += 1

    def record_persist(self, duration):
        with self._lock:
            self.persist_count += 1
            self.persist_duration += duration

    def record_lexical_rebuild(self, duration):
        with self._lock:
            self.lexical_rebuild_count += 1
            self.lexical_rebuild_duration += duration

    def record_retriever_cache(self, hit: bool):
        with self._lock:
            if hit:
                self.retriever_cache_hits += 1
            else:
                self.retriever_cache_misses += 1

    def record_query_stages(
        self,
        retrieval=0.0,
        generation=0.0,
        evaluation=0.0,
    ):
        with self._lock:
            self.query_retrieval_duration += retrieval
            self.query_generation_duration += generation
            self.query_evaluation_duration += evaluation

    # ------------------------------------------------------------------ #
    # Console reports
    # ------------------------------------------------------------------ #

    def query_usage_delta(self, before_snapshot):
        after = self.llm_snapshot()
        return (
            max(0, after[0] - before_snapshot[0]),
            max(0, after[2] - before_snapshot[2]),
            max(0, after[3] - before_snapshot[3]),
            max(0, after[1] - before_snapshot[1]),
        )

    def print_sync_observability(self):
        print("\n[INGESTION / INDEXING METRICS]")

        print(f"  Sync time:          {self.sync_duration:.2f}s")

        if not self.sync_had_ingestion:
            print("  LlamaParse:         No parsing performed")
            print("  Local pipeline:     No ingestion performed")
            return

        print("  LlamaParse:")
        if self.llama_parse_calls:
            print(f"    API calls:        {self.llama_parse_calls}")
            print(f"    Parse time:       {self.llama_parse_duration:.2f}s")
        else:
            print("    No parsing performed")

        print("  Local pipeline:")
        print(f"    Nodes indexed:    {self.nodes_indexed:,}")
        print(f"    Indexing time:    {self.indexing_duration:.2f}s")
        print(f"    Chunking time:    {self.chunking_duration:.2f}s")
        print(f"    Texts embedded:   {self.embedding_calls}")
        print(f"    Embedding tokens: {self.embedding_tokens:,} (estimated)")
        print(f"    Embedding time:   {self.embedding_duration:.2f}s")
        print(f"    Embedding batches: {self.embedding_batches}")

    def print_query_usage(
        self,
        before_snapshot,
        strategy=None,
        retrieved_nodes=None,
        query_seconds=None,
    ):
        calls, input_tokens, output_tokens, total = self.query_usage_delta(
            before_snapshot
        )

        print("\n[QUERY METRICS]")

        print("  Retrieval:")
        if strategy is not None:
            print(f"    Strategy:         {strategy}")

        if retrieved_nodes is not None:
            print(f"    Retrieved nodes:  {retrieved_nodes}")

        if query_seconds is not None:
            print(f"    Query time:       {query_seconds:.2f}s")

        print("  Gemini (token counts estimated):")
        print(f"    API calls:        {calls}")
        print(f"    Input tokens:     {input_tokens:,}")
        print(f"    Output tokens:    {output_tokens:,}")
        print(f"    Total tokens:     {total:,}")

    def print_summary(self):
        calls, total_tokens, input_tokens, output_tokens = self.llm_snapshot()
        session_seconds = time.perf_counter() - self.started_at

        print("\n" + "=" * 70)
        print("                    SESSION METRICS")
        print("=" * 70)

        print("\nQUERY / RETRIEVAL")
        print(f"  Queries:            {self.query_count}")
        print(f"  Retrieved nodes:    {self.retrieval_nodes:,}")
        print(f"  Query time:         {self.query_duration:.2f}s")

        print("\nGEMINI (token counts estimated)")
        print(f"  API calls:          {calls}")
        print(f"  Input tokens:       {input_tokens:,}  (prompts + instructions + RAG context)")
        print(f"  Output tokens:      {output_tokens:,}  (Gemini-generated text)")
        print(f"  Total tokens:       {total_tokens:,}")
        print("  By operation:")
        for name, usage in self.llm_operations.items():
            print(
                f"    {name}: {usage['calls']} calls | "
                f"input {usage['input_tokens']:,} | "
                f"output {usage['output_tokens']:,} | "
                f"total {usage['total_tokens']:,}"
            )

        print("\nRAG QUALITY")
        print(f"  Evaluations:        {self.evaluation_count}")
        print(f"  Evaluation failures:  {self.evaluation_failures}")
        print(f"  Self-corrections:   {self.self_corrections}")

        print("\nPERFORMANCE")
        print(f"  Index persists:     {self.persist_count}")
        print(f"  Persist time:       {self.persist_duration:.2f}s")
        print(f"  Lexical rebuilds:   {self.lexical_rebuild_count}")
        print(f"  Lexical rebuild:    {self.lexical_rebuild_duration:.2f}s")
        print(f"  Retrieval cache:    {self.retriever_cache_hits} hits / {self.retriever_cache_misses} misses")
        print(f"  Retrieval time:     {self.query_retrieval_duration:.2f}s")
        print(f"  Query embedding:    {self.query_embedding_duration:.2f}s")
        print(f"  Generation time:    {self.query_generation_duration:.2f}s")
        print(f"  Evaluation time:    {self.query_evaluation_duration:.2f}s")

        print(f"\nTOTAL PROCESS TIME:   {session_seconds:.2f}s")
        print("=" * 70)


# Do not attach the callback manager at import time. Model configuration owns
# initialization (see models.configure_models), so importing utilities does
# not mutate global LlamaIndex state before the engine is actually started.
API_TRACKER = APIUsageTracker()