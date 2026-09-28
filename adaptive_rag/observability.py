import time

from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

class APIUsageTracker:
    """Lightweight, real-time observability for the current application session."""

    def __init__(self):
        self.started_at = time.perf_counter()
        self.llm_token_counter = TokenCountingHandler()
        self.callback_manager = CallbackManager([self.llm_token_counter])

        # External API / model activity
        self.llm_calls = 0
        self.llm_operations = {}
        self.llama_parse_calls = 0
        self.llama_parse_duration = 0.0

        # Local pipeline activity
        self.embedding_calls = 0
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

    def llm_snapshot(self):
        return (
            len(self.llm_token_counter.llm_token_counts),
            self.llm_token_counter.total_llm_token_count,
            self.llm_token_counter.prompt_llm_token_count,
            self.llm_token_counter.completion_llm_token_count,
        )

    def embedding_snapshot(self):
        handler = self.llm_token_counter
        return (
            len(getattr(handler, "embedding_token_counts", [])),
            getattr(handler, "total_embedding_token_count", 0),
        )

    def record_llm_operation(self, name, before_snapshot):
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
        self.llama_parse_calls += 1
        self.llama_parse_duration += duration

    def record_embedding_delta(self, before_snapshot):
        after = self.embedding_snapshot()
        calls = max(0, after[0] - before_snapshot[0])
        tokens = max(0, after[1] - before_snapshot[1])
        self.embedding_calls += calls
        self.embedding_tokens += tokens

    def record_embedding_time(self, duration):
        self.embedding_duration += duration
        self.embedding_batches += 1

    def record_query_embedding_time(self, duration):
        self.query_embedding_duration += duration

    def record_indexing(self, duration, nodes):
        self.indexing_duration += duration
        self.nodes_indexed += nodes

    def record_chunking(self, duration):
        self.chunking_duration += duration

    def record_sync(self, duration):
        self.sync_duration += duration

    def print_sync_observability(self):
        print("\n[INGESTION / INDEXING METRICS]")

        print(
            f"  Sync time:          "
            f"{self.sync_duration:.2f}s"
        )

        if not self.sync_had_ingestion:
            print("  LlamaParse:         No parsing performed")
            print("  Local pipeline:     No ingestion performed")
            return

        print("  LlamaParse:")
        if self.llama_parse_calls:
            print(
                f"    API calls:        "
                f"{self.llama_parse_calls}"
            )
            print(
                f"    Parse time:       "
                f"{self.llama_parse_duration:.2f}s"
            )
        else:
            print("    No parsing performed")

        print("  Local pipeline:")
        print(
            f"    Nodes indexed:    "
            f"{self.nodes_indexed:,}"
        )
        print(
            f"    Indexing time:    "
            f"{self.indexing_duration:.2f}s"
        )
        print(
            f"    Chunking time:    "
            f"{self.chunking_duration:.2f}s"
        )
        print(
            f"    Embedding calls:  "
            f"{self.embedding_calls}"
        )
        print(
            f"    Embedding tokens: "
            f"{self.embedding_tokens:,}"
        )
        print(
            f"    Embedding time:   "
            f"{self.embedding_duration:.2f}s"
        )
        print(
            f"    Embedding batches:{self.embedding_batches}"
        )

    def record_query(self, query_count, duration, retrieved_nodes):
        self.query_count += 1
        self.query_duration += duration
        self.retrieval_nodes += retrieved_nodes

    def query_usage_delta(self, before_snapshot):
        after = self.llm_snapshot()
        return (
            max(0, after[0] - before_snapshot[0]),
            max(0, after[2] - before_snapshot[2]),
            max(0, after[3] - before_snapshot[3]),
            max(0, after[1] - before_snapshot[1]),
        )

    def print_query_usage(
        self,
        before_snapshot,
        strategy=None,
        retrieved_nodes=None,
        query_seconds=None,
    ):
        calls, input_tokens, output_tokens, total = (
            self.query_usage_delta(before_snapshot)
        )

        print("\n[QUERY METRICS]")

        print("  Retrieval:")
        if strategy is not None:
            print(f"    Strategy:         {strategy}")

        if retrieved_nodes is not None:
            print(f"    Retrieved nodes:  {retrieved_nodes}")

        if query_seconds is not None:
            print(
                f"    Query time:       "
                f"{query_seconds:.2f}s"
            )

        print("  Gemini:")
        print(f"    API calls:        {calls}")
        print(f"    Input tokens:     {input_tokens:,}")
        print(f"    Output tokens:    {output_tokens:,}")
        print(f"    Total tokens:     {total:,}")

    def record_evaluation(self, failed: bool = False):
        self.evaluation_count += 1
        if failed:
            self.evaluation_failures += 1

    def record_self_correction(self):
        self.self_corrections += 1

    def record_persist(self, duration):
        self.persist_count += 1
        self.persist_duration += duration

    def record_lexical_rebuild(self, duration):
        self.lexical_rebuild_count += 1
        self.lexical_rebuild_duration += duration

    def record_retriever_cache(self, hit: bool):
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
        self.query_retrieval_duration += retrieval
        self.query_generation_duration += generation
        self.query_evaluation_duration += evaluation

    def print_summary(self):
        input_tokens = self.llm_token_counter.prompt_llm_token_count
        output_tokens = self.llm_token_counter.completion_llm_token_count
        total_tokens = self.llm_token_counter.total_llm_token_count
        session_seconds = time.perf_counter() - self.started_at

        print("\n" + "=" * 70)
        print("                    SESSION METRICS")
        print("=" * 70)

        print("\nQUERY / RETRIEVAL")
        print(f"  Queries:            {self.query_count}")
        print(f"  Retrieved nodes:    {self.retrieval_nodes:,}")
        print(f"  Query time:         {self.query_duration:.2f}s")

        print("\nGEMINI")
        print(f"  API calls:          {self.llm_calls}")
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

API_TRACKER = APIUsageTracker()
API_TRACKER.attach()
