from .engine import AdaptiveRAG
from .observability import API_TRACKER

def run():
    rag = AdaptiveRAG(data_dir="./data")

    print("--> [SYSTEM LOG] Initiating file inventory and database sync...")
    rag.sync(force_rebuild=False)
    API_TRACKER.print_sync_observability()
    print("--> [SYSTEM LOG] Synchronized successfully. Knowledge base is online.")

    print("\n" + "=" * 60)
    print("ADAPTIVE RAG KNOWLEDGE BASE")
    print("  Type your questions below. Type 'exit' or 'quit' to close.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_query = input("Enter your query: ").strip()
            if user_query.lower() in {"exit", "quit"}:
                print("\n--> [SYSTEM LOG] Shutting down connection layers. Goodbye.")
                API_TRACKER.print_summary()
                break
            if not user_query:
                continue

            rag_mode = input(
                "RAG mode [auto/semantic/keyword/hybrid/summary] "
                "(default auto): "
            ).strip() or "auto"

            rag.ask(user_query, rag_mode=rag_mode)

        except KeyboardInterrupt:
            print("\n\n--> [SYSTEM LOG] System execution interrupted by user. Closing safely.")
            break
        except Exception as error:
            print(f"\n[RUNTIME ERROR]: An error occurred: {error}\n")
