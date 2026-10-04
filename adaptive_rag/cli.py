import argparse
import logging

from .engine import AdaptiveRAG
from .observability import API_TRACKER


def _print_sync_report(report: dict) -> None:
    print(
        f"--> [SYNC] added {len(report['added'])}, "
        f"updated {len(report['updated'])}, "
        f"deleted {len(report['deleted'])}, "
        f"unchanged {report['unchanged']}"
    )
    if report.get("rebuild"):
        print(f"--> [SYNC] full rebuild: {report['rebuild']}")
    for item in report["failed"]:
        print(f"--> [SYNC WARNING] {item['path']} was not indexed: {item.get('error')}")
    for item in report["skipped"]:
        print(f"--> [SYNC WARNING] {item['path']} skipped: {item['reason']}")


def _print_sources(sources: list) -> None:
    seen = set()
    lines = []
    for source in sources:
        section = source.get("header_path") or source.get("sheet_name") or ""
        key = (source.get("file_name"), section)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  - {source.get('file_name') or 'unknown'} {section}".rstrip())
    if lines:
        print("\n[SOURCES]")
        print("\n".join(lines))


def run(argv=None):
    parser = argparse.ArgumentParser(description="Adaptive RAG knowledge base")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="wipe the index and rebuild it from the data folder",
    )
    args = parser.parse_args(argv)

    # The chunker and engine log through `logging`: show this package's INFO
    # lines (profiling/strategy) without turning on every library's chatter.
    logging.basicConfig(level=logging.WARNING)
    if __package__:
        logging.getLogger(__package__).setLevel(logging.INFO)

    try:
        # No data_dir argument: use DATA_DIR from config.py.
        rag = AdaptiveRAG()
        print("--> [SYSTEM LOG] Initiating file inventory and database sync...")
        report = rag.sync(force_rebuild=args.rebuild)
    except Exception as error:
        print(f"\n[STARTUP ERROR]: {error}")
        return

    _print_sync_report(report)
    API_TRACKER.print_sync_observability()
    print("--> [SYSTEM LOG] Sync finished. Knowledge base is online.")

    print("\n" + "=" * 60)
    print("ADAPTIVE RAG KNOWLEDGE BASE")
    print("  Type your questions below. Type 'exit' or 'quit' to close.")
    print("=" * 60 + "\n")

    try:
        while True:
            try:
                user_query = input("Enter your query: ").strip()
                if user_query.lower() in {"exit", "quit"}:
                    print("\n--> [SYSTEM LOG] Shutting down connection layers. Goodbye.")
                    break
                if not user_query:
                    continue

                rag_mode = input(
                    "RAG mode [auto/semantic/keyword/hybrid/summary] "
                    "(default auto): "
                ).strip() or "auto"

                result = rag.ask_detailed(user_query, rag_mode=rag_mode)

                # A verified answer is already printed by the engine. Every
                # other outcome (no evidence, unverified, error) would
                # otherwise show the user nothing.
                if result["status"] != "verified":
                    print(f"\n[{result['status'].upper()}] {result['answer']}")
                _print_sources(result["sources"])

            except (KeyboardInterrupt, EOFError):
                print("\n\n--> [SYSTEM LOG] Input closed. Closing safely.")
                break
            except ValueError as error:
                print(f"\n[INPUT ERROR]: {error}\n")
            except Exception as error:
                print(f"\n[RUNTIME ERROR]: An error occurred: {error}\n")
    finally:
        API_TRACKER.print_summary()