import re
from typing import List
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser, CodeSplitter
from llama_index.core.ingestion import IngestionPipeline

from .documents import DocumentProfile, profile_document

class AdaptiveChunker:

    def process(
        self,
        documents: List[Document],
    ):

        nodes = []

        for document in documents:

            path = Path(
                document.metadata.get(
                    "file_path",
                    "unknown",
                )
            )

            profile = profile_document(
                path,
                extracted_text=document.text,
            )

            document.metadata.update(
                {
                    "word_count": profile.word_count,
                    "line_count": profile.line_count,
                    "has_headers": profile.has_headers,
                    "has_tables": profile.has_tables,
                    "has_code": profile.has_code,
                    "structure_depth": profile.structure_depth,
                }
            )

            strategy = self.choose_strategy(
                profile
            )

            print(
                f"--> [PROFILE] "
                f"{path.name} | "
                f"type={profile.document_type} | "
                f"words={profile.word_count} | "
                f"strategy={strategy}"
            )

            generated_nodes = (
                self.chunk_document(
                    document,
                    profile,
                    strategy,
                )
            )

            nodes.extend(
                generated_nodes
            )

        return nodes

    def choose_strategy(
        self,
        profile: DocumentProfile,
    ) -> str:

        if profile.document_type == "code":
            return "code"

        if profile.document_type == "markdown":
            return "markdown"

        if profile.document_type == "document":

            if (
                profile.has_headers
                or profile.has_tables
            ):
                return "structured_document"

            return "document"

        if profile.document_type in {
            "json",
            "tabular",
            "spreadsheet",
        }:
            return "structured_data"

        if profile.document_type == "html":
            return "html"

        if profile.word_count < 700:
            return "short_text"

        return "adaptive_text"

    def chunk_document(
        self,
        document: Document,
        profile: DocumentProfile,
        strategy: str,
    ):

        if strategy == "code":

            return self.chunk_code(
                document,
                profile,
            )

        if strategy in {
            "markdown",
            "structured_document",
        }:

            return self.chunk_markdown(
                document
            )

        if strategy == "structured_data":

            return self.chunk_structured(
                document
            )

        if strategy == "short_text":

            return [
                self.set_strategy(
                    document,
                    "minimal",
                )
            ]

        return self.chunk_text(
            document,
            profile,
        )

    def chunk_code(
        self,
        document: Document,
        profile: DocumentProfile,
    ):

        language = (
            profile.language
            or "python"
        )

        try:

            splitter = CodeSplitter(
                language=language,
                chunk_lines=self.dynamic_code_lines(
                    profile.line_count
                ),
            )

            pipeline = IngestionPipeline(
                transformations=[
                    splitter
                ]
            )

            nodes = pipeline.run(
                documents=[document]
            )

            for node in nodes:

                node.metadata.update(
                    {
                        "chunk_strategy": "code",
                        "language": language,
                    }
                )

            return nodes

        except Exception as error:

            print(
                f"--> [WARNING] "
                f"Code splitter failed: {error}"
            )

            return self.chunk_text(
                document,
                profile,
            )

    def dynamic_code_lines(
        self,
        line_count: int,
    ) -> int:

        if line_count < 100:
            return 80

        if line_count < 500:
            return 60

        if line_count < 2000:
            return 50

        if line_count < 5000:
            return 40

        return 30

    def chunk_markdown(
        self,
        document: Document,
    ):

        pipeline = IngestionPipeline(
            transformations=[
                MarkdownNodeParser()
            ]
        )

        nodes = pipeline.run(
            documents=[document]
        )

        for node in nodes:

            node.metadata[
                "chunk_strategy"
            ] = "markdown_structure"

        return nodes

    def chunk_structured(
        self,
        document: Document,
    ):

        word_count = len(
            re.findall(
                r"\b\w+\b",
                document.text,
            )
        )

        if word_count < 2000:

            return [
                self.set_strategy(
                    document,
                    "structured_data",
                )
            ]

        return self.chunk_text(
            document,
            DocumentProfile(
                path=Path(
                    document.metadata[
                        "file_path"
                    ]
                ),
                extension=document.metadata.get(
                    "extension",
                    "",
                ),
                file_size_bytes=0,
                document_type=document.metadata.get(
                    "document_type", 
                    "tabular"
                ),
                word_count=word_count,
            ),
        )

    def chunk_text(
        self,
        document: Document,
        profile: DocumentProfile,
    ):

        chunk_size = (
            self.dynamic_chunk_size(
                profile.word_count
            )
        )

        overlap = (
            self.dynamic_overlap(
                chunk_size
            )
        )

        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

        pipeline = IngestionPipeline(
            transformations=[
                splitter
            ]
        )

        nodes = pipeline.run(
            documents=[document]
        )

        for node in nodes:

            node.metadata[
                "chunk_strategy"
            ] = "adaptive_text"

        return nodes

    def dynamic_chunk_size(
        self,
        word_count: int,
    ) -> int:

        if word_count < 1000:
            return 700

        if word_count < 5000:
            return 650

        if word_count < 20000:
            return 550

        if word_count < 100000:
            return 450

        return 350

    def dynamic_overlap(
        self,
        chunk_size: int,
    ) -> int:

        return max(
            30,
            int(chunk_size * 0.10),
        )

    def set_strategy(
        self,
        document,
        strategy,
    ):

        document.metadata[
            "chunk_strategy"
        ] = strategy

        return document
