
import pytest

from langchain_core.documents import Document
from app.ingest import load_documents, split_documents


def test_load_documents_reads_txt_file(tmp_path):

    sample_file = tmp_path / "sample.txt"

    sample_file.write_text(
        "Python is a programming language.",
        encoding="utf-8"
    )

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].page_content == "Python is a programming language."
    assert documents[0].metadata["source"] == "sample.txt"


def test_split_documents_creates_multiple_chunks():

    document = Document(
        page_content="Python is useful. " * 40,
        metadata={"source": "sample.txt"}
    )

    chunks = split_documents(
        [document],
        chunk_size=100,
        chunk_overlap=20
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.page_content) <= 100
        assert chunk.metadata["source"] == "sample.txt"


def test_split_documents_rejects_invalid_overlap():

    document = Document(
        page_content="Python is a programming language.",
        metadata={"source": "sample.txt"}
    )

    with pytest.raises(ValueError):
        split_documents(
            [document],
            chunk_size=100,
            chunk_overlap=100
        )
