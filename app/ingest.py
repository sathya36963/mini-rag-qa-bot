
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(data_dir: str | Path) -> list[Document]:
    """
    Load TXT and PDF documents from a directory.

    Each returned Document contains:
    - page_content: extracted text
    - metadata: source filename and, for PDFs, page number
    """

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_path}"
        )

    if not data_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {data_path}"
        )

    documents = []

    # Read files in a predictable order.
    for file_path in sorted(data_path.iterdir()):

        if not file_path.is_file():
            continue

        # Handle text files.
        if file_path.suffix.lower() == ".txt":

            text = file_path.read_text(encoding="utf-8").strip()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "file_type": "txt"
                        }
                    )
                )

        # Handle PDF files.
        elif file_path.suffix.lower() == ".pdf":

            pdf_reader = PdfReader(str(file_path))

            for page_number, page in enumerate(
                pdf_reader.pages, start=1
            ):
                text = (page.extract_text() or "").strip()

                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file_path.name,
                                "page": page_number,
                                "file_type": "pdf"
                            }
                        )
                    )

    if not documents:
        raise ValueError(
            "No readable TXT or PDF content found in the data directory."
        )

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> list[Document]:
    """
    Split loaded documents into smaller chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def main():
    # Find the project's data directory.
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    print("\nLoading documents...")

    documents = load_documents(data_dir)

    print(f"Documents loaded: {len(documents)}")

    print("\nSplitting documents...")

    chunks = split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    # Display up to five chunks.
    for index, chunk in enumerate(chunks[:5], start=1):

        print(f"\n---------- CHUNK {index} ----------")
        print("Metadata:", chunk.metadata)
        print("Content:")
        print(chunk.page_content)


if __name__ == "__main__":
    main()
