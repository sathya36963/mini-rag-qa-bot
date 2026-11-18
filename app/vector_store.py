from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config import (
    DATA_DIR,
    CHROMA_DIR,
    COLLECTION_NAME
)

from app.ingest import load_documents, split_documents
def build_vector_store():
    print("Loading documents...")

    documents = load_documents(DATA_DIR)

    print("Splitting documents...")

    chunks = split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    print("Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR)
    )

    print("Documents successfully stored in ChromaDB!")

    return vector_store
def main():

    vector_store = build_vector_store()

    question = "What is the difference between a list and tuple?"

    print("\nSearching for relevant documents...")

    results = vector_store.similarity_search(
        query=question,
        k=2
    )

    for result in results:
        print("\nRetrieved content:")
        print(result.page_content)


if __name__ == "__main__":
    main()