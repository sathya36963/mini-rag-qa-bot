
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import CHROMA_DIR, COLLECTION_NAME


# Load the existing ChromaDB vector database
def load_vector_store():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    return vector_store


# Retrieve relevant documents and generate an answer
def answer_question(question, vector_store):

    # Retrieve the three most relevant document chunks
    retrieved_docs = vector_store.similarity_search(
        query=question,
        k=3
    )

    # Check whether any documents were retrieved
    if not retrieved_docs:
        return (
            "No relevant information found in the documents.",
            retrieved_docs
        )

    # Combine retrieved document chunks into context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Create instructions for the AI model
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a strict document-based question answering assistant.

            Follow these rules:

            1. Use ONLY facts explicitly mentioned in the context.
            2. Never use your general knowledge.
            3. Do not add additional properties or explanations.
            4. If the document mentions only two facts, provide only those two facts..
            5. Do not guess or assume missing information,
            . Include all relevant information from the context.
            7. Do not omit details that directly answer the question.
            8. If the answer is not present, say:
                "This information is not available in the documents."
             Answer format:
               Provide a short and direct answer.

            Context:
            {context}
            """
        ),
        ("human", "{question}")
    ])

    # Connect to the local Llama model using Ollama
    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0
    )

    # Create the LangChain processing pipeline
    chain = prompt | llm | StrOutputParser()

    # Generate the answer
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, retrieved_docs


# Main interactive chatbot
def main():

    vector_store = load_vector_store()

    print("\n===== MINI RAG QA BOT =====")
    print("Ask questions about your documents.")
    print("Type 'exit' to quit.")

    while True:

        question = input("\nYour question: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        # Generate answer using RAG
        answer, _ = answer_question(
            question,
            vector_store
        )

        # Display only the answer
        print("\nAnswer:")
        print(answer)
if __name__ == "__main__":
    main()