
# Mini RAG QA Bot

A beginner-friendly Retrieval-Augmented Generation (RAG) application built with Python, LangChain, Hugging Face, ChromaDB and Ollama.

The application allows users to ask questions about their own PDF and TXT documents. It retrieves relevant document chunks using semantic search and generates answers using a locally running LLM.

## Features

- Load PDF and TXT documents.
- Extract text from documents.
- Split documents into smaller chunks.
- Generate embeddings locally using Hugging Face.
- Store and retrieve embeddings using ChromaDB.
- Perform semantic similarity search.
- Generate answers using Llama 3.2 through Ollama.
- Display answers through an interactive terminal interface.
- Run automated tests using pytest.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| LangChain | RAG application framework |
| Hugging Face | Local text embeddings |
| Sentence Transformers | Embedding model |
| ChromaDB | Vector database |
| Ollama | Local LLM runtime |
| Llama 3.2 (1B) | Answer generation |
| PyPDF | PDF text extraction |
| Pytest | Automated testing |

## Architecture

```text
             PDF / TXT Documents
                      |
                      v
               Document Loader
                      |
                      v
                 Text Splitter
                      |
                      v
              Hugging Face Model
                  (Embeddings)
                      |
                      v
                  ChromaDB
                      |
                      |
User Question ------+
       |
       v
  Semantic Search
       |
       v
 Relevant Document Chunks
       |
       v
   Context + Question
       |
       v
    Ollama / Llama 3.2
       |
       v
    Generated Answer
```

## Project Structure

```text
mini-rag-qa-bot/
|
|-- app/
|   |-- config.py
|   |-- ingest.py
|   |-- vector_store.py
|   |-- qa.py
|
|-- data/
|   |-- sample.txt
|
|-- tests/
|   |-- test_ingest.py
|
|-- .env.example
|-- .gitignore
|-- requirements.txt
|-- README.md
```

## Requirements

- Python 3.11 or newer
- Git
- Ollama
- Internet connection for the initial model downloads

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sathya36963/mini-rag-qa-bot.git

cd mini-rag-qa-bot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the Ollama model

Make sure Ollama is installed.

```bash
ollama pull llama3.2:1b
```

### 5. Generate document embeddings

```bash
python -m app.vector_store
```

This loads the documents, creates chunks, generates local embeddings and stores them in ChromaDB.

### 6. Start the chatbot

```bash
python -m app.qa
```

Ask questions about your documents in the terminal.

Type `exit` to quit.

## Example

**Question:**

What is the difference between a list and tuple?

**Answer:**

A list is mutable, meaning its elements can be changed after creation.

A tuple is immutable, meaning its elements cannot be changed after creation.

## Running Tests

Run the automated test suite:

```bash
python -m pytest tests/ -v
```

The tests cover:

- TXT document loading.
- Document chunking.
- Validation of chunk parameters.

## Models Used

**Embedding Model:**

`sentence-transformers/all-MiniLM-L6-v2`

**Language Model:**

`llama3.2:1b`

Both models run locally after their initial downloads.

## Privacy and Cost

- No OpenAI API key is required.
- Embeddings are generated locally.
- Llama inference runs locally through Ollama.
- Documents and the ChromaDB database remain on the local machine unless you explicitly share them.

## Limitations

- Small language models may occasionally generate incomplete or inaccurate answers.
- Scanned PDFs without extractable text are not supported by the current PDF extraction pipeline.
- Answer quality depends on document content, retrieval and the language model.

## Future Improvements

- Support more document formats.
- Improve retrieval accuracy.
- Add conversation history.
- Add a web-based chatbot interface.
- Improve answer verification and source citations.

## Author

Sathya Reddy

GitHub: https://github.com/sathya36963
