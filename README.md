# Medium Article RAG Assistant

A production-style Retrieval-Augmented Generation (**RAG**) system for querying a Medium articles dataset using **LangChain**, **OpenAI embeddings**, **Pinecone**, **FastAPI**, and **Vercel**.

The assistant answers questions **strictly from the provided article corpus** and returns transparent retrieval evidence, including the retrieved context and the augmented prompt sent to the chat model.

---

## Overview

This project was built to support question answering over a structured Medium dataset stored in CSV format.

Each article record includes:

- `title`
- `text`
- `url`
- `authors`
- `timestamp`
- `tags`

The system follows a closed-domain RAG design:

1. Load articles from CSV.
2. Split article text into token-based chunks.
3. Generate embeddings using `text-embedding-3-small`.
4. Store vectors in Pinecone.
5. Retrieve the most relevant chunks for a user question.
6. Send the retrieved context to `gpt-5-mini` with a strict grounding prompt.
7. Return the answer, retrieved context, and augmented prompt through an HTTP API.

---

## Key goals

- Answer **only** from the Medium dataset context.
- Avoid external knowledge and internet-based answering.
- Support transparent debugging and evaluation.
- Keep the system modular and easy to extend.
- Enable efficient experimentation with chunking and retrieval hyperparameters.
- Provide a public deployable API.

---

## System behavior

The assistant is intentionally constrained by a strict system prompt:

> You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: “I don’t know based on the provided Medium articles data.”
>
> Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.

This makes the application a **grounded retrieval system** rather than a general-purpose chatbot.

---

## Architecture

```text
CSV Dataset
   ↓
Token-based Chunking
   ↓
OpenAI Embeddings (text-embedding-3-small)
   ↓
Pinecone Vector Store
   ↓
Retriever (top-k)
   ↓
OpenAI Chat Model (gpt-5-mini)
   ↓
FastAPI Endpoints
   ↓
Vercel Deployment
```

### Main components

| Component | Responsibility |
|-----------|----------------|
| `medium_rag/config.py` | Central configuration for API keys and RAG hyperparameters |
| `medium_rag/ingest.py` | CSV loading, token-based chunking, embedding generation, Pinecone ingestion |
| `medium_rag/rag_chain.py` | Retrieval logic and grounded answer generation |
| `medium_rag/query.py` | Local manual testing of the RAG pipeline |
| `api/index.py` | FastAPI app exposing the required HTTP endpoints |
| `data/` | Local CSV dataset subset used during experimentation |

---

## Repository structure

```text
your-repo/
├── medium_rag/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── rag_chain.py
│   └── query.py
├── data/
│   └── medium_articles_small.csv
├── api/
│   └── index.py
├── .env
├── .gitignore
├── requirements.txt
├── runtime.txt
└── vercel.json
```

---

## Development approach

The project was developed in stages to reduce cost and improve reliability.

### 1. Local-first validation

The full RAG pipeline was implemented and tested locally before deployment:

- dataset loading
- chunking
- embedding generation
- Pinecone indexing
- retrieval
- grounded answer generation

This made debugging significantly easier before introducing API and deployment complexity.

### 2. Small-subset experimentation

Instead of embedding the entire corpus immediately, the system was first tested on a small subset of articles.

Benefits:

- faster debugging
- lower API cost
- easier iteration on chunking parameters
- simpler verification of end-to-end correctness

### 3. Reusable namespace strategy

Different chunking configurations can be stored in different Pinecone namespaces, for example:

- `tok400_ov50`
- `tok800_ov100`
- `tok1200_ov150`

This avoids re-embedding the full corpus every time retrieval settings are adjusted.

---

## Hyperparameters

The system focuses on three main RAG hyperparameters:

| Parameter | Meaning |
|-----------|---------|
| `chunk_size` | Size of each text chunk in tokens |
| `chunk_overlap` | Token overlap between neighboring chunks |
| `top_k` | Number of retrieved chunks passed into the answering step |

These values are centralized in `medium_rag/config.py` so that both ingestion and the live API stay synchronized.

### Why this matters

Keeping the parameters in one place ensures that:

- the ingestion namespace matches the configured chunking scheme
- the deployed `/api/stats` endpoint always reflects the current setup
- experiments are reproducible
- parameter changes do not silently break the system

---

## Batching and efficiency

The ingestion flow distinguishes between two separate ideas:

### 1. Chunk size

This controls how article text is split for retrieval.

### 2. Embedding batch size

This controls how many chunks are sent to the embedding model in a single API call.

These are **not** the same parameter.

The project uses batching for embedding generation to improve ingestion speed and reduce the overhead of sending one request per chunk.

---

## Supported question types

The system was designed around four practical query categories:

### 1. Precise fact retrieval
Example:

> Find an article that reframes marketing as a conversation with readers, aimed at writers who find self-promotion uncomfortable. Provide the title and author.

### 2. Multi-result topic listing
Example:

> List exactly 3 articles about education. Return only the titles.

### 3. Key idea summary extraction
Example:

> Find an article that argues past pandemics can spur innovation and recovery, and summarise its central argument.

### 4. Recommendation with evidence-based justification
Example:

> I want practical, beginner-friendly advice on building habits that actually stick. Which article would you recommend, and why?

These query types helped shape the retrieval logic, especially the need for:

- strong article-level precision
- support for up to 3 distinct article sources
- concise grounded summaries
- evidence-based recommendations

---

## API specification

The deployed system exposes two required HTTP endpoints.

### `POST /api/prompt`

Used to query the system.

#### Request body

```json
{
  "question": "natural language question here"
}
```

#### Response body

```json
{
  "response": "Final natural language answer from the model.",
  "context": [
    {
      "article_id": "1234",
      "title": "Sample article title",
      "chunk": "article chunk retrieved",
      "score": 0.1234
    }
  ],
  "Augmented_prompt": {
    "System": "the system prompt used to query the chat model",
    "User": "the user prompt used to query the chat model"
  }
}
```

### `GET /api/stats`

Returns the currently active RAG configuration.

#### Response body

```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.2,
  "top_k": 7
}
```

This endpoint is especially useful for reproducibility and evaluation.

---

## Local setup

### 1. Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

> On Windows PowerShell, use:
>
> ```powershell
> .venv\Scripts\Activate.ps1
> ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=medium-rag-index
```

### 5. Add your dataset

Place a CSV file in `data/`, for example:

```text
data/medium_articles_small.csv
```

---

## Running the project locally

### Ingest articles into Pinecone

```bash
python -m medium_rag.ingest
```

### Test a local query

```bash
python -m medium_rag.query
```

### Run the FastAPI app locally

```bash
uvicorn api.index:app --reload
```

Then open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/api/stats`

---

## Deployment

The API is deployed with **Vercel** using the Python runtime for FastAPI.

### Required deployment files

- `api/index.py`
- `requirements.txt`
- `runtime.txt`
- `vercel.json`

### Example `runtime.txt`

```text
3.11.9
```

### Example `vercel.json`

```json
{
  "version": 2,
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/index.py"
    }
  ]
}
```

### Required environment variables on Vercel

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`

---

## Important implementation notes

### Grounding before generation

The system is intentionally designed so that retrieval happens first and generation happens second. The language model does not answer freely; it answers only from retrieved context.

### Transparent outputs

The response returns:

- the final answer
- the retrieved supporting chunks
- the full augmented prompt used for generation

This makes the system easier to inspect, debug, and evaluate.

### Configuration consistency

A single configuration source is used so that ingestion parameters and deployment parameters remain aligned.

### Local-first engineering

Building locally before deployment reduced debugging friction and made the system easier to reason about.

---

## Future improvements

Possible next steps include:

- article-level deduplication during retrieval
- article reranking across multiple retrieved chunks
- evaluation scripts for comparing chunking configurations
- automated scoring for retrieval quality
- support for larger corpus ingestion with stronger batching control
- frontend UI for interactive question answering

---

## Tech stack

- **Python**
- **LangChain**
- **OpenAI API**
- **text-embedding-3-small**
- **gpt-5-mini**
- **Pinecone**
- **FastAPI**
- **Vercel**
- **Pandas**
- **tiktoken**

---

## Project highlights

- Closed-domain RAG over a Medium dataset
- Strict grounded-answering behavior
- Token-based chunking
- Configurable retrieval pipeline
- Transparent API responses
- Modular Python project structure
- Local validation before cloud deployment
- Public FastAPI deployment on Vercel

---

## License

Add your preferred license here, for example:

```text
MIT License
```

---

## Acknowledgments

This project was developed as part of a RAG-focused academic / practical implementation workflow, with emphasis on reproducibility, transparent retrieval, and deployment-ready API design.
