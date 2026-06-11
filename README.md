# Medium Article RAG Assistant

A Retrieval-Augmented Generation (**RAG**) system for querying a Medium articles dataset using **LangChain**, **OpenAI embeddings**, **Pinecone**, **FastAPI**, and **Vercel**.

The assistant answers questions **strictly from the provided article corpus** and returns transparent retrieval evidence, including the retrieved context and the augmented prompt sent to the chat model.

---

## Live endpoints

> [`/api/prompt`](https://medium-article-rag-assistant-hazel.vercel.app/api/prompt)  
> [`/api/stats`](https://medium-article-rag-assistant-hazel.vercel.app/api/stats)  

Or alternatively: 
> [`/docs`](https://medium-article-rag-assistant-hazel.vercel.app/docs)

---

## Overview

This project was built to support question answering over a structured Medium dataset (stored in CSV format).

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

### Reusable namespace strategy

Different chunking configurations can be stored in different Pinecone namespaces, for example:

- `chunk_size_tokens512_overlap51`
- `chunk_size_tokens512_overlap102`
- `chunk_size_tokens300_overlap50`

This avoids re-embedding the full corpus every time retrieval settings are adjusted.

---

## Hyperparameters

The system focuses on three main RAG hyperparameters:

| Parameter | Meaning | Default for this project |
|-----------|---------|--------------------------|
| `chunk_size` | Size of each text chunk in tokens | 512 |
| `chunk_overlap` | Token overlap between neighboring chunks | 102 (20%) |
| `top_k` | Number of retrieved chunks passed into the answering step | 12 |

These values are centralized in `medium_rag/config.py` so that both ingestion and the live API stay synchronized.

---

## Supported question types:

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
- support (typically) for up to 3 distinct article sources
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
  "top_k": 12
}
```

---

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/MaayanMor555/Medium-Article-RAG-Assistant.git
cd Medium-Article-RAG-Assistant
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
data/medium-english-50mb.csv
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

### Required environment variables on Vercel

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`

## Acknowledgments

This project was developed as part of a RAG-focused academic / practical implementation workflow, with emphasis on reproducibility, transparent retrieval, and deployment-ready API design.
