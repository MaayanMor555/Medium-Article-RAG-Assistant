import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medium-rag-index")

# ─── RAG Hyperparameters ────────────────────────────────────────────
# Change values HERE only. Both ingest.py and api/index.py read from here.
CHUNK_SIZE    = 512    # tokens, should be maximum 1024.
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.1)    # tokens
TOP_K         = 7      # number of retrieved chunks, should be between 1 and 30.

# Derived values:
OVERLAP_RATIO = CHUNK_OVERLAP / CHUNK_SIZE # should be maximum 30%.
NAMESPACE     = f"chunk_size_tokens{CHUNK_SIZE}_overlap{CHUNK_OVERLAP}"