import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

from .config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# One chunk config to start with
from .config import (
    CHUNK_SIZE, CHUNK_OVERLAP, NAMESPACE
)

def build_docs_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    docs = []
    for _, row in df.iterrows():
        text = f"Title: {row['title']}\n\n{row['text']}"
        meta = {
            "title": row["title"],
            "url": row["url"],
            "authors": row["authors"],
            "timestamp": row["timestamp"],
            "tags": row["tags"],
        }
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)

def ensure_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # Create index if not exists
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # text-embedding-3-small dim.
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
    return pc.Index(PINECONE_INDEX_NAME)

def main():
    csv_path = "/Users/maayanmor/Desktop/קורסים תואר שני טכניון/מערכות סוכני בינה מלאכותית/individual_assignment/data/medium-english-50mb-small.csv"

    print("Loading CSV...")
    docs = build_docs_from_csv(csv_path)
    print(f"Loaded {len(docs)} docs")

    print("Chunking...")
    chunks = chunk_docs(docs)
    print(f"Produced {len(chunks)} chunks")

    print("Ensuring Pinecone index...")
    index = ensure_index()

    print("Creating embeddings + upserting...")
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url="https://api.llmod.ai/v1",
        model="4UHRUIN-text-embedding-3-small"
    )

    PineconeVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=NAMESPACE,
    )

    print(f"Done. Namespace={NAMESPACE}")

if __name__ == "__main__":
    main()