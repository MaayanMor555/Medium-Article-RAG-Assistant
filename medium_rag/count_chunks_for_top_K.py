import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter

from .config import CHUNK_SIZE, CHUNK_OVERLAP

CSV_PATH = "/Users/maayanmor/Desktop/קורסים תואר שני טכניון/מערכות סוכני בינה מלאכותית/individual_assignment/data/medium-english-50mb.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks_per_article = []

    for _, row in df.iterrows():
        text = f"Title: {row['title']}\n\n{row['text']}"
        chunks = splitter.split_text(text)
        chunks_per_article.append(len(chunks))

    total_chunks = sum(chunks_per_article)
    max_chunks   = max(chunks_per_article)
    min_chunks   = min(chunks_per_article)
    avg_chunks   = total_chunks / len(chunks_per_article)

    print(f"Articles         : {len(chunks_per_article)}")
    print(f"Total chunks     : {total_chunks}")
    print(f"Max chunks/article: {max_chunks}")
    print(f"Min chunks/article: {min_chunks}")
    print(f"Avg chunks/article: {avg_chunks:.1f}")
    print(f"\nChunk size used  : {CHUNK_SIZE} tokens")
    print(f"Overlap used     : {CHUNK_OVERLAP} tokens")

    # Distribution: how many articles have N chunks
    print("\nDistribution (chunks → number of articles):")
    dist = Counter(chunks_per_article)
    for n_chunks in sorted(dist.keys()):
        print(f"  {n_chunks:3d} chunks: {dist[n_chunks]} articles")

if __name__ == "__main__":
    main()