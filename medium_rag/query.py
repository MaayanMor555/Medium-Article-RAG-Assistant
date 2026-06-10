from .rag_chain import answer_query

def main():
    namespace = "chunk_size_tokens512_overlap102"  # same as in ingest.py
    question = "Find an article that reframes marketing as a conversation with readers," \
               "aimed at writers who find self-promotion uncomfortable. Provide the title and author."

    answer, docs = answer_query(question, namespace=namespace, top_k=3, return_scores=True)

    print("ANSWER:\n")
    print(answer)
    print("\nRETRIEVED TITLES:")
    for d in docs:
        print("-", d.metadata.get("title"))

if __name__ == "__main__":
    main()