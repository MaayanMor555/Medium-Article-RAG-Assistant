from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.messages import SystemMessage, HumanMessage

from .config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

SYSTEM_PROMPT = """
You are a Medium-article assistant that answers questions strictly and only
based on the Medium articles dataset context provided to you (metadata
and article passages). You must not use any external knowledge, the open
internet, or information that is not explicitly contained in the retrieved context. 
If the answer cannot be determined from the provided context, respond: 
“I don’t know based on the provided Medium articles data.”

Always explain your answer using the given context, quoting or
paraphrasing the relevant article passage or metadata when helpful.
"""

def get_llm():
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://api.llmod.ai/v1",
        model="4UHRUIN-gpt-5-mini",
        #temperature=0,   # 0 = deterministic/consistent output, 1 = more creative/random. So 0 is better
                         # for factual/structured tasks.
    )

def get_retriever(namespace: str, top_k: int = 3):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url="https://api.llmod.ai/v1",
        model="4UHRUIN-text-embedding-3-small"
    )

    vs = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace,
    )
    return vs.as_retriever(search_kwargs={"k": top_k})

def build_context(docs):
    blocks = []
    for d in docs:
        m = d.metadata
        block = (
            f"Title: {m.get('title')}\n"
            f"URL: {m.get('url')}\n"
            f"Authors: {m.get('authors')}\n"
            f"Timestamp: {m.get('timestamp')}\n"
            f"Tags: {m.get('tags')}\n\n"
            f"Passage:\n{d.page_content}"
        )
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)

def answer_query(query: str, namespace: str, top_k: int = 3, return_scores: bool = False):
    llm = get_llm()
    retriever = get_retriever(namespace, top_k=top_k)
    if return_scores:
        vs = retriever.vectorstore
        docs_and_scores = vs.similarity_search_with_score(query, k=top_k)
        docs = [ds[0] for ds in docs_and_scores]
    else:
        docs = retriever.invoke(query)
        docs_and_scores = None
    context = build_context(docs)

    user_prompt = (
        f"Here are retrieved passages from Medium articles:\n\n"
        f"{context}\n\n"
        f"User question: {query}\n\n"
        f"Answer the question following the system instructions."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        resp = llm.invoke(messages)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e

    return resp.content, docs_and_scores