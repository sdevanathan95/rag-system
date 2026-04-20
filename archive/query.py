"""
RAG QUERY PIPELINE
Retrieves relevant chunks and generates answers using LLM
"""

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_DB_DIR = "./chroma_db"
OLLAMA_URL = "http://localhost:11434"

def load_vector_store() -> Chroma:
    """Load existing vector store from disk."""
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    return vectorstore

def create_qa_chain(vectorstore: Chroma):
    """Create the RAG chain."""
    llm = OllamaLLM(
        model="gemma3:4b",
        base_url=OLLAMA_URL,
        temperature=0.1
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k":6, "fetch_k": 20}
    )

    template = """You are a helpful assistant answering questions based on the provided context.

                    CONTEXT:
                    {context}

                    QUESTION: {question}

                    INSTRUCTIONS:
                    - Answer based ONLY on the context above
                    - If the context doesn't contain enough information, say so
                    - Be concise but thorough

                    ANSWER:
                """
    
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def main():
    print("Loading vector store...")
    vectorstore = load_vector_store()

    print("Creating QA Chain...")
    chain, retriever = create_qa_chain(vectorstore)

    print("\n" + "=" * 50)
    print("RAG System Ready! Ask questions about your documents.")
    print("Type 'quit' to exit")
    print("=" * 50 + "\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() == 'quit':
            break
        if not question:
            continue

        print("\n Searching... \n")

        docs = retriever.invoke(question)
        answer = chain.invoke(question)

        print("Answer:")
        print("-" * 40)
        print(answer)

        print("\n Sources:")
        print("-" * 40)
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            print(f"{i}. {source}")
        print()

if __name__ == "__main__":
    main()
