"""
generation pipeline
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.config import OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODEL, CHROMA_PATH

class GenerationPipeline:

    def __init__(self):
        self.llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        self.vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
        self.rephrase_prompt = ChatPromptTemplate.from_template(
            "Generate 3 different search queries for this question, one per line, no numbering:\n{question}"
        )
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions based on the provided context.
        Use ONLY the context below to answer the question. If you cannot find the answer
        in the context, say so clearly.

        Context:
        {context}

        Question: {question}

        Answer:""")

    def query_rag(self, question: str, selected_docs: list = []):
        """Query the RAG system - returns (answer, source_documents)"""
        search_kwargs = {"k": 6, "fetch_k": 20}
        if selected_docs:
            if len(selected_docs) == 1:
                search_kwargs["filter"] = {"source": selected_docs[0]}
            else:
                search_kwargs["filter"] = {"source": {"$in": selected_docs}}
        retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

        queries_text = (self.rephrase_prompt | self.llm | StrOutputParser()).invoke({"question": question})
        alt_queries = [q.strip() for q in queries_text.strip().split("\n") if q.strip()]

        all_docs = retriever.invoke(question)
        seen = {d.page_content for d in all_docs}
        for q in alt_queries[:3]:
            for doc in retriever.invoke(q):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    all_docs.append(doc)

        context = self.format_docs(all_docs)
        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return answer, all_docs
    
    def format_docs(self, docs):
        """Format retrieved documents for the prompt"""
        return "\n\n".join(doc.page_content for doc in docs)
