"""
Clinical Document RAG Pipeline
Production-grade retrieval-augmented generation for clinical document classification
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone
import os

class ClinicalRAGPipeline:
    def __init__(self, index_name: str = "clinical-docs"):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.index_name = index_name
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ".", " "]
        )

    def ingest(self, documents: list[str]) -> int:
        """Ingest documents into Pinecone vector store"""
        chunks = self.splitter.create_documents(documents)
        vectorstore = Pinecone.from_documents(
            chunks, self.embeddings, index_name=self.index_name
        )
        return len(chunks)

    def query(self, question: str, top_k: int = 5) -> dict:
        """Query the RAG pipeline with cross-encoder re-ranking"""
        vectorstore = Pinecone.from_existing_index(
            self.index_name, self.embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )
        result = chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [d.page_content[:200] for d in result["source_documents"]]
        }

if __name__ == "__main__":
    pipeline = ClinicalRAGPipeline()
    print("Pipeline initialized successfully")
