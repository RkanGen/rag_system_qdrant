import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    DirectoryLoader
)
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import qdrant_client.http.models as rest

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for the RAG system"""
    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "documents"
    vector_size: int = 1536  # OpenAI embedding dimension
    
    # Text processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval settings
    k_documents: int = 5
    score_threshold: float = 0.7
    
    # Model settings
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.1


class DocumentProcessor:
    """Handles document loading and text splitting"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
    
    def load_documents(self, file_path: str, file_type: str = "auto") -> List[Document]:
        """Load documents from various file types"""
        try:
            if file_type == "auto":
                file_type = self._detect_file_type(file_path)
            
            if file_type == "txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_type == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_type == "directory":
                loader = DirectoryLoader(
                    file_path, 
                    glob="**/*.{txt,pdf,md}",
                    loader_cls=TextLoader
                )
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return []
    
    def process_documents(self, documents: List[Document], metadata: Dict[str, Any] = None) -> List[Document]:
        """Split documents into chunks and add metadata"""
        processed_docs = []
        
        for doc in documents:
            # Split document into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                # Preserve original metadata
                chunk_metadata = chunk.metadata.copy()
                
                # Add custom metadata
                if metadata:
                    chunk_metadata.update(metadata)
                
                # Add chunk-specific metadata
                chunk_metadata.update({
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content)
                })
                
                chunk.metadata = chunk_metadata
                processed_docs.append(chunk)
        
        logger.info(f"Processed {len(processed_docs)} document chunks")
        return processed_docs
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type based on extension"""
        if os.path.isdir(file_path):
            return "directory"
        
        extension = os.path.splitext(file_path)[1].lower()
        type_mapping = {
            ".txt": "txt",
            ".md": "txt",
            ".pdf": "pdf"
        }
        return type_mapping.get(extension, "txt")


class QdrantVectorStore:
    """Manages Qdrant vector store operations"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = QdrantClient(url=config.qdrant_url)
        self.embeddings = OpenAIEmbeddings(model=config.embedding_model)
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.config.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.config.collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        try:
            # Create Qdrant vector store instance
            vector_store = Qdrant(
                client=self.client,
                collection_name=self.config.collection_name,
                embeddings=self.embeddings
            )
            
            # Add documents
            vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def create_retriever(self, search_type: str = "similarity_score_threshold"):
        """Create a retriever from the vector store"""
        vector_store = Qdrant(
            client=self.client,
            collection_name=self.config.collection_name,
            embeddings=self.embeddings
        )
        
        search_kwargs = {
            "k": self.config.k_documents,
        }
        
        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = self.config.score_threshold
        
        return vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def search_documents(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents"""
        k = k or self.config.k_documents
        
        vector_store = Qdrant(
            client=self.client,
            collection_name=self.config.collection_name,
            embeddings=self.embeddings
        )
        
        return vector_store.similarity_search(query, k=k)


class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.doc_processor = DocumentProcessor(self.config)
        self.vector_store = QdrantVectorStore(self.config)
        self.llm = ChatOpenAI(
            model_name=self.config.llm_model,
            temperature=self.config.temperature
        )
        self._setup_qa_chain()
    
    def _setup_qa_chain(self):
        """Set up the QA chain with custom prompt"""
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}

        Answer: """
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
    
    def add_documents(self, file_path: str, file_type: str = "auto", metadata: Dict[str, Any] = None) -> bool:
        """Add documents to the RAG system"""
        # Load documents
        documents = self.doc_processor.load_documents(file_path, file_type)
        if not documents:
            return False
        
        # Process documents (split and add metadata)
        processed_docs = self.doc_processor.process_documents(documents, metadata)
        
        # Add to vector store
        return self.vector_store.add_documents(processed_docs)
    
    def create_qa_chain(self, use_compression: bool = True):
        """Create the QA chain with optional compression"""
        # Get base retriever
        base_retriever = self.vector_store.create_retriever()
        
        if use_compression:
            # Use contextual compression for better results
            compressor = LLMChainExtractor.from_llm(self.llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        else:
            retriever = base_retriever
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        
        return qa_chain
    
    def query(self, question: str, use_compression: bool = True) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            qa_chain = self.create_qa_chain(use_compression)
            result = qa_chain({"query": question})
            
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"],
                "metadata": [doc.metadata for doc in result["source_documents"]]
            }
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return {
                "answer": "Sorry, I encountered an error while processing your question.",
                "source_documents": [],
                "metadata": []
            }
    
    def search_similar_documents(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents without generating an answer"""
        return self.vector_store.search_documents(query, k)


# Example usage and testing
def main():
    """Example usage of the RAG system"""
    # Set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    
    # Initialize RAG system
    config = RAGConfig(
        collection_name="my_documents",
        k_documents=3,
        chunk_size=800,
        chunk_overlap=100
    )
    
    rag = RAGSystem(config)
    
    # Add documents with metadata
    success = rag.add_documents(
        file_path="path/to/your/documents",  # Replace with actual path
        file_type="directory",
        metadata={
            "source": "company_docs",
            "department": "engineering",
            "date_added": "2024-01-15"
        }
    )
    
    if success:
        print("Documents added successfully!")
        
        # Query the system
        response = rag.query(
            "What are the main features of the product?",
            use_compression=True
        )
        
        print("\n" + "="*50)
        print("ANSWER:")
        print(response["answer"])
        
        print("\n" + "="*50)
        print("SOURCE DOCUMENTS:")
        for i, doc in enumerate(response["source_documents"]):
            print(f"\nDocument {i+1}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
    
    else:
        print("Failed to add documents")


if __name__ == "__main__":
    main()