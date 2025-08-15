import os
import gradio as gr
import logging
from typing import List, Tuple, Any
from pathlib import Path
import shutil
import tempfile
from datetime import datetime

from rag_system import RAGSystem, RAGConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/gradio_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGInterface:
    def __init__(self):
        # Initialize RAG system
        self.config = RAGConfig(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name="gradio_documents",
            k_documents=5,
            chunk_size=1000,
            chunk_overlap=200
        )
        self.rag_system = RAGSystem(self.config)
        self.uploaded_files_dir = Path("/app/uploaded_files")
        self.uploaded_files_dir.mkdir(exist_ok=True)
        
        # Track uploaded documents
        self.document_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "last_update": None
        }
    
    def upload_documents(self, files: List[Any], metadata_source: str = "", metadata_department: str = "") -> str:
        """Handle document upload and processing"""
        if not files:
            return "❌ No files uploaded. Please select files to upload."
        
        try:
            success_count = 0
            total_files = len(files)
            
            # Prepare metadata
            metadata = {
                "upload_timestamp": datetime.now().isoformat(),
                "source": metadata_source or "gradio_upload",
                "department": metadata_department or "general"
            }
            
            for file in files:
                if file is None:
                    continue
                    
                # Save uploaded file
                file_path = self.uploaded_files_dir / Path(file.name).name
                shutil.copy2(file.name, file_path)
                
                # Add to RAG system
                success = self.rag_system.add_documents(
                    str(file_path),
                    metadata=metadata
                )
                
                if success:
                    success_count += 1
                    logger.info(f"Successfully processed: {file.name}")
                else:
                    logger.error(f"Failed to process: {file.name}")
            
            # Update stats
            self.document_stats["total_documents"] += success_count
            self.document_stats["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            result_msg = f"✅ Successfully processed {success_count}/{total_files} documents."
            if success_count < total_files:
                result_msg += f" {total_files - success_count} files failed to process."
            
            return result_msg
            
        except Exception as e:
            logger.error(f"Error uploading documents: {str(e)}")
            return f"❌ Error uploading documents: {str(e)}"
    
    def query_documents(self, question: str, use_compression: bool = True, k_docs: int = 3) -> Tuple[str, str]:
        """Query the RAG system and return answer with sources"""
        if not question.strip():
            return "❌ Please enter a question.", ""
        
        try:
            # Update config for this query
            self.rag_system.config.k_documents = k_docs
            
            # Query the system
            response = self.rag_system.query(question, use_compression=use_compression)
            
            answer = response["answer"]
            
            # Format source documents
            sources = ""
            if response["source_documents"]:
                sources = "## 📚 Source Documents:\n\n"
                for i, (doc, metadata) in enumerate(zip(response["source_documents"], response["metadata"])):
                    sources += f"**Source {i+1}:**\n"
                    sources += f"- **Content:** {doc.page_content[:200]}...\n"
                    sources += f"- **Source:** {metadata.get('source', 'Unknown')}\n"
                    sources += f"- **Department:** {metadata.get('department', 'Unknown')}\n"
                    sources += f"- **Chunk ID:** {metadata.get('chunk_id', 'N/A')}\n"
                    sources += "---\n\n"
            else:
                sources = "No source documents found."
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            return f"❌ Error processing query: {str(e)}", ""
    
    def get_system_stats(self) -> str:
        """Get system statistics"""
        try:
            # Try to get collection info from Qdrant
            collection_info = ""
            try:
                client = self.rag_system.vector_store.client
                info = client.get_collection(self.config.collection_name)
                collection_info = f"- **Vector Count:** {info.vectors_count}\n"
                collection_info += f"- **Collection Status:** {info.status}\n"
            except:
                collection_info = "- **Vector Count:** Unable to fetch\n"
            
            stats = f"""
## 📊 System Statistics

### Document Statistics
- **Total Documents Processed:** {self.document_stats['total_documents']}
- **Last Update:** {self.document_stats['last_update'] or 'Never'}

### Vector Store Statistics
{collection_info}

### Configuration
- **Collection Name:** {self.config.collection_name}
- **Chunk Size:** {self.config.chunk_size}
- **Chunk Overlap:** {self.config.chunk_overlap}
- **Embedding Model:** {self.config.embedding_model}
- **LLM Model:** {self.config.llm_model}
            """
            return stats.strip()
            
        except Exception as e:
            return f"❌ Error fetching statistics: {str(e)}"
    
    def search_similar_documents(self, query: str, k: int = 5) -> str:
        """Search for similar documents without generating an answer"""
        if not query.strip():
            return "❌ Please enter a search query."
        
        try:
            docs = self.rag_system.search_similar_documents(query, k=k)
            
            if not docs:
                return "No similar documents found."
            
            result = f"## 🔍 Found {len(docs)} similar documents:\n\n"
            
            for i, doc in enumerate(docs):
                result += f"**Document {i+1}:**\n"
                result += f"- **Content:** {doc.page_content[:300]}...\n"
                result += f"- **Source:** {doc.metadata.get('source', 'Unknown')}\n"
                result += f"- **Department:** {doc.metadata.get('department', 'Unknown')}\n"
                result += "---\n\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return f"❌ Error searching documents: {str(e)}"


def create_gradio_interface():
    """Create the Gradio interface"""
    rag_interface = RAGInterface()
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    
    .stats-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=css, title="🤖 RAG System with Qdrant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 RAG System with LangChain & Qdrant
        
        Welcome to your Retrieval-Augmented Generation system! Upload documents and ask questions to get AI-powered answers with source citations.
        """)
        
        with gr.Tab("📤 Upload Documents"):
            gr.Markdown("### Upload your documents to build the knowledge base")
            
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Select Documents",
                        file_count="multiple",
                        file_types=[".txt", ".pdf", ".md", ".docx"]
                    )
                    
                with gr.Column(scale=1):
                    metadata_source = gr.Textbox(
                        label="Source (optional)",
                        placeholder="e.g., company_docs, research_papers",
                        value=""
                    )
                    metadata_department = gr.Textbox(
                        label="Department (optional)",
                        placeholder="e.g., engineering, marketing",
                        value=""
                    )
            
            upload_btn = gr.Button("🚀 Upload & Process Documents", variant="primary", size="lg")
            upload_output = gr.Textbox(label="Upload Status", lines=3)
            
            upload_btn.click(
                fn=rag_interface.upload_documents,
                inputs=[file_input, metadata_source, metadata_department],
                outputs=upload_output
            )
        
        with gr.Tab("💬 Ask Questions"):
            gr.Markdown("### Ask questions about your uploaded documents")
            
            with gr.Row():
                with gr.Column(scale=3):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What would you like to know about your documents?",
                        lines=2
                    )
                    
                with gr.Column(scale=1):
                    use_compression = gr.Checkbox(
                        label="Use Contextual Compression",
                        value=True,
                        info="Improves answer quality but slower"
                    )
                    k_docs = gr.Slider(
                        label="Number of Documents to Retrieve",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1
                    )
            
            ask_btn = gr.Button("🔍 Ask Question", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    answer_output = gr.Textbox(
                        label="📋 Answer",
                        lines=8,
                        show_copy_button=True
                    )
                
                with gr.Column():
                    sources_output = gr.Markdown(
                        label="📚 Sources",
                        value="Sources will appear here..."
                    )
            
            ask_btn.click(
                fn=rag_interface.query_documents,
                inputs=[question_input, use_compression, k_docs],
                outputs=[answer_output, sources_output]
            )
            
            # Example questions
            gr.Markdown("### 💡 Example Questions:")
            example_questions = [
                "What are the main topics covered in the documents?",
                "Can you summarize the key findings?",
                "What are the recommendations mentioned?",
                "Who are the main stakeholders discussed?"
            ]
            
            for question in example_questions:
                gr.Button(question, size="sm").click(
                    lambda q=question: q,
                    outputs=question_input
                )
        
        with gr.Tab("🔍 Search Documents"):
            gr.Markdown("### Search for similar documents without generating answers")
            
            with gr.Row():
                search_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter keywords to find similar documents",
                    scale=3
                )
                search_k = gr.Slider(
                    label="Number of Results",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    scale=1
                )
            
            search_btn = gr.Button("🔍 Search", variant="primary")
            search_output = gr.Markdown(label="Search Results")
            
            search_btn.click(
                fn=rag_interface.search_similar_documents,
                inputs=[search_input, search_k],
                outputs=search_output
            )
        
        with gr.Tab("📊 System Stats"):
            gr.Markdown("### System Information and Statistics")
            
            stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
            stats_output = gr.Markdown(label="System Statistics")
            
            stats_btn.click(
                fn=rag_interface.get_system_stats,
                outputs=stats_output
            )
            
            # Auto-load stats on tab open
            demo.load(
                fn=rag_interface.get_system_stats,
                outputs=stats_output
            )
        
        gr.Markdown("""
        ---
        ### 📝 Tips for Best Results:
        - Upload relevant documents in supported formats (PDF, TXT, MD, DOCX)
        - Ask specific questions for more accurate answers
        - Use metadata fields to organize your documents
        - Enable contextual compression for better answer quality
        """)
    
    return demo

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found in environment variables")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    # Launch with configuration for Docker
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )