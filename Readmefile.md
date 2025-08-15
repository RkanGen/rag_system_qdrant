# 🤖 RAG System with LangChain, Qdrant & Gradio

A production-ready Retrieval-Augmented Generation (RAG) system built with LangChain, Qdrant vector database, and a beautiful Gradio web interface.

## 🌟 Features

- **📄 Multi-format Document Support**: PDF, TXT, MD, DOCX files
- **🔍 Advanced Search**: Similarity search with configurable parameters
- **🎯 Contextual Compression**: Improved answer quality with LLM-based compression
- **📊 Rich Metadata**: Track document sources, departments, and timestamps
- **🖥️ Web Interface**: Beautiful Gradio-based UI for easy interaction
- **🐳 Docker Ready**: Complete containerization with Docker Compose
- **⚡ Production Ready**: Logging, health checks, and error handling

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Gradio UI     │    │  RAG System  │    │  Qdrant Vector │
│  (Port 7860)    │◄──►│  (LangChain) │◄──►│  DB (Port 6333) │
└─────────────────┘    └──────────────┘    └─────────────────┘
        │                       │                     │
        │              ┌────────▼────────┐           │
        │              │  Document       │           │
        └─────────────►│  Processing     │───────────┘
                       │  Pipeline       │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API Key
- At least 4GB RAM recommended

### 1. Clone and Setup



### 2. Environment Configuration

```bash
# Copy the environment template
cp .env.example .env

# Edit .env file and add your OpenAI API key
nano .env
```

Add your OpenAI API key to the `.env` file:
```env
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

### 3. Launch with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### 4. Access the Application

- **Web Interface**: http://localhost:7860
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📁 Project Structure

```
rag-system/
├── rag_system.py          # Core RAG implementation
├── gradio_app.py          # Gradio web interface
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Multi-service orchestration
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── README.md             # This file
├── documents/            # Document storage (volume mount)
├── logs/                # Application logs (volume mount)
└── uploaded_files/      # Temporary upload storage
```

## 🛠️ Usage Guide

### Upload Documents

1. Navigate to the **"📤 Upload Documents"** tab
2. Select multiple files (PDF, TXT, MD, DOCX)
3. Optionally add metadata (source, department)
4. Click **"🚀 Upload & Process Documents"**

### Ask Questions

1. Go to the **"💬 Ask Questions"** tab
2. Type your question in the text box
3. Configure settings:
   - **Contextual Compression**: Better quality, slower processing
   - **Number of Documents**: How many sources to retrieve
4. Click **"🔍 Ask Question"**

### Search Documents

1. Visit the **"🔍 Search Documents"** tab
2. Enter keywords to find similar content
3. Adjust the number of results
4. View matching documents with metadata

### Monitor System

1. Check the **"📊 System Stats"** tab for:
   - Document count and processing status
   - Vector database statistics
   - System configuration
   - Performance metrics

## ⚙️ Configuration

### RAG System Configuration

Edit `rag_system.py` to modify:

```python
@dataclass
class RAGConfig:
    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "documents"
    vector_size: int = 1536
    
    # Text processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval settings
    k_documents: int = 5
    score_threshold: float = 0.7
    
    # Model settings
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-3.5-turbo"
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | None |
| `QDRANT_URL` | Qdrant server URL | `http://qdrant:6333` |
| `GRADIO_SERVER_NAME` | Gradio bind address | `0.0.0.0` |
| `GRADIO_SERVER_PORT` | Gradio port | `7860` |
| `LOG_LEVEL` | Logging level | `INFO` |

## 🔧 Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
# or
rag_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Start Qdrant locally
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Run the application
python gradio_app.py
```

### Adding New Document Types

To support additional file formats, modify the `DocumentProcessor` class in `rag_system.py`:

```python
def load_documents(self, file_path: str, file_type: str = "auto"):
    # Add new file type handling
    if file_type == "your_new_type":
        loader = YourCustomLoader(file_path)
    # ...
```

### Custom Retrievers

Implement custom retrieval strategies by extending the `QdrantVectorStore` class:

```python
def create_custom_retriever(self):
    # Your custom retrieval logic
    pass
```

## 📊 Monitoring & Logging

### Application Logs

```bash
# View real-time logs
docker-compose logs -f rag_system

# View logs for specific time period
docker-compose logs --since="1h" rag_system
```

### Health Checks

The system includes built-in health checks:
- **Qdrant**: `http://localhost:6333/health`
- **RAG System**: `http://localhost:7860` (Gradio interface)

### Performance Monitoring

Monitor system performance through:
1. Gradio interface stats tab
2. Qdrant dashboard metrics
3. Docker container resources: `docker stats`

## 🔒 Security Considerations

### Production Deployment

1. **API Key Security**: Use Docker secrets or external secret management
2. **Network Security**: Configure firewalls and VPNs
3. **Data Encryption**: Enable TLS/SSL for external access
4. **Access Control**: Implement authentication for sensitive deployments

### Secure Environment Setup

```bash
# Use Docker secrets for production
echo "your-api-key" | docker secret create openai_api_key -

# Update docker-compose.yml to use secrets
secrets:
  - openai_api_key
```

## 🚨 Troubleshooting

### Common Issues

1. **Connection Refused Error**
   ```bash
   # Check if Qdrant is running
   curl http://localhost:6333/health
   
   # Restart services
   docker-compose restart
   ```

2. **Out of Memory Error**
   ```bash
   # Increase Docker memory limit
   # Check Docker Desktop settings or system resources
   ```

3. **API Key Issues**
   ```bash
   # Verify API key in .env file
   cat .env | grep OPENAI_API_KEY
   
   # Restart container after updating
   docker-compose restart rag_system
   ```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Add to docker-compose.yml environment
- LOG_LEVEL=DEBUG

# Or run locally with debug
python gradio_app.py --debug
```

## 📈 Performance Tuning

### Optimization Tips

1. **Chunk Size**: Balance between context and processing speed
2. **Vector Dimensions**: Use appropriate embedding model
3. **Retrieval Count**: Optimize k_documents for your use case
4. **Compression**: Disable for faster responses if quality is sufficient

### Scaling Considerations

- **Horizontal Scaling**: Use multiple Qdrant nodes
- **Load Balancing**: Deploy multiple RAG system instances
- **Caching**: Implement Redis for frequent queries
- **Storage**: Use persistent volumes for production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


