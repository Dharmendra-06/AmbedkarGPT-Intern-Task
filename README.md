#  AmbedkarGPT - RAG Q&A System

A command-line Retrieval-Augmented Generation (RAG) system that answers questions based on Dr. B.R. Ambedkar's speech excerpt from "Annihilation of Caste". Built using LangChain, ChromaDB, HuggingFace Embeddings, and Ollama with Mistral 7B.

## 📋 Overview

This project implements a complete RAG pipeline that:
1. **Loads** text from Dr. B.R. Ambedkar's speech
2. **Splits** the text into manageable chunks
3. **Creates embeddings** using HuggingFace's sentence-transformers
4. **Stores** embeddings in a local ChromaDB vector database
5. **Retrieves** relevant context based on user questions
6. **Generates** answers using Ollama's Mistral 7B LLM

## 🛠️ Technical Stack

- **Python**: 3.8+
- **Framework**: LangChain
- **Vector Database**: ChromaDB (local, persistent)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Ollama with Mistral 7B
- **Dependencies**: See `requirements.txt`

## 🚀 Installation & Setup

### Prerequisites

1. **Python 3.8 or higher** installed on your system
2. **Ollama** installed with Mistral 7B model

### Step 1: Install Ollama

**For Linux/Mac:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**For Windows:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

**Pull the Mistral model:**
```bash
ollama pull mistral
```

**Verify Ollama is running:**
```bash
ollama run mistral
# Type a test message, then exit with /bye
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/Dharmendra-06/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### Step 3: Set Up Python Environment

**Create a virtual environment:**
```bash
# Using venv
python3 -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

**Or using conda:**
```bash
conda create -n ambedkar-rag python=3.10
conda activate ambedkar-rag
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First-time installation may take several minutes as it downloads the embedding model and dependencies.

## 🎯 Usage

### Run the Q&A System

```bash
python3 main.py
```

### First Run

On the first run, the system will:
1. Load `speech.txt`
2. Split the text into chunks
3. Download the embedding model (if not cached)
4. Create embeddings and store them in `./chroma_db/`
5. Set up the QA chain

This process takes 1-3 minutes depending on your system.

### Subsequent Runs

The system will load the existing vector store from `./chroma_db/`, making startup much faster (5-10 seconds).

### Example Interaction

```
 Welcome to AmbedkarGPT - RAG Q&A System
============================================================

📚 Loading speech text...
✅ Loaded 1 document(s)
✂️  Splitting text into chunks...
✅ Created 1 text chunks
🧠 Creating embeddings...
✅ Vector store created successfully!

🤖 Setting up Ollama LLM...
🔗 Creating RetrievalQA chain...
✅ QA chain ready!

============================================================
🎯 System is ready! You can now ask questions.
============================================================
💡 Tip: Ask questions about the speech content
📝 Type 'exit' or 'quit' to stop

Your question: What is the real remedy according to Dr. Ambedkar?

❓ Question: What is the real remedy according to Dr. Ambedkar?
🔍 Retrieving relevant context and generating answer...

💡 Answer: According to Dr. Ambedkar, the real remedy is to destroy the belief in the sanctity of the shastras...

📄 Source chunks used:
[Chunk 1]
The real remedy is to destroy the belief in the sanctity of the shastras...
```

### Sample Questions to Try

- "What is the real remedy according to Dr. Ambedkar?"
- "What does Dr. Ambedkar say about social reform?"
- "What is the relationship between caste and the shastras?"
- "What analogy does he use to describe social reform work?"
- "What is the real enemy according to this speech?"

## 📁 Project Structure

```
AmbedkarGPT-Intern-Task/
│
├── main.py              # Main RAG application code
├── speech.txt           # Dr. Ambedkar's speech text
├── requirements.txt     # Python dependencies
├── README.md           # This file
│
└── chroma_db/          # ChromaDB vector store (created on first run)
    └── ...
```

## 🔧 How It Works

### 1. Document Loading
- Uses `TextLoader` to load the speech from `speech.txt`

### 2. Text Chunking
- Uses `CharacterTextSplitter` with:
  - Chunk size: 500 characters
  - Overlap: 50 characters (maintains context between chunks)

### 3. Embedding Creation
- Uses HuggingFace's `all-MiniLM-L6-v2` model
- Converts text chunks into 384-dimensional vectors
- Runs entirely locally on CPU

### 4. Vector Storage
- Stores embeddings in ChromaDB
- Persists to `./chroma_db/` directory
- Enables semantic search for relevant chunks

### 5. Question Answering
- User asks a question
- System creates embedding of the question
- Retrieves top 3 most similar chunks from ChromaDB
- Sends context + question to Ollama Mistral 7B
- LLM generates answer based only on provided context

## 🐛 Troubleshooting

### Ollama Connection Error
```
Error: Failed to connect to Ollama
```
**Solution:** Make sure Ollama is running:
```bash
ollama serve
```

### Model Not Found
```
Error: model 'mistral' not found
```
**Solution:** Pull the Mistral model:
```bash
ollama pull mistral
```

### Import Errors
```
ModuleNotFoundError: No module named 'langchain'
```
**Solution:** Ensure you're in the virtual environment and run:
```bash
pip install -r requirements.txt
```

### Slow Performance
- **First run**: Embedding model download and initial processing is slow
- **Subsequent runs**: Should be faster as ChromaDB is persisted
- **Low RAM**: Close other applications, consider using a smaller model

### ChromaDB Issues
If you encounter ChromaDB errors, delete the database and recreate:
```bash
rm -rf chroma_db/
python main.py
```

## 🧪 Testing the Components

### Test Ollama Alone
```bash
ollama run mistral
# Ask: "What is the capital of France?"
# Exit with: /bye
```

### Test Embeddings
```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vec = embeddings.embed_query("test text")
print(f"Embedding dimension: {len(vec)}")  # Should output: 384
```

## 📚 Key Dependencies

- `langchain`: RAG pipeline orchestration
- `chromadb`: Local vector database
- `sentence-transformers`: Embedding model
- `ollama`: LLM inference
- `torch`: PyTorch for embeddings
- `transformers`: HuggingFace model support

## 🎓 Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://ollama.ai/)
- [Sentence Transformers](https://www.sbert.net/)

##  Assignment Details

This project was created as part of the AI Intern assignment for **Kalpit Pvt Ltd, UK**.

**Assignment**: Phase 1 - Core Skills Evaluation  
**Task**: Build a functional RAG Q&A prototype  
**Timeline**: 7 days

## 📝 Notes

- No API keys required - everything runs locally
- No costs involved - all components are free and open-source
- First run takes longer due to model downloads
- Vector store persists between runs for efficiency
- System only answers based on the provided speech text

##  Acknowledgments

- Dr. B.R. Ambedkar for the profound insights
- LangChain, ChromaDB, HuggingFace, and Ollama communities

## 📧 Contact

For questions or issues, contact: dharmendrra06@gmail.com

---

**Note**: This is a demonstration prototype for educational purposes. For production use, consider additional error handling, logging, security measures, and scalability optimizations.