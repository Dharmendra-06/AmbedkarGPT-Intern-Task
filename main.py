"""
AmbedkarGPT - RAG-based Q&A System
A command-line application that answers questions based on Dr. B.R. Ambedkar's speech
using Retrieval-Augmented Generation (RAG) with LangChain, ChromaDB, and Ollama.
"""

import os
import sys
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def setup_vector_store(file_path="speech.txt", persist_directory="./chroma_db"):
    """
    Load the speech text, split into chunks, create embeddings, and store in ChromaDB.
    
    Args:
      file_path: Path to the speech text file
      persist_directory: Directory to persist the ChromaDB database
        
    Returns:
      vectorstore: ChromaDB vector store instance
    """
    print("📚 Loading speech text...")
    
    # Check if file exists
    if not os.path.exists(file_path):
      print(f"❌ Error: {file_path} not found!")
      sys.exit(1)
    
    # Load the document
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} document(s)")
    
    # Split text into chunks
    print("✂️  Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(
      chunk_size=500,  # Size of each chunk
      chunk_overlap=50,  # Overlap between chunks to maintain context
      separator="\n"
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks")
    
    # Create embeddings using HuggingFace model
    print("🧠 Creating embeddings (this may take a moment on first run)...")
    embeddings = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2",
      model_kwargs={'device': 'cpu'}  # Use CPU for compatibility
    )
    
    # Create and persist vector store
    print("💾 Creating vector store with ChromaDB...")
    vectorstore = Chroma.from_documents(
      documents=chunks,
      embedding=embeddings,
      persist_directory=persist_directory
    )
    vectorstore.persist()
    print("✅ Vector store created and persisted successfully!")
    
    return vectorstore


def load_existing_vector_store(persist_directory="./chroma_db"):
    """
    Load an existing ChromaDB vector store.
    
    Args:
      persist_directory: Directory where ChromaDB is persisted
        
    Returns:
      vectorstore: ChromaDB vector store instance
    """
    print("📂 Loading existing vector store...")
    embeddings = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2",
      model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = Chroma(
      persist_directory=persist_directory,
      embedding_function=embeddings
    )
    print("✅ Vector store loaded successfully!")
    return vectorstore


def setup_qa_chain(vectorstore):
    """
    Set up the RetrievalQA chain with Ollama LLM.
    
    Args:
      vectorstore: ChromaDB vector store instance
        
    Returns:
      qa_chain: RetrievalQA chain instance
    """
    print("🤖 Setting up Ollama LLM...")
    
    # Initialize Ollama with Mistral model
    llm = Ollama(
      model="mistral",
      temperature=0.3  # Lower temperature for more focused answers
    )
    
    # Create a custom prompt template
    prompt_template = """Use the following pieces of context from Dr. B.R. Ambedkar's speech to answer the question. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """
    
    PROMPT = PromptTemplate(
      template=prompt_template,
      input_variables=["context", "question"]
    )
    
    # Create RetrievalQA chain
    print("🔗 Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",  # 'stuff' puts all retrieved docs into context
      retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
      ),
      return_source_documents=True,
      chain_type_kwargs={"prompt": PROMPT}
    )
    
    print("✅ QA chain ready!")
    return qa_chain


def ask_question(qa_chain, question):
    """
    Ask a question and get an answer from the RAG system.
    
    Args:
      qa_chain: RetrievalQA chain instance
      question: User's question string
        
    Returns:
      result: Dictionary containing answer and source documents
    """
    print(f"\n❓ Question: {question}")
    print("🔍 Retrieving relevant context and generating answer...\n")
    
    result = qa_chain({"query": question})
    
    print(f"💡 Answer: {result['result']}\n")
    
    # Show source chunks used for the answer
    if result.get('source_documents'):
      print("📄 Source chunks used:")
      for i, doc in enumerate(result['source_documents'], 1):
        print(f"\n[Chunk {i}]")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    
    return result


def main():
    """
    Main function to run the Q&A system.
    """
    print("=" * 60)
    print("🙏 Welcome to AmbedkarGPT - RAG Q&A System")
    print("=" * 60)
    print()
    
    persist_dir = "./chroma_db"
    
    # Check if vector store already exists
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
      print("ℹ️  Found existing vector store. Loading...")
      vectorstore = load_existing_vector_store(persist_dir)
    else:
      print("ℹ️  No existing vector store found. Creating new one...")
      vectorstore = setup_vector_store(persist_directory=persist_dir)
    
    print()
    
    # Setup QA chain
    qa_chain = setup_qa_chain(vectorstore)
    
    print("\n" + "=" * 60)
    print("🎯 System is ready! You can now ask questions.")
    print("=" * 60)
    print("💡 Tip: Ask questions about the speech content")
    print("📝 Type 'exit' or 'quit' to stop\n")
    
    # Interactive Q&A loop
    while True:
        try:
            question = input("Your question: ").strip()
            
            if not question:
              print("⚠️  Please enter a question.\n")
              continue
            
            if question.lower() in ['exit', 'quit', 'q']:
              print("\n👋 Thank you for using AmbedkarGPT. Goodbye!")
              break
            
            ask_question(qa_chain, question)
            print("\n" + "-" * 60 + "\n")
            
        except KeyboardInterrupt:
          print("\n\n👋 Interrupted. Goodbye!")
          break
        except Exception as e:
          print(f"\n❌ Error: {str(e)}")
          print("Please try again.\n")


if __name__ == "__main__":
  main()