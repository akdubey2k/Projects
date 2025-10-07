# Library to load text data from files (e.g., .txt, .pdf, .docx)
from langchain.document_loaders import TextLoader

# Splits large chunks of text into smaller, vector storage, embeddings, and 
# retrieval-based QA, manageable pieces while preserving semantic meaning.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store to hold document embeddings for efficient retrieval.
# FAISS (Facebook AI Similarity Search) is a library for efficient similarity 
# search and clustering of dense vectors.
from langchain.vectorstores import FAISS

# Embedding model to convert text into vector representations.
# Generates dense vector representations (embeddings) of text using pre-trained
# Hugging Face models.
from langchain_huggingface import HuggingFaceEmbeddings

# Chain for retrieval-based question answering.
# Combines a retriever (e.g., FAISS) with a language model to answer questions 
# based on retrieved documents.
from langchain.chains import RetrievalQA

"""
Summary of Workflow

1. Load Data: Use TextLoader to load documents into memory.
2. Preprocess Data: Use RecursiveCharacterTextSplitter to split the text into smaller chunks.
3. Generate Embeddings: Use HuggingFaceEmbeddings to convert text chunks into vector representations.
4. Store Vectors: Use FAISS to index and store the embeddings for efficient retrieval.
5. Answer Questions: Use RetrievalQA to retrieve relevant chunks and generate answers using a language model.
"""
# Define a class to encapsulate the RAG system and all its functionalities components
# into a single reusable unit fro better state management.
# data_path: Path to the text documents data file.
# model_name: Name of the Hugging Face model to use for embeddings.
# DistilBERT: A smaller, faster alternative to BERT. DistilBERT is Good balance between performance 
# and resource usage. Alternatively can use "bert-base-uncased" (more accurate but slower), 
# "all-MiniLM-L6-v2" (even lighter)
class RAGSystem:
    def __init__(self, data_path, model_name="distilbert-base-uncased"):
        self.data_path = data_path
        self.model_name = model_name
        self.vector_store = None
        self.qa_chain = None

    # Load and preprocess data, create vector store, and initialize QA chain.
    # load text data from the specified file path.
    # read the documents contents and split them into smaller chunks at natural boundaries 
    # (paragraphs, sentences)
    # each chunk ~1000 characters with 100 characters overlap.
    # This prevents loss of context between chunks boundary.
    def load_data(self):
        loader = TextLoader(self.data_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        return texts

    # Create a vector store using FAISS and HuggingFaceEmbeddings.
    # Creates embedding model "all-MiniLM-L6-v2" Lightweight sentence transformer
    # Builds fast similarity search vector database from documents, efficient retrieval.
    # texts: List of text chunks to be embedded and stored.
    # embeddings: Converts text chunks into dense vector representations.
    def create_vector_store(self, texts):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(texts, embeddings)

    def initialize_qa(self, llm):
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4})  # Return top 4 documents
        )

    def query(self, question):
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized.")
        return self.qa_chain.invoke({"query": question})["result"]
