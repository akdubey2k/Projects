# RAG + Agentic AI + LCM Project Setup
**Retrieval-Augmented Generation (RAG)** system combined with **Agentic AI** and a **Large Context Model (LCM)**.
Project leveraging libraries like **LangChain** for _RAG, agentic workflows, and Hugging Face transformers for a large context model._ The project will include a simple RAG pipeline, an agentic AI component for task execution, and integration with a model capable of handling large contexts (e.g., a transformer model with a large context window)

RAG + Agentic AI + LCM Project
This project implements a Retrieval-Augmented Generation (RAG) system with an agentic AI workflow and a Large Context Model (LCM) using Python, LangChain, and Hugging Face transformers.

Setup Instructions
Create and activate a virtual environment:

> python3 -m venv venv
> source venv/bin/activate

Install dependencies:

> pip install -r requirements.txt

Run the main script:

> python main.py

Components

1. RAG: Uses FAISS for vector storage and sentence transformers for embeddings.
2. Agentic AI: Leverages LangChain agents to execute tasks with RAG as a tool.
3. LCM: Uses a transformer model (e.g., distilgpt2) for large context generation.

Notes

- Replace distilgpt2 with a larger model (e.g., gpt2-large) for better LCM performance if hardware allows.
- Ensure sufficient memory for large context models.
- Add more data to data/sample_data.txt for richer RAG results.

