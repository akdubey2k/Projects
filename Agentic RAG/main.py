# Imports the class (or module) RAGSystem from local file src/rag.py
# Handles retrieval, augmentation, and answer generation
from src.rag import RAGSystem

# Imports the AgenticAI class (or module) from local file src/agent.py
# Adds reasoning, planning, and tool-use capabilities
from src.agent import AgenticAI

# Imports the LargeContextModel class from local src/lcm.py module.
# Enables long-context understanding for RAG
from src.lcm import LargeContextModel

# Imports the HuggingFacePipeline class from the langchain_huggingface integration
# Wraps Hugging Face transformers for use in LangChain
from langchain_huggingface import HuggingFacePipeline

# Imports Python’s built-in warnings module.
# Suppresses or manages runtime warnings
import warnings

# Imports Python’s standard os module for interacting with the operating system.
# Handles environment variables, file paths, and directories
import os

# Suppress LangSmith warning
os.environ["LANGSMITH"] = "false"

def main():
    # Suppresses all FutureWarning messages emitted by Python or other libraries.
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Initialize Large Context Model
    # Creates an instance of custom LargeContextModel (LCM) class using "distilgpt2" as the underlying 
    # model.
    lcm = LargeContextModel(model_name="distilgpt2")

    # Calls the method that loads the actual model weights and tokenizer into memory (usually from 
    # Hugging Face).
    lcm.load_model()
    
    # Create LangChain-compatible LLM
    # Wraps the Hugging Face pipeline from your LCM into a LangChain-compatible LLM interface.
    # The HuggingFacePipeline bridges Hugging Face and LangChain by providing this adapter.
    llm = HuggingFacePipeline(
        pipeline = lcm.pipeline,
        pipeline_kwargs={"clean_up_tokenization_spaces": True, "max_new_tokens": 200}
    )

    # Initialize RAG system
    # Instantiates Retrieval-Augmented Generation (RAG) system with a path to a text dataset.
    # The system will later read this file, split it into smaller chunks, embed them, and create a 
    # vector store.
    rag = RAGSystem(data_path="data/sample_data.txt")
    texts = rag.load_data()
    rag.create_vector_store(texts)
    rag.initialize_qa(llm)

    # Initialize Agentic AI
    # Creates an instance of your AgenticAI system, providing it access to the RAG system.
    agent = AgenticAI(rag)

    # Initializes the agent’s decision-making pipeline using the LLM.
    agent.create_agent(llm)

    # Example: Run a query through RAG
    # Demonstrates querying the RAG system directly. The question goes through:
    # → Embedding → Retrieval → LLM answer generation.
    question = "What is a large context model?"
    rag_response = rag.query(question)
    print("RAG Response:", rag_response)

    # Example: Execute a task with the agent
    # Sends a higher-level instruction/task to the AgenticAI. Unlike the simple RAG query, this 
    # “task” may trigger multiple reasoning steps like the agent could:-
    # 1. Retrieve information using RAG,
    # 2. Summarize it using the LLM,
    task = "Find information about large context models and summarize it."
    agent_response = agent.execute_task(task)
    print("Agent Response:", agent_response)

    # Example: Generate text with LCM
    # Demonstrates the standalone generative capability of the LCM, LargeContextModel directly 
    # for pure text generation (without RAG or Agent). This helps to compare raw model output vs 
    # RAG/Agent-enhanced answers. The LCM takes the input prompt, processes it through its 
    # large-context transformer pipeline, and generates a coherent response.
    prompt = "Explain the benefits of large context models in AI."
    lcm_response = lcm.generate(prompt)
    print("LCM Response:", lcm_response)

# ensure main() only runs when this file is executed directly.
if __name__ == "__main__":
    main()