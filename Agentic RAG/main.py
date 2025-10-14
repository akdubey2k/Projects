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
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Initialize Large Context Model
    lcm = LargeContextModel(model_name="distilgpt2")
    lcm.load_model()
    
    # Create LangChain-compatible LLM
    llm = HuggingFacePipeline(
        pipeline=lcm.pipeline,
        pipeline_kwargs={"clean_up_tokenization_spaces": True, "max_new_tokens": 200}
    )

    # Initialize RAG system
    rag = RAGSystem(data_path="data/sample_data.txt")
    texts = rag.load_data()
    rag.create_vector_store(texts)
    rag.initialize_qa(llm)

    # Initialize Agentic AI
    agent = AgenticAI(rag)
    agent.create_agent(llm)

    # Example: Run a query through RAG
    question = "What is a large context model?"
    rag_response = rag.query(question)
    print("RAG Response:", rag_response)

    # Example: Execute a task with the agent
    task = "Find information about large context models and summarize it."
    agent_response = agent.execute_task(task)
    print("Agent Response:", agent_response)

    # Example: Generate text with LCM
    prompt = "Explain the benefits of large context models in AI."
    lcm_response = lcm.generate(prompt)
    print("LCM Response:", lcm_response)

if __name__ == "__main__":
    main()