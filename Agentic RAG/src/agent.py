# setup for creating an agent-based system using LangChain, where an agent can reason, use tools, 
# and leverage a language model for tasks like question-answering or decision-making. The imports 
# focus on building a ReAct (Reasoning and Acting) agent with a Hugging Face language model.

# Import AgentExecutor, Tool, and create_react_agent for building a ReAct agent
# AgentExecutor is a LangChain class that manages the execution of an agent. It handles the 
# interaction loop between the agent, its tools, and the language model, ensuring the agent 
# processes inputs, decides on actions, uses tools, and generates outputs.
# 
# Tool class defines external functions or services the agent can use to gather information or 
# perform actions (e.g., search APIs, calculators, or database queries).
# 
# create_react_agent is a factory function that constructs a ReAct (Reasoning and Acting) agent, a 
# type of agent that combines reasoning (using a language model) with action-taking (using tools) 
# based on the ReAct framework.
from langchain.agents import AgentExecutor, Tool, create_react_agent

# Import hub to access pre-defined prompts from LangChain's prompt hub
# hub module allows users to pull pre-configured prompts from LangChain’s prompt hub, a centralized
# repository of optimized prompts for various tasks, such as ReAct agent reasoning or 
# question-answering.
from langchain import hub

# Import HuggingFacePipeline to integrate a Hugging Face language model
# HuggingFacePipeline creates a pipeline to use a Hugging Face language model (e.g., GPT-2, LLaMA, 
# or Mistral) for tasks like text generation or reasoning. It wraps Hugging Face’s transformers 
# library, making it compatible with LangChain’s agent and chain frameworks.
from langchain_huggingface import HuggingFacePipeline

class AgenticAI:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.agent = None

    def create_agent(self, llm):
        tools = [
            Tool(
                name="RAG Search",
                func=self.rag_system.query,
                description="Search the RAG system for relevant information."
            )
        ]
        # Use default ReAct prompt from langchain-hub
        prompt = hub.pull("hwchase17/react")
        self.agent = AgentExecutor(
            agent=create_react_agent(llm, tools, prompt),
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def execute_task(self, task):
        if self.agent is None:
            raise ValueError("Agent not initialized.")
        return self.agent.invoke({"input": task})["output"]