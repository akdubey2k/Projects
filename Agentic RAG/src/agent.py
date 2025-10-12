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

# Import hub to access pre-defined prompts from LangChain's prompt hub module allows users to pull 
# pre-configured prompts from LangChain’s prompt hub, a centralized repository of optimized prompts
# for various tasks, such as ReAct agent reasoning or question-answering.
from langchain import hub

# Import HuggingFacePipeline to integrate a Hugging Face language model to creates a pipeline for
# use a Hugging Face language model (e.g., GPT-2, LLaMA, or Mistral) for tasks like text generation
# or reasoning. It wraps Hugging Face’s transformers library, making it compatible with LangChain’s 
# agent and chain frameworks.
from langchain_huggingface import HuggingFacePipeline

# AgenticAI class to encapsulate the functionality of an AI agent integrated with a RAG system.
class AgenticAI:
    # The constructor initializes an instance of AgenticAI by storing a reference to a RAG system,
    def __init__(self, rag_system):
        # object, allows the agent to query a knowledge base for relevant information. Stores the 
        # RAG system as an instance variable, making it accessible to other methods in the class
        self.rag_system = rag_system
        # agent not created, defers creation to the create_agent method with a language model.
        self.agent = None

    # method sets up the ReAct agent by defining tools, pulling a prompt, and creating an 
    # AgentExecutor. It integrates the RAG system as a tool and uses the provided language model 
    # for reasoning.
    # llm parameter is typically a LangChain-compatible language model (e.g., HuggingFacePipeline) 
    # that the agent uses for reasoning and generating responses.
    def create_agent(self, llm):
        # creates a list of Tool objects, with a single tool that wraps RAG system’s query method.
        tools = [
            # Tool that allows the ReAct agent to interact with the RAG system. The tool is named 
            # “RAG Search,” uses the rag_system.query method as its function, and includes a 
            # description to guide the agent’s decision-making.
            Tool(
                # The name (“RAG Search”) and description (“Search the RAG system for relevant 
                # information”) are used by the ReAct agent’s language model to decide when to call 
                # the tool. A clear description ensures the agent understands the tool’s purpose 
                # (e.g., to retrieve document-based information).
                name="RAG Search",
                # The rag_system is assumed to have a query method (common in RAG systems, e.g., a
                # RetrievalQA chain or FAISS retriever) that takes a query string and returns 
                # relevant documents or answers. This method is wrapped as a tool so the agent can
                # use it during reasoning.
                func=self.rag_system.query,
                description="Search the RAG system for relevant information."
            )
        ]
        # Use default ReAct prompt from langchain-hub fetches a pre-defined ReAct prompt template, 
        # which instructs the language model to reason step-by-step, decide whether to use tools, 
        # and format responses (e.g., using [THOUGHT], [ACTION], [OBSERVATION] tags).
        # LangChain’s prompt hub provides optimized, community-tested prompts that simplify agent 
        # setup. The “hwchase17/react” prompt is specifically designed for ReAct agents, ensuring 
        # the language model follows a structured reasoning-and-acting process.
        prompt = hub.pull("hwchase17/react")
        

        # Initializes the ReAct agent and wraps it in an AgentExecutor to manage its execution. The
        # AgentExecutor handles the interaction loop, including reasoning, tool calls, and response 
        # generation.
        self.agent = AgentExecutor(
            # create_react_agent function (imported previously) constructs a ReAct agent using the 
            # provided language model (llm), tools (the “RAG Search” tool), and prompt (from the 
            # hub). The agent combines reasoning (via the llm) with action-taking (via tools) based 
            # on the prompt’s instructions.
            agent=create_react_agent(llm, tools, prompt),
            # Passes the list of tools (containing “RAG Search”) to the AgentExecutor, ensuring the 
            # agent can invoke them during execution.
            tools=tools,
            # Enables verbose logging, printing intermediate steps (e.g., thoughts, actions, 
            # observations) during the agent’s execution.
            verbose=True,
            # Configures the AgentExecutor to handle parsing errors gracefully, such as when the 
            # language model produces malformed output (e.g., incorrect action syntax).
            handle_parsing_errors=True
        )

    # Provides a public interface to run tasks through the ReAct agent, returning the agent’s output.
    # The task is a string representing the user’s query or instruction (e.g., “What’s the capital of
    # France?”). It’s passed to the agent for processing.
    def execute_task(self, task):
        # Ensures that the agent has been created (via create_agent) before attempting to execute a 
        # task, preventing runtime errors due to an uninitialized agent.
        if self.agent is None:
            raise ValueError("Agent not initialized.")
        # Calls the AgentExecutor’s invoke method to process the task, passing it as an "input" key 
        # in a dictionary, and extracts the "output" from the result.
        return self.agent.invoke({"input": task})["output"]