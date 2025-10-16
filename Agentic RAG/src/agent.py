from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain import hub
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