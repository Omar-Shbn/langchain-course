from dotenv import load_dotenv

load_dotenv()
from typing import List
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
import os
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
# to create an AI agent, it needs minimum 2 things
# 1- LLM
# 2- Tools


tavily = TavilySearch()


# @tool
# def search_engine(query: str) -> str:
#     """
#     tool that searches over internet
#     args:
#         query: the query to search for
#     retruns:
#         the search results
#     """
#     print(f"searching for {query}")
#     return str(tavily.search(query=query))
class Source(BaseModel):
    """
    scheme for a source used by the agent
    """
    url: str = Field(description="The URL of the source!")
class AgentResponse(BaseModel):
    """
    Scheme for the agent response with answer and sources!
    """
    answer: str = Field(description=" the agent's answer to the query")
    sources: List[Source] = Field(default_factory= list, description=" the list of sources that is used to describe the answer.")



tools = [tavily]
llm = ChatOllama(
    temperature=0, model="gemma4:latest", base_url=os.environ.get("OLLAMA_BASE_URL")
)
agent = create_agent(model=llm, tools=tools, response_format= AgentResponse)


def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke(
        {"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in Abu Dhabi on linkedin and list their details?")]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
