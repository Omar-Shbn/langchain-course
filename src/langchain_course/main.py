from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
import os
from langchain_tavily import TavilySearch

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


tools = [tavily]
llm = ChatOllama(
    temperature=0, model="gemma4:latest", base_url=os.environ.get("OLLAMA_BASE_URL")
)
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke(
        {"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in Abu Dhabi on linkedin and list their details?")]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
