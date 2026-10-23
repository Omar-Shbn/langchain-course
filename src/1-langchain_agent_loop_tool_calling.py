from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
import os

load_dotenv()

MAX_ITER = 10
model = "gemma4:latest"


# ------ tools and decorators ------


@tool
def get_product_price(product: str) -> float:
    """look up the price of a product in the catalog"""
    print(f"getting product price for {product}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """apply a discount tier to a price and return the final price.
    available tiers are: bronze, silver, gold"""
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 15}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dic = {t.name: t for t in tools}

    llm = init_chat_model(
        f"ollama:{model}", temperature=0, base_url=os.getenv("OLLAMA_BASE_URL")
    )
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("=" * 60)

    message = [
        SystemMessage(
            content=(
                "your a helpful Shopping assistant."
                "you have access to a product catalog and a discount tool \n\n"
                "STRICT RULES - you must follow these exactly: \n"
                "1- NEVER guess or assume any product price."
                "You must call get_product_price first get the specific price for the wanted item"
                "2- only call apply_discout AFTER you have received a price from get_product_price. pass the exact price"
                "returned by get_product_price - do NOT pass a made-up number. \n"
                "3- NEVER calculate discounts yourself using math."
                "always use the apply_discount tool."
                "4- If a user doesnt specify a discount tier, ask them which tier to use - DO NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITER + 1):
        print(f"ITERATION NUMBER:{iteration}")
        ai_message = llm_with_tools.invoke(message)
        tool_calls = ai_message.tool_calls

        # if no tool calls, this is the final answer
        if not tool_calls:
            print(f"\n final answer:{ai_message.content}")
            return ai_message.content
        # process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get('id')
        print(f"[tool selected] {tool_name} - with args: {tool_args}")

        tool_to_use = tools_dic.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"tool{tool_name} is not found.")
        observation = tool_to_use.invoke(tool_args)
        print(f"tool results: {observation}")

        message.append(ai_message)
        message.append(
            ToolMessage(content=str(observation), tool_call_id = tool_call_id)
            )
    print(f'ERROR: Max iterations reached without a final answer')
    return None



if __name__ == "__main__":
    print("hello langChain Agent (.bind_tools)!")
    print()
    result = run_agent("what is the price of a laptop after applying a gold discount?")
