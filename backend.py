# backend.py
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests
import json

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("OPENAI_API_KEY"),           # ← replace with your key
    model="llama-3.3-70b-versatile"
)

# -------------------
# 2. Tools
# -------------------

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price, volume, open, high, low, previous close,
    and percentage change for a given symbol (e.g. 'AAPL', 'TSLA').
    Always use this when the user asks about a stock price or quote.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol.upper()}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url, timeout=10)
    data = r.json()
    quote = data.get("Global Quote", {})
    if not quote:
        return {"error": f"No data found for symbol {symbol}. It may be invalid."}
    return {
        "symbol": quote.get("01. symbol"),
        "price": quote.get("05. price"),
        "open": quote.get("02. open"),
        "high": quote.get("03. high"),
        "low": quote.get("04. low"),
        "volume": quote.get("06. volume"),
        "previous_close": quote.get("08. previous close"),
        "change": quote.get("09. change"),
        "change_percent": quote.get("10. change percent"),
    }


@tool
def compare_stocks(symbols: list[str]) -> dict:
    """
    Compare prices and stats for multiple stock symbols at once.
    Use this when the user wants to compare 2 or more stocks.
    E.g. compare_stocks(["AAPL", "MSFT", "GOOGL"])
    """
    results = {}
    for symbol in symbols:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={symbol.upper()}&apikey=C9PE94QUEW9VWGFM"
        )
        r = requests.get(url, timeout=10)
        data = r.json()
        quote = data.get("Global Quote", {})
        if quote:
            results[symbol.upper()] = {
                "price": quote.get("05. price"),
                "change_percent": quote.get("10. change percent"),
                "volume": quote.get("06. volume"),
                "high": quote.get("03. high"),
                "low": quote.get("04. low"),
            }
        else:
            results[symbol.upper()] = {"error": "Symbol not found"}
    return results


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.
    Always requires human approval before executing.
    Use this when the user explicitly says they want to buy shares.
    """
    decision = interrupt(f"Approve buying {quantity} shares of {symbol.upper()}? (yes/no)")
    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"✅ Purchase order placed: {quantity} shares of {symbol.upper()}.",
            "symbol": symbol.upper(),
            "quantity": quantity,
        }
    return {
        "status": "cancelled",
        "message": f"❌ Purchase of {quantity} shares of {symbol.upper()} was cancelled.",
        "symbol": symbol.upper(),
        "quantity": quantity,
    }


@tool
def sell_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate selling a given quantity of a stock symbol.
    Always requires human approval before executing.
    Use this when the user explicitly says they want to sell shares.
    """
    decision = interrupt(f"Approve selling {quantity} shares of {symbol.upper()}? (yes/no)")
    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"✅ Sell order placed: {quantity} shares of {symbol.upper()}.",
            "symbol": symbol.upper(),
            "quantity": quantity,
        }
    return {
        "status": "cancelled",
        "message": f"❌ Sale of {quantity} shares of {symbol.upper()} was cancelled.",
        "symbol": symbol.upper(),
        "quantity": quantity,
    }


@tool
def set_price_alert(symbol: str, target_price: float, direction: str) -> dict:
    """
    Set a price alert for a stock symbol.
    direction must be 'above' or 'below'.
    E.g. set_price_alert('AAPL', 200.0, 'above') means alert when AAPL goes above $200.
    Use this when the user wants to be notified when a stock hits a price.
    """
    valid_directions = ["above", "below"]
    if direction.lower() not in valid_directions:
        return {"error": "direction must be 'above' or 'below'"}
    return {
        "status": "alert_set",
        "message": f"🔔 Alert set: notify when {symbol.upper()} goes {direction} ${target_price:.2f}.",
        "symbol": symbol.upper(),
        "target_price": target_price,
        "direction": direction.lower(),
    }


tools = [get_stock_price, compare_stocks, purchase_stock, sell_stock, set_price_alert]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. System prompt
# -------------------
SYSTEM_PROMPT = SystemMessage(content="""You are StockBot, an expert AI stock trading assistant.

Your capabilities:
- Look up real-time stock prices and detailed quotes (get_stock_price)
- Compare multiple stocks side by side (compare_stocks)
- Place buy orders with human approval (purchase_stock)
- Place sell orders with human approval (sell_stock)
- Set price alerts for stocks (set_price_alert)

Behavior rules:
1. ALWAYS use tools when the user mentions a stock symbol or asks about prices — never guess prices from memory.
2. When showing prices, include change % and key stats (open, high, low, volume).
3. For comparisons, always use compare_stocks — do not call get_stock_price multiple times separately.
4. For buy/sell orders, confirm the symbol and quantity before calling the tool.
5. If a symbol seems wrong or the user is vague, ask for clarification.
6. After a successful purchase or sale, summarize what happened clearly.
7. For price alerts, confirm the alert was set and explain what will trigger it.
8. Be concise, professional, and friendly. Use numbers and facts — avoid filler text.
9. If asked for investment advice, remind the user you provide data only, not financial advice.
10. Format prices clearly: always include the $ sign and 2 decimal places.
""")

# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState):
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
memory = MemorySaver()

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=memory)

# -------------------
# 8. CLI usage
# -------------------
if __name__ == "__main__":
    thread_id = "demo-thread"
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break
        state = {"messages": [HumanMessage(content=user_input)]}
        result = chatbot.invoke(state, config={"configurable": {"thread_id": thread_id}})
        interrupts = result.get("__interrupt__", [])
        if interrupts:
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()
            result = chatbot.invoke(Command(resume=decision), config={"configurable": {"thread_id": thread_id}})
        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")