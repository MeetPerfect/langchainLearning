import getpass
import os

from docutils.nodes import topic
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch

# description: langchain工具调用示例，使用TavilySearch

load_dotenv()

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

search = TavilySearch(
    max_results=5,
    topic="general"
)

response = search.invoke({"query": "元宝和千问大战"})
print(response)

model_generated_tool_call = {
    "args": {"query": "euro 2024 host nation"},
    "id": "1",
    "name": "tavily",
    "type": "tool_call",
}

tool_msg = search.invoke(model_generated_tool_call)
print(tool_msg.content[:400])

tools = [search]

model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key=os.getenv("API_KEY"),
)

agent = create_agent(model, tools)
user_input = "What nation hosted the Euro 2024? Include only wikipedia sources."

for step in agent.stream(
        {"messages": user_input},
        stream_mode="values",
):
    step["messages"][-1].pretty_print()

