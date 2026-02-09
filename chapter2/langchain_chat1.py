import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model


class Configuration:

    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.base_url = "https://api.siliconflow.cn/v1/"
        self.model = "Qwen/Qwen3-8B"

    @staticmethod
    def load_servers(file_path="servers_config.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})


async def run_chat_loop():
    config = Configuration()
    server_config = config.load_servers()

    mcp_client = MultiServerMCPClient(server_config)
    tools = await mcp_client.get_tools()

    logging.info(f"✅ 已加载 {len(tools)} 个 MCP 工具： {[t.name for t in tools]}")

    llm = init_chat_model(
        model=config.model,
        model_provider="openai",
        base_url=config.base_url,
        api_key=config.api_key,
    )

    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    print("\n🤖 MCP Agent 已启动，输入 'quit' 退出")

    while True:
        user_input = input().strip()
        if user_input.lower() == 'quit':
            print("👋 退出 MCP Agent")
            break
        try:
            result = await agent_executor.ainvoke({"input": user_input})
            print(f"\nAI: {result['output']}")
        except Exception as exc:
            print(f"\n⚠️  出错: {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(run_chat_loop())
