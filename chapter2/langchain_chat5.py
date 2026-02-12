#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/2/12 14:45
# @Author  : myymgkm
# @File    : langchain_chat5.py
# @Description: 接入langchain内置工具
import os

import dotenv
from langchain.chat_models import init_chat_model
from langchain_classic import hub
from langchain_classic.agents import create_react_agent, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch

dotenv.load_dotenv()

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    tavily_api_key=os.getenv("TAILLY_API_KEY"),
)

# prompt = ChatPromptTemplate([
#     ("system", "你是一个智能助手，会根据需求调用合适的工具"),
#     ("user", "{input}"),
#     ("placeholder", "{agent_scratchpad}")
# ])

prompt = hub.pull("hwchase17/react")

tools = [search_tool]

model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key=os.getenv("API_KEY"),
)

agent = create_react_agent(llm=model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
response = agent_executor.invoke({"input": "请帮我搜索最近OpenAI CEO在访谈中的核心观点。"})

print(response.get("output"))
