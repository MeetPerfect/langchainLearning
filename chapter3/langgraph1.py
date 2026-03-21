#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/3/21 19:37
# @Author  : myymgkm
# @File    : langgraph1.py
# @Description:
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph

load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("API_KEY")

)


# 定义状态
class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str


# 定义节点

def generate_joke(state: State):
    """
    大模型调用，根据标题生成joke
    :param state:
    :return:
    """
    topic = state["topic"]

    msg = llm.invoke(f"写一个关于{topic}的简短笑话")

    return {
        "joke": msg.content
    }


def check_punchline(state: State):
    joke = state["joke"]
    if "?" in joke or "？" in joke:
        return "fail"
    return "success"


def improved_joke(state: State):
    """
    改进joke
    :param state:
    :return:
    """
    joke = state["joke"]

    msg = llm.invoke(f'为这个笑话添加一个令人惊讶的转折: {joke}')
    return {
        'improved_joke': msg.content
    }


def publish_joke(state: State):
    improved_joke = state["improved_joke"]
    msg = llm.invoke(f"为这个笑话添加一个令人惊讶的转折: {improved_joke}")

    return {
        "final_joke": msg.content
    }


workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)

workflow.add_node("improved_joke", improved_joke)
workflow.add_node("publish_joke", publish_joke)

workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges("generate_joke", check_punchline, {
    "fail": "improved_joke",
    "success": END
})

workflow.add_edge("improved_joke", "publish_joke")
workflow.add_edge("publish_joke", END)


chain = workflow.compile()

state  = chain.invoke({"topic": "小猫"})

print('初始笑话:')
print(state['joke'])

if "improved_joke" in state:
    print('改进后笑话:')
    print(state['improved_joke'])

    print('最终笑话:')
    print(state['final_joke'])


