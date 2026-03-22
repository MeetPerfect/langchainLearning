#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/3/22 12:57
# @Author  : myymgkm
# @File    : langgraph2.py
# @Description: Langgraph的路由模式
import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.constants import START, END
from langgraph.graph import StateGraph

load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("API_KEY"),
)


# 定义状态

class State(TypedDict):
    topic: str
    joke: str
    story: str
    poetry: str
    combined_output: str


# 定义节点
def generate_joke(state: State):
    """
    写笑话
    :param state:
    :return:
    """
    print('进入写笑话处理逻辑')
    result = llm.invoke(state['topic'])
    return {
        'joke': result.content
    }


def generate_story(state: State):
    """
    写故事
    :param state:
    :return:
    """
    print('进入写故事处理逻辑')
    result = llm.invoke(state['topic'])
    return {
        'story': result.content
    }


def generate_poetry(state: State):
    '''
    写诗歌
    '''
    print('进入写诗歌处理逻辑')
    result = llm.invoke(state['topic'])
    return {
        'poetry': result.content
    }


def aggregator(state: State):
    topic = state['topic']
    joke = state['joke']
    story = state['story']
    poetry = state['poetry']
    combined = f'这是一个关于 {topic} 的故事、笑话和诗歌的合集\n\n'
    combined += f'故事\n {story}\n\n'
    combined += f'笑话\n {joke}\n\n'
    combined += f'诗歌\n {poetry}\n\n'
    return {
        'combined_output': combined
    }


workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("generate_story", generate_story)
workflow.add_node("generate_poetry", generate_poetry)
workflow.add_node("aggregator", aggregator)

workflow.add_edge(START, "generate_joke")
workflow.add_edge(START, "generate_story")
workflow.add_edge(START, "generate_poetry")

workflow.add_edge("generate_joke", "aggregator")
workflow.add_edge("generate_story", "aggregator")
workflow.add_edge("generate_poetry", "aggregator")
workflow.add_edge("aggregator", END)

graph = workflow.compile()

result = graph.invoke({"topic": "关于新有菜的"})

print(result['combined_output'])
