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
    input: str
    decision: str
    output: str


# 定义节点
def generate_story(state: State):
    """
    写故事
    :param state:
    :return:
    """
    print('进入写故事处理逻辑')
    result = llm.invoke(state['input'])
    return {
        'output': result.content
    }


def generate_joke(state: State):
    """
    写笑话
    :param state:
    :return:
    """
    print('进入写笑话处理逻辑')
    result = llm.invoke(state['input'])
    return {
        'output': result.content
    }


def generate_poetry(state: State):
    '''
    写诗歌
    '''
    print('进入写诗歌处理逻辑')
    result = llm.invoke(state['input'])
    return {
        'output': result.content
    }


class Classification(TypedDict):
    response_format: Literal['story', 'joke', 'poetry']


def llm_call_router(state: State):
    structured_llm = llm.with_structured_output(Classification)
    input = state['input']

    response = structured_llm.invoke([
        SystemMessage(content="""
            你是一个分类路由，根据用户的输入进行分类，分类结果是story, joke, poetry三者中的一种
        """),
        HumanMessage(content=input)
    ])

    return {
        'decision': response['response_format']
    }


def route_decision(state: State):
    if state["decision"] == 'story':
        return "llm_story"
    if state["decision"] == 'joke':
        return "llm_joke"
    if state["decision"] == 'poetry':
        return "llm_poetry"

workflow = StateGraph(State)
workflow.add_node("generate_story", generate_story)
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("generate_poetry", generate_poetry)
workflow.add_node("llm_call_router", llm_call_router)

workflow.add_edge(START, "llm_call_router")
workflow.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {
        "llm_story": "generate_story",
        "llm_joke": "generate_joke",
        "llm_poetry": "generate_poetry"
    }
)

workflow.add_edge("generate_story", END)
workflow.add_edge("generate_joke", END)
workflow.add_edge("generate_poetry", END)

graph = workflow.compile()

result = graph.invoke({
    "input": "给我写一个关于新有菜的笑话"
})

print(result['output'])

