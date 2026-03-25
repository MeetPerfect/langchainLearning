#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/3/25 16:52
# @Author  : myymgkm
# @File    : langchain5.py
# @Description: 评估优化器
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("API_KEY"),
)


class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description='判断笑话是否有趣'
    )
    feedback: str = Field(
        description='如果笑话不好笑，提供改进它的反馈'
    )


evaluator = llm.with_structured_output(Feedback)


# 状态定义
class State(TypedDict):
    topic: str
    joke: str
    feedback: str
    funny_or_not: str


# 定义节点
def generate_joke(state: State):
    topic = state["topic"]
    if state.get("feedback"):
        feedback = state["feedback"]
        msg = llm.invoke(f'请写一个关于{topic}的笑话，但是要考虑反馈:{feedback}')
    else:
        msg = llm.invoke(f'写一个关于{topic}的笑话')

    return {
        "joke": msg
    }


def evaluate_joke(state: State):
    joke = state["joke"]
    grade = evaluator.invoke(f'评估笑话{joke}是否好笑,如果不好笑给出修改建议')

    return {
        "funny_or_not": grade.grade,
        "feedback": grade.feedback
    }


# 定义边和图
def route_joke(state: State):
    if state["funny_or_not"] == "funny":
        return "Accept"
    elif state["funny_or_not"] == "not funny":
        return "Reject"


graph_builder = StateGraph(State)

graph_builder.add_node("generate_joke", generate_joke)
graph_builder.add_node("evaluate_joke", evaluate_joke)

graph_builder.add_edge(START, "generate_joke")
graph_builder.add_edge("generate_joke", "evaluate_joke")
graph_builder.add_conditional_edges(
    "evaluate_joke", route_joke,
    {
        "Accept": END,
        "Reject": "generate_joke"
    }
)

graph = graph_builder.compile()

result = graph.invoke({
    'topic': '贾乃亮与pg one'
})

print(result["joke"])
