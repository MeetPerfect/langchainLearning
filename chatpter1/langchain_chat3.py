from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# description: langchain添加多轮对话记忆功能

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你叫苍井空，是日本著名女演员。"),
    MessagesPlaceholder(variable_name="messages"),
])

model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key="sk-lzcunbanmnmklpxtfehbnmupbgytyqgujjulndjtvhzjhqdq",
)

chain = prompt | model | StrOutputParser()

messages_list = []
print("🔹 输入 exit 结束对话")

while True:
    user_query = input("用户: ")

    if user_query.lower() in {"exit", "quit"}:
        print("对话结束。")
        break
    messages_list.append(HumanMessage(content=user_query))

    # 流式输出
    assistant_reply = ""
    for chunk in chain.stream({"messages": messages_list}):
        assistant_reply+=chunk
        print(chunk, end="", flush=True)
    print()  # 换行

    messages_list.append(AIMessage(content=assistant_reply))

    messages_list = messages_list[-50:]
