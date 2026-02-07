import os

import gradio as gr
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# description: langchain添加多轮对话记忆功能

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你叫苍井空，是日本著名女演员。"),
    MessagesPlaceholder(variable_name="messages"),
])



model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key=os.getenv("API_KEY"),
)

chain = prompt | model | StrOutputParser()

CSS = """
.main-container {max-width: 1200px; margin: 0 auto; padding: 20px;}
.header-text {text-align: center; margin-bottom: 20px;}
"""


def create_chatbot():
    with gr.Blocks(title="聊天机器人", css=CSS) as demo:
        with gr.Column(elem_classes=["main-container"]):
            gr.Markdown("# 🤖 LangChain智能对话机器人系统", elem_classes=["header-text"])

            chatbot = gr.Chatbot(
                height=500,
                avatar_images=(
                    "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f004.png",
                    "https://cdn.jsdelivr.net/gh/twitter/twemoji@v14.0.2/assets/72x72/1f916.png",
                ),
            )

            msg = gr.Textbox(placeholder="请输入你的消息，然后按回车键发送...", container=False, scale=7)
            submit = gr.Button("发送", scale=1, variant="primary")
            clear = gr.Button("清空", scale=1)

        state = gr.State([])

        async def response(user_msg: str, chat_history: list, messages_list: list):
            if not user_msg.strip():
                yield "", chat_history, messages_list
                return

            messages_list.append(HumanMessage(content=user_msg))
            chat_history = chat_history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": ""}
            ]
            yield "", chat_history, messages_list

            partial = ""
            async for chunk in chain.astream({"messages": messages_list}):
                partial += chunk
                chat_history[-1] = {"role": "assistant", "content": partial}
                yield "", chat_history, messages_list

            messages_list.append(AIMessage(content=partial))

            messages_list = messages_list[-50:]

            # 5) 最终返回（Gradio 需要把新的 state 传回）
            yield "", chat_history, messages_list

        def clear_history():
            return [], "", []

        msg.submit(response, [msg, chatbot, state], [msg, chatbot, state])
        submit.click(response, [msg, chatbot, state], [msg, chatbot, state])
        clear.click(clear_history, outputs=[chatbot, msg, state])

    return demo


demo = create_chatbot()

demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
