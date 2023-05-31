import gradio as gr
import os
import shutil
from datetime import datetime
from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk
from models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


# def get_vs_list():
#     lst_default = ["财务知识"]
#     if not os.path.exists(VS_ROOT_PATH):
#         return lst_default
#     lst = os.listdir(VS_ROOT_PATH)
#     if not lst:
#         return lst_default
#     lst.sort()
#     return lst_default + lst


vs_list = ["财务专家", ]

embedding_model_dict_list = list(embedding_model_dict.keys())

llm_model_dict_list = list(llm_model_dict.keys())

local_doc_qa = LocalDocQA()

flag_csv_logger = gr.CSVLogger()


def get_answer(query, vs_path, history, mode, streaming: bool = STREAMING):
    history = [[None, None]]
    
    if mode == "知识库问答" and vs_path is not None and os.path.exists(vs_path):
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=query, vs_path=vs_path, chat_history=history, streaming=streaming):
            yield history, ""
        else:
            time_now_str = datetime.now().strftime("%Y年%m月%d日%H时%M分%S秒")
            history[-1][-1] += f'(由WhaleMan鲸人在{time_now_str}回答)'
            yield history, ""
    else:
        for answer_result in local_doc_qa.llm.generatorAnswer(prompt=query, history=history,
                                                              streaming=streaming):

            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][-1] = resp + (
                "\n\n当前知识库为空，如需基于知识库进行问答，请先加载知识库后，再进行提问。" if mode == "知识库问答" else "")
            yield history, ""
    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
    flag_csv_logger.flag([query, vs_path, history, mode], username=FLAG_USER_NAME)


def init_model(llm_model: BaseAnswer = None):
    try:
        local_doc_qa.init_cfg(llm_model=llm_model)
        generator = local_doc_qa.llm.generatorAnswer("你好")
        for answer_result in generator:
            print(answer_result.llm_output)
        reply = """模型已成功加载，可以问答"""
        logger.info(reply)
        return reply
    except Exception as e:
        logger.error(e)
        reply = """模型未成功加载，请检查"""
        if str(e) == "Unknown platform: darwin":
            logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
                        " https://github.com/imClumsyPanda/langchain-ChatGLM")
        else:
            logger.info(reply)
        return reply


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}
.qatext textarea {font-size: 18px !important}"""
users_init_message = "🐻答案仅供参考,您做重要决策时请做出独立判断或咨询财务方面的专家"
webui_title = f"""
## 🤖WhaleMan鲸人的智能财务知识问答
### {users_init_message}
"""
default_vs = vs_list[0] if len(vs_list) >= 1 else "为空"

### ⚠️ 答案仅供参考,您做重要决策时请做出独立判断或咨询财务方面的专家


# 初始化消息
args = None
args = parser.parse_args()

args_dict = vars(args)
shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
llm_model_ins = shared.loaderLLM()
llm_model_ins.set_history_len(LLM_HISTORY_LEN)

model_status = init_model(llm_model=llm_model_ins)
init_message = f"""欢迎使用智能财务知识问答,答案来自{default_vs},{model_status}
"""

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    vs_path, file_status, model_status, vs_list = gr.State(
        os.path.join(VS_ROOT_PATH, vs_list[0]) if len(vs_list) >= 1 else ""), gr.State(""), gr.State(
        model_status), gr.State(vs_list)

    gr.Markdown(webui_title)
    with gr.Row():
        with gr.Column(scale=15):
            # users 用于展示问题和当前的使用用户
            users = gr.Textbox(show_label=False, elem_id="users", elem_classes="qatext",
                placeholder="").style(container=False)

            chatbot = gr.Chatbot([[None, None]],
                                    elem_id="chat-box", elem_classes="qatext",
                                    show_label=False).style(height=730, container=True)
            # 输入查询题目和查询状态
            query = gr.Textbox(show_label=False, elem_id="askquestion", elem_classes="qatext",
                placeholder="", ).style(container=False)  # 
            mode = gr.Radio([""], label="", value="知识库问答", visible=False)
            flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
            query.submit(get_answer,
                            [query, vs_path, chatbot, mode,],
                            [chatbot, query])


(demo
 .queue(concurrency_count=3)
 .launch(server_name='0.0.0.0',
         server_port=7860,
         show_api=True,
         share=False,
         inbrowser=False,
         favicon_path="/home/kent/fgpt/langchain-ChatGLM-Text/whale.png"))
