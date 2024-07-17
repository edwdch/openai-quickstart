import gradio as gr
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


def initialize_sales_bot(vector_store_dir: str = "customer_service_specialist_data"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # 定义自定义 prompt
    custom_prompt_template = """你是 CargoSmart 的客服专员，你的回复应尽可能贴近 CargoSmart 客服专员的回答风格。
    你将只回答关于 CargoSmart 业务相关的问题，当话题超出范围时，请礼貌地回避这个问题。
    你需要使用提供的上下文来回答问题。如果上下文中没有任何信息可用来回答问题，请礼貌地向客户解释你无法回答这个问题。

    上下文: {context}

    问题: {question}

    回答:"""

    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )

    global SALES_BOT
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    SALES_BOT = RetrievalQA(
        retriever=db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8}),
        combine_documents_chain=qa_chain)
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")

    ans = SALES_BOT({"query": message})

    # 由于已经定义了无法回答时的话术，所以不需要再判断是否有关联的上下文
    # 可以直接返回 RetrievalQA combine_documents_chain 整合的结果

    print(f"[result]{ans['result']}")
    print(f"[source_documents]{ans['source_documents']}")
    return ans["result"]


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="CargoSmart Helpdesk",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=False, server_name="0.0.0.0")


if __name__ == "__main__":
    # 初始化 CargoSmart 客服机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
