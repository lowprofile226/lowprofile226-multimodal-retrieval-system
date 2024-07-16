import base64
from io import BytesIO

from PIL import Image
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ['CUDA_VISIBLE_DEVICES']= '3,2,1,0' # 实际应用没有GPU
import streamlit as st
from langchain_community.vectorstores.chroma import Chroma
from M3e_Embedding import M3EEmbeddings
from Cn_Clip_Embedding import CNCLIPEmbeddings

# @st.cache_data
@st.cache_resource
def init():
    # 优先加载定义 Embeddings
    m3e_embedding = M3EEmbeddings()
    clip_embedding = CNCLIPEmbeddings()
    return m3e_embedding, clip_embedding

def get_vectordb(embedding, clip_embd):
    # 向量数据库持久化路径
    docment_persist_directory = 'chromadb/chromadb_docment'
    image_persist_directory = 'chromadb/chromadb_image'
    # 加载数据库
    docment_vectordb = Chroma(
        persist_directory=docment_persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    image_vectordb = Chroma(
        persist_directory=image_persist_directory,
        embedding_function=clip_embd,
    )
    return docment_vectordb, image_vectordb

# 显示结果
def generate_response(input_text, m3e_embedding, clip_embedding):
    # 余弦距离相似度检索
    docment_vectordb, image_vectordb = get_vectordb(m3e_embedding, clip_embedding)
    sim_docs = docment_vectordb.similarity_search(input_text, k=3)
    sim_images = image_vectordb.similarity_search(input_text, k=2)
    return sim_docs, sim_images

# Streamlit 应用程序界面
def main():
    st.title('故障检测检索系统')
    m3e_embedding, clip_embedding = init()
    
    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    messages = st.container()#height=1400)
    if prompt := st.chat_input("Say something"):
        # st.info('running...')
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        # 调用 respond 函数获取回答
        answers, img_answers = generate_response(prompt, m3e_embedding, clip_embedding)
        # 检查回答是否为 None
        if answers is not None:
            # 将LLM的回答添加到对话历史中
            for i, answer in enumerate(answers):
                st.session_state.messages.append({"role": "assistant", "text": '第{}项结果：来源：'.format(i+1)
                                                + answer.metadata['file_path'] + '：' + answer.page_content[:500]})
        if img_answers is not None:
            # 将LLM的回答添加到对话历史中
            for i, answer in enumerate(img_answers):
                image = base64.b64decode(answer.page_content)
                image = BytesIO(image)
                image = Image.open(image)
                st.session_state.messages.append({"role": "assistant", "text": image})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])


main() 
