import base64
from io import BytesIO
#from opencc import OpenCC

from PIL import Image
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ['CUDA_VISIBLE_DEVICES']= '3,2,1,0' # 实际应用没有GPU
import streamlit as st
from langchain_community.vectorstores.chroma import Chroma
import sys
sys.path.append('cnLLM')
from M3e_Embedding import M3EEmbeddings
from Cn_Clip_Embedding import CNCLIPEmbeddings
sys.path.append('RealtimeSTT')
from audio_recorder import AudioToTextRecorder
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Torchaudio's I/O functions now support par-call bakcend dispatch.*")# streamlit版本问题，可忽略
warnings.filterwarnings("ignore", category=UserWarning, message="1Torch was not compiled with flash attention.*")# 可用GPU的提示，忽略

# @st.cache_data
@st.cache_resource
def init():
    # 优先加载定义 Embeddings
    m3e_embedding = M3EEmbeddings()
    clip_embedding = CNCLIPEmbeddings()
    # 加载语音模块
    recorder_config = {
        'spinner': False,
        'model': 'base',
        'language': 'zh',
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 2,
        'post_speech_silence_duration': 0.2,
        'min_length_of_recording': 0,
        'min_gap_between_recordings': 0,
    }
    recorder = AudioToTextRecorder(**recorder_config)
    #cc = OpenCC('t2s')
    return m3e_embedding, clip_embedding, recorder#, cc

@st.cache_resource
def get_vectordb(_m3e_embd, _clip_embd):
    # 向量数据库持久化路径
    docment_persist_directory = 'chromadb/chromadb_docment'
    image_persist_directory = 'chromadb/chromadb_image'
    # 加载数据库
    docment_vectordb = Chroma(
        persist_directory=docment_persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=_m3e_embd
    )
    image_vectordb = Chroma(
        persist_directory=image_persist_directory,
        embedding_function=_clip_embd,
    )
    return docment_vectordb, image_vectordb

# 显示检索结果
def generate_response(input_text, m3e_embedding, clip_embedding):
    # 余弦距离相似度检索
    docment_vectordb, image_vectordb = get_vectordb(m3e_embedding, clip_embedding)
    sim_docs = docment_vectordb.similarity_search(input_text, k=3)
    sim_images = image_vectordb.similarity_search(input_text, k=2)
    return sim_docs, sim_images

def setrecording():
    st.session_state.recording = 0
def resetrecording():
    if st.session_state.recording == 0:
        st.session_state.recording = 1
    else:
        st.session_state.recording = 2

# Streamlit 应用程序界面
if __name__ == '__main__':
    # 初始化模型与向量库
    m3e_embedding, clip_embedding, recorder = init()
    get_vectordb(m3e_embedding, clip_embedding)

    # 加载共享变量用于跟踪对话历史
    if 'voices' not in st.session_state:
        st.session_state.voices = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'text' not in st.session_state:
        st.session_state.text = ''
    if 'recording' not in st.session_state:
        st.session_state.recording = 1
    messages = st.container()#height=1400)
    voices = st.container()

    with st.sidebar:
        st.title('故障检测检索系统')
        selected_method = st.radio(
            "请选择检索模式",
            ["None", "text", "vioce"],
            captions = ["关闭", "文字检索模式", "语音检索模式"])
    
    if selected_method == "vioce":
        # 如果正在录音
        if st.session_state.recording >= 3:
            st.sidebar.info('正在录音...')
            st.sidebar.info('录音结束...')
        if st.session_state.recording == 1:
            st.sidebar.info('正在录音...')
            # text = cc.convert('这是中文，支持詞彙級別的轉換')
            text = recorder.text()
            st.sidebar.info('录音结束...')
            st.session_state.recording = 3
        elif st.session_state.recording == 2:
            st.sidebar.info('正在录音...')
            # st.session_state.text_input = '456'
            st.session_state.text_input = recorder.text()
            st.sidebar.info('录音结束...')
            st.session_state.recording = 4
        if st.session_state.recording == 3:
            st.session_state.text = st.sidebar.text_area('识别结果如下，可在此处修改：', text, key='text_input')
        elif st.session_state.recording == 4:
            st.session_state.text = st.sidebar.text_area('识别结果如下，可在此处修改：', key='text_input')
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.button('重新录入', on_click=resetrecording)
        with col2:
            if st.session_state.recording:
                st.button('确认', on_click=setrecording, key='submitted')

        if not st.session_state.recording:
            # 将用户输入添加到对话历史中
            st.session_state.voices.append({"role": "user", "text": st.session_state.text, "type": "text"})

            # 调用 respond 函数获取回答
            answers, img_answers = generate_response(st.session_state.text, m3e_embedding, clip_embedding)
            # 检查回答是否为 None
            if answers is not None:
                # 将LLM的回答添加到对话历史中
                for i, answer in enumerate(answers):
                    st.session_state.voices.append({"role": "assistant", "text": '第{}项结果：来源：'.format(i+1)
                                                    + answer.metadata['file_path'] + '：' + answer.page_content[:500], "type": "text"})
            if img_answers is not None:
                # 将LLM的回答添加到对话历史中
                for i, answer in enumerate(img_answers):
                    image = base64.b64decode(answer.page_content)
                    image = BytesIO(image)
                    image = Image.open(image)
                    st.session_state.voices.append({"role": "assistant", "text": image, "type": "image"})
            st.session_state.submitted = False
            # text_detected(st.session_state.text)
            # st.write('输入的文字是', st.session_state.text)
            # if st.session_state.text:
            #     st.session_state.voices.append({"role": "user", "text": st.session_state.text})
            #     voices.chat_message("user").write(st.session_state.voices[-1]["text"])

        # 显示整个对话历史
        for voice in st.session_state.voices:
            if voice["role"] == "user":
                voices.chat_message("user").write(voice["text"])
            elif voice["role"] == "assistant":
                if voice["type"] == "image":
                    voices.chat_message("assistant").image(voice["text"], width=200)
                elif voice["type"] == "text":
                    voices.chat_message("assistant").write(voice["text"])
    
    elif selected_method == "text":
        st.session_state.recording = 1
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

    elif selected_method == "None":
        st.session_state.recording = 1
        st.subheader("如需使用，请从左侧选择一个模式")

    # 点击按钮开始录音并直接显示录音结果
    # flag = 0
    # if st.button('Clink me'):
    #     flag = 1
    # # while True:
    # while flag:
    #     text = recorder.text()#process_text)
    #     text_detected(text)
    #     if text:
    #         st.session_state.messages.append({"role": "user", "text": text})
    #         messages.chat_message("user").write(st.session_state.messages[-1]["text"])
    #     # for message in st.session_state.messages:
    #     #     if message["role"] == "user":
    #     #         messages.chat_message("user").write(message["text"])

