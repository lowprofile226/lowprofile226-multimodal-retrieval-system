import base64
import os
import shutil
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from Cn_Clip_Embedding import CNCLIPEmbeddings
from langchain_community.vectorstores.chroma import Chroma


# 图片
# 获取folder_path下所有文件路径，储存在file_paths里
image_paths = []
image_folder_path = 'data/image_data'
for root, dirs, files in os.walk(image_folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        image_paths.append(file_path)
print(image_paths[:2])


## 构建Chroma向量库
clip_embd = CNCLIPEmbeddings()
persist_directory = 'chromadb/chromadb_image'
# 删除旧的数据库文件（如果文件夹中有文件的话）
if os.path.isdir('chromadb/chromadb_image'):
    shutil.rmtree('chromadb/chromadb_image')
langchain_chroma = Chroma(
    embedding_function=clip_embd,
    persist_directory=persist_directory,
)
langchain_chroma.add_images(uris=image_paths)
# langchain_chroma.persist()
print(f"向量库中存储的数量：{langchain_chroma._collection.count()}")


## 向量搜索
# 余弦距离相似度检索
question="变压器"
sim_docs = langchain_chroma.similarity_search(question,k=3)
print(f"检索到的内容数：{len(sim_docs)}")
for i, sim_doc in enumerate(sim_docs):
    with open('test_image_output/base64_similarity_search_{}.jpg'.format(i),'wb') as file:
        img = base64.b64decode(sim_doc.page_content)
        file.write(img)

# 最大边际相关性检索MMR
# from IPython.display import display, HTML
mmr_docs = langchain_chroma.max_marginal_relevance_search(question,k=1)
for i, sim_doc in enumerate(mmr_docs):
    with open('test_image_output/base64_max_marginal_relevance_search_{}.jpg'.format(i),'wb') as file:
        img = base64.b64decode(sim_doc.page_content)
        file.write(img)