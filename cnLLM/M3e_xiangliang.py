import os
import shutil
import re
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from M3e_Embedding import M3EEmbeddings
from langchain_community.vectorstores.chroma import Chroma

# 获取folder_path下所有文件路径，储存在file_paths里
file_paths = []
folder_path = 'data/docment_data'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:2])

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []
for file_path in file_paths:
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))

# 下载文件并存储到text
texts = []
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
for loader in loaders: 
    pdf_pages = loader.load()
    for pdf_page in pdf_pages:
        pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
    texts.extend(pdf_pages)
'''text = texts[1]
print(f"每一个元素的类型：{type(text)}.", 
    f"该文档的描述性数据：{text.metadata}", 
    f"查看该文档的内容:\n{text.page_content[0:]}", 
    sep="\n------\n")'''

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)

split_docs = text_splitter.split_documents(texts)
print(len(split_docs))


## 构建Chroma向量库
# 使用M3e Embedding
embedding = M3EEmbeddings()
persist_directory = 'chromadb/chromadb_docment'

# 删除旧的数据库文件（如果文件夹中有文件的话）
if os.path.isdir('chromadb/chromadb_docment'):
    shutil.rmtree('chromadb/chromadb_docment')  

vectordb = Chroma.from_documents(
    documents=split_docs[:],
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
#vectordb.persist()
print(f"向量库中存储的数量：{vectordb._collection.count()}")

## 向量搜索测试
# 余弦距离相似度检索
question="深度学习"
sim_docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(sim_docs)}")
for i, sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

# 最大边际相关性检索MMR
mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)
for i, sim_doc in enumerate(mmr_docs):
    print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")