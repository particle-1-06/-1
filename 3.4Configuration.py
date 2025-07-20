# 搭建并使用向量数据库
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from dotenv import load_dotenv, find_dotenv


# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 如果你需要通过代理端口访问，你需要如下配置
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

# 获取folder_path下所有文件路径，储存在file_paths里
file_paths = []
folder_path = '/workspaces/test_codespace/llm-universe/data_base/knowledge_db'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:3])

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

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

for loader in loaders: texts.extend(loader.load())

text = texts[1]
# print(f"每一个元素的类型：{type(text)}.", 
#     f"该文档的描述性数据：{text.metadata}", 
#     f"查看该文档的内容:\n{text.page_content[0:]}", 
#     sep="\n------\n")

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)

split_docs = text_splitter.split_documents(texts)
# print(split_docs)

# 使用 OpenAI Embedding
from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
# 使用百度千帆 Embedding
# from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
# 使用我们自己封装的智谱 Embedding，需要将封装代码下载到本地使用
# from zhipuai_embedding import ZhipuAIEmbeddings

# 定义 Embeddings
embedding = OpenAIEmbeddings(
    api_key=os.getenv("API_KEY"),  # ✅ 你的 API Key
    base_url="https://xiaoai.plus/v1",  # 如果你用 OpenAI 官方直接服务就删掉这行
    model="text-embedding-3-small"  # 或 "text-embedding-3-large"
) 
# embedding = ZhipuAIEmbeddings()
# embedding = QianfanEmbeddingsEndpoint()

# 定义持久化路径
persist_directory = '/workspaces/test_codespace/llm-universe/data_base/vector_db/chroma'

import shutil
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

from langchain_community.vectorstores import Chroma
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")


question="什么是大语言模型"
sim_docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(sim_docs)}")
# for i, sim_doc in enumerate(sim_docs):
#     print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
# mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)
# for i, sim_doc in enumerate(mmr_docs):
#     print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

