import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 如果你需要通过代理端口访问，你需要如下配置
# os.environ['HTTPS_PROXY'] = 'https://xiaoai.plus'
# os.environ["HTTP_PROXY"] = 'https://xiaoai.plus/v1'

def openai_embedding(text: str, model: str=None):
    # 获取环境变量 OPENAI_API_KEY
    api_key=os.environ['OPENAI_API_KEY']
    client = OpenAI(
        api_key=api_key,
        base_url="https://xiaoai.plus/v1"
        )

    # embedding model：'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'
    if model == None:
        model="text-embedding-3-small"

    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response

response = openai_embedding(text='要生成 embedding 的输入文本，字符串形式。')

# print(response)
# print(f'返回的embedding类型为：{response.object}')
# print(f'embedding长度为：{len(response.data[0].embedding)}')
# print(f'embedding（前10）为：{response.data[0].embedding[:10]}')
# print(f'本次embedding model为：{response.model}')
# print(f'本次token使用情况为：{response.usage}')


