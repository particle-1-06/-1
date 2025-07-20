import os
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量。

# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 如果你需要通过代理端口访问，还需要做如下配置
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'


from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://xiaoai.plus/v1"
)

# 导入所需库
# 注意，此处我们假设你已根据上文配置了 OpenAI API Key，如没有将访问失败
completion = client.chat.completions.create(
    # 调用模型：ChatGPT-4o
    model="gpt-4o",
    # messages 是对话列表
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
print(completion.choices[0].message.content)



from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://xiaoai.plus/v1"
)


def gen_gpt_messages(prompt):
    '''
    构造 GPT 模型请求参数 messages
    
    请求参数：
        prompt: 对应的用户提示词
    '''
    messages = [{"role": "user", "content": prompt}]
    return messages


def get_completion(prompt, model="gpt-4o", temperature = 0):
    '''
    获取 GPT 模型调用结果

    请求参数：
        prompt: 对应的提示词
        model: 调用的模型，默认为 gpt-4o，也可以按需选择 gpt-o1 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~2。温度系数越低，输出内容越一致。
    '''
    response = client.chat.completions.create(
        model=model,
        messages=gen_gpt_messages(prompt),
        temperature=temperature,
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"

print(get_completion("你好"))




#下面是deepseek写出来的一问一答式ai代码

# from openai import OpenAI
# import os

# # 初始化客户端
# client = OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY"),
#     base_url="https://xiaoai.plus/v1"
# )

# # 全局变量保存对话历史
# conversation_history = []

# def gen_gpt_messages(prompt):
#     """构造带历史上下文的 messages 参数"""
#     global conversation_history
#     # 将用户新提问加入历史
#     conversation_history.append({"role": "user", "content": prompt})
#     # 返回完整历史（包含之前所有问答）
#     return conversation_history

# def get_completion(prompt, model="gpt-4o", temperature=0):
#     """获取回复并保存到历史"""
#     global conversation_history
#     response = client.chat.completions.create(
#         model=model,
#         messages=gen_gpt_messages(prompt),
#         temperature=temperature,
#     )
#     if len(response.choices) > 0:
#         # 将AI回复加入历史
#         ai_reply = response.choices[0].message.content
#         conversation_history.append({"role": "assistant", "content": ai_reply})
#         return ai_reply
#     return "generate answer error"

# # 交互式问答循环
# while True:
#     user_input = input("\n用户: ")
#     if user_input.lower() in ["退出", "exit", "quit"]:
#         print("对话结束")
#         break
#     print("AI:", get_completion(user_input))
