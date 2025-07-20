import openai
from openai import OpenAI
import httpx
from httpx import HTTPProxyTransport

# 1. 配置代理
proxy_url = "http://127.0.0.1:7890"  # 替换为你的代理

# 2. 创建带代理的 HTTP 客户端
http_client = httpx.Client(
    transport=HTTPProxyTransport(proxy_url),
    timeout=30.0  # 可选：设置超时时间
)

# 3. 创建 OpenAI 客户端
client = OpenAI(
    api_key="sk-Lv2LPpvp2FEKhv2VSPTjGF408drHpfGTS1KYrumn3IMREHoY",
    http_client=http_client
)

# 4. 测试调用
try:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "你好，请用中文回复"}
        ]
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"错误: {str(e)}")