
from openai import OpenAI

client = OpenAI(api_key="sk-9895344488ad43c989b6d66587f23ad9", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "使用弱智吧的语气,给我说一下你认为的世界上top10 的编程语言"},
    ],
    stream=False

)

# 打印响应内容 response 是stream类型，需要进行处理
# for chunk in response:
#     print("%s" % chunk.choices[0].delta.content)

print(response.choices[0].message.content)
