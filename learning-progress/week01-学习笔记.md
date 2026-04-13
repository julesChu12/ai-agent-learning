# Week 1 学习笔记：AI基础 + LLM机制

> 学习日期：2026-04-01
> 学习时长：约2-3小时（概念学习+代码实践）
> 完成度：100%

---

## 先给结论

**本周最关键的3个点：**

1. **Token是LLM的最小处理单元** — 理解Token才能理解LLM的一切限制（Context Window、成本、延迟）
2. **LLM本质是概率模型** — 它不是在"思考"，而是在预测下一个词的概率分布
3. **Streaming是工程刚需** — 不是炫技，是用户体验和成本控制的基础

---

## 1. Token 和 Tokenizer

### 1.1 原理

**Token = 文本的最小处理单元**

| 语言 | 分词方式 | 1 Token ≈ |
|------|---------|----------|
| 英文 | 单词或子词 | 0.75个词 |
| 中文 | 2-3个字符 | 1.5-2个字 |

### 1.2 Tokenizer的边界情况（常考）

```
"你好呀" → [你, 好, 呀] → 3个token（中文按字符分）
"LLM" → [LLM] → 1个token
"2024-01-01" → [2024, -, 01, -, 01] → 5个token
```

### 1.3 为什么LLM要用Token而不是直接处理字符？

- 字符级别太细粒度，语义信息少
- Token可以包含完整词义（如"LLM"是一个token）
- 子词分词（BPE/WordPiece）兼顾覆盖率和效率

### 1.4 工程实现

```python
# 用tiktoken计算Token
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4/3.5用的编码

text = "你好，LLM！"
tokens = enc.encode(text)
print(f"文本: {text}")
print(f"Token数: {len(tokens)}")
print(f"Token IDs: {tokens}")
```

### 1.5 常见面试题

**Q: 为什么LLM要用Token而不是直接处理字符？**

> Token可以包含完整词义（如"LLM"是一个token），字符级别太细粒度、语义信息少。

---

## 2. LLM 工作流程

### 2.1 原理图

```
输入文本 → Tokenize → Embedding → Transformer → Softmax → De-tokenize → 输出文本
                        ↑
                   预训练权重
```

**核心理解：LLM不是在"思考"，而是在"预测下一个词"**

```
输入: "今天天气"
预测: "很" (概率0.3), "不" (概率0.2), "不错" (概率0.15), ...
输出: "很" → "今天天气很..."
```

### 2.2 Context Window（上下文窗口）

| 模型 | Context Window |
|------|---------------|
| GPT-3.5-turbo | 16K |
| GPT-4 | 128K |
| Claude 3 | 200K |
| 通义千问 | 32K |

> Context Window = 最大输入+输出的Token数

### 2.3 Temperature（温度）

| 值 | 效果 |
|----|------|
| 0 | 几乎确定，每次输出相同 |
| 0.7 | 平衡创造性和确定性 |
| 1.0 | 高随机性，可能胡说八道 |

**数学原理：**
```
Temperature越高 → softmax的熵越大 → 分布越平坦 → 选择越随机
Temperature越低 → softmax的熵越小 → 分布越尖锐 → 选择越确定
```

### 2.4 常见面试题

**Q: Temperature和Top-p在实际应用中如何调优？**

| 任务类型 | Temperature推荐值 |
|---------|------------------|
| 确定性任务（翻译、代码、分类） | 0-0.3 |
| 平衡任务（聊天、问答） | 0.5-0.7 |
| 创意任务（写诗、头脑风暴） | 0.8-1.0 |

---

## 3. Streaming（流式输出）

### 3.1 Streaming vs 非Streaming

| 非Streaming | Streaming |
|------------|----------|
| 等待LLM生成完整答案（可能30秒+） | 边生成边返回（首Token 200ms） |
| 用户等待焦虑 | 用户看到打字效果，体验好 |
| 必须在生成完成后才能显示 | 可以实时显示思考过程 |

### 3.2 工程实现：Python

```python
from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True  # 开启流式
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 3.3 工程实现：Go

```go
func streamChat(ctx context.Context, prompt string) (<-chan string, <-chan error) {
    respCh := make(chan string)
    errCh := make(chan error, 1)

    go func() {
        defer close(respCh)
        defer close(errCh)

        req := &Request{Prompt: prompt}
        stream, err := client.CreateChatCompletionStream(ctx, req)
        if err != nil {
            errCh <- err
            return
        }
        defer stream.Close()

        for {
            select {
            case <-ctx.Done():
                return
            default:
                resp, err := stream.Recv()
                if errors.Is(err, io.EOF) {
                    return
                }
                if err != nil {
                    errCh <- err
                    return
                }
                respCh <- resp.Choices[0].Delta.Content
            }
        }
    }()

    return respCh, errCh
}
```

---

## 4. 代码任务

### 任务1：Token计算器

```python
# 用tiktoken计算任意文本的token数
# 同时计算成本（GPT-3.5-turbo: $0.0015/1K token）

import tiktoken

def calculate_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def calculate_cost(text: str, model: str = "gpt-3.5-turbo") -> float:
    tokens = calculate_tokens(text)
    # GPT-3.5-turbo: $0.0015/1K input, $0.002/1K output
    return tokens * 0.0015 / 1000

# 测试
text = """
LLM（Large Language Model，大语言模型）是人工智能领域的重要突破。
它能够理解和生成人类语言，在各种任务中表现出色。
"""
print(f"Token数: {calculate_tokens(text)}")
print(f"预估成本: ${calculate_cost(text):.6f}")
```

### 任务2：带Streaming的对话CLI

```python
# 实现一个命令行聊天工具，支持流式输出

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_loop():
    print("=== LLM Chat CLI ===")
    print("输入 'quit' 退出\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        print("\nAssistant: ", end="", flush=True)
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}],
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                print(token, end="", flush=True)
                full_response += token
        print("\n")

chat_loop()
```

---

## 5. 自测题

| # | 问题 | 答案 |
|---|------|------|
| 1 | Token是什么？1个中文token约等于多少汉字？ | 文本最小处理单元，约1.5-2个汉字 |
| 2 | Temperature=0和1的区别？ | 0=确定输出，1=高随机输出 |
| 3 | Streaming输出的优势？ | 减少等待时间，改善用户体验 |

### 自测结果

| # | 问题 | 我的答案 | 结果 |
|---|------|---------|------|
| 1 | Token是什么？1个中文token约等于多少汉字？ | | ✅/❌ |
| 2 | Temperature=0和1的区别？ | | ✅/❌ |
| 3 | Streaming输出的优势？ | | ✅/❌ |

---

## 6. 面试问题与回答

### Q1: Token和Tokenizer的关系是什么？

**普通回答：**
> Token是文本的最小单元，Tokenizer是把文本转成Token的工具。

**更强回答：**
> Token是LLM的处理单元，中英文分词方式不同——英文按子词分（"tokenization"可能分成["token", "ization"]），中文按字符分。Tokenizer的算法主要有BPE和WordPiece，核心思想是用统计方法找到最常见的字符组合，形成一个词汇表。为什么不直接用字符？是因为字符级别太细、语义信息少，比如"LLM"三个字符组合才有意义。另外，Token数直接影响LLM的成本和Context Window限制，所以生产环境中精确计算Token是必须的。

### Q2: Temperature和Top-p在实际应用中如何调优？

**普通回答：**
> Temperature控制随机性，高了会胡说八道，低了会很死板。

**更强回答：**
> Temperature和Top-p都是控制LLM输出多样性的参数。Temperature本质是调整softmax的熵，0时几乎每次输出相同（适合翻译、代码等确定性任务），1.0时高度随机（适合创意写作）。实际经验是：
> - 确定性任务（翻译、代码、分类）：Temperature 0-0.3
> - 平衡任务（聊天、问答）：Temperature 0.5-0.7
> - 创意任务（写诗、头脑风暴）：Temperature 0.8-1.0
>
> Top-p是核采样，我一般固定0.9，让模型在前90%概率的词中选择，这样比固定Top-k更自然。生产环境建议先用Temperature=0.7、Top-p=0.9调优，然后根据效果调整。

### Q3: 在生产环境中如何计算Token成本？

**更强回答：**
> GPT-3.5-turbo定价是$0.0015/1K输入Token、$0.002/1K输出Token。我一般用tiktoken库精确计算：
> ```python
> tokens = len(tiktoken.get_encoding("cl100k_base").encode(prompt))
> cost = tokens * 0.0015 / 1000
> ```
> 另外要考虑：
> 1. 对话历史也要算Token，所以长对话成本会累积
> 2. 可以用缓存+去重减少重复计算
> 3. 要监控单用户/单接口的Token消耗，防止滥用

---

## 7. 生产踩坑

| 坑 | 原因 | 解决方案 |
|----|------|---------|
| **Token溢出** | 输入超Context Window | 实现超长文本的自动分片 + 摘要压缩 |
| **Streaming超时** | 网络不稳定/模型慢 | 前端显示loading骨架 + 超时兜底 |
| **成本失控** | 没用缓存+没限流 | 接入Redis缓存 + 按用户/接口限流 |
| **模型幻觉** | Temperature太高 | 生产环境Temperature≤0.7 |

---

## 8. 检验清单

- [ ] 能口算：1000字中文 ≈ 多少Token（约1333个）
- [ ] 能回答：GPT-3.5-turbo的Context Window是多少（16K）
- [ ] 能解释：Temperature=0.7 vs 1.0的区别
- [ ] 能实现：带Streaming的LLM调用（OpenAI API）
- [ ] 能回答：为什么Streaming能改善用户体验

---

## 9. 本周总结

**学习内容：**
- Token和Tokenizer的原理
- LLM的工作流程和概率本质
- Context Window和Temperature的机制
- Streaming输出的工程实现

**关键收获：**
1. LLM本质是"预测下一个词的概率分布"，不是真正的"思考"
2. Token数直接决定成本和Context Window限制
3. Streaming不是炫技，是用户体验和成本控制的基础

**待解决问题：**
1. 如何精确计算中文Token数（不同Tokenizer有差异）
2. 多轮对话的Token累计如何优化

---

## 10. 下周计划

1. 开始 **Week 2：RAG核心概念 + 全链路**
2. 重点加强：RAG为什么能解决LLM幻觉问题
3. 代码练习：实现一个简单的RAG Pipeline

---

*总结时间：2026-04-01*
