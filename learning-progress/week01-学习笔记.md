# Week 1 学习笔记：AI基础 + LLM机制

> 学习日期：2026-04-01（笔记更新：2026-04-13）
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

enc = tiktoken.get_encoding("cl100k_base")  # GPT系列用的编码

text = "你好，LLM！"
tokens = enc.encode(text)
print(f"文本: {text}")
print(f"Token数: {len(tokens)}")
print(f"Token IDs: {tokens}")
```

### 1.5 常见面试题

**Q: 为什么LLM要用Token而不是直接处理字符？**

> Token可以包含完整词义（如"LLM"是一个token），字符级别太细粒度，语义信息少。

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

### 2.2 主流模型 Context Window（2026年最新）

| 模型 | Context Window | 特点 |
|------|---------------|------|
| **Claude Opus 4** | 200K | 最强推理、长上下文 |
| **GPT-4.5** | 128K | 通用能力强 |
| **GPT-4o** | 128K | 性价比高 |
| **GPT-4o-mini** | 128K | 轻量版，极致性价比 |
| **Gemini 3.1 Pro** | 2M | 超长上下文、谷歌生态 |
| **Gemini 2.0 Flash** | 1M | 高速响应 |
| **Kimi (Moonshot)** | 128K/1M | 中文强、超长上下文 |
| **DeepSeek-V3** | 128K | 开源、高性价比 |
| **Qwen-Max** | 128K | 阿里中文强 |
| **Codex 5-4** | 128K | 编程专用 |

> Context Window = 最大输入+输出的Token数

### 2.3 主流模型定价（2026年参考）

| 模型 | 输入价格 | 输出价格 |
|------|---------|---------|
| GPT-4o-mini | $0.15/1M | $0.60/1M |
| GPT-4o | $2.50/1M | $10/1M |
| GPT-4.5 | $75/1M | $150/1M |
| Claude Opus 4 | $15/1M | $75/1M |
| Claude Sonnet 4 | $3/1M | $15/1M |
| Claude Haiku 4 | $0.25/1M | $1.25/1M |
| Gemini 3.1 Pro | $1.25/1M | $5/1M |
| Gemini 2.0 Flash | $0/1M | $0/1M |
| Kimi | ¥0.5/1K | ¥2/1K |
| DeepSeek-V3 | ¥1/1M | ¥2/1M |
| Codex 5-4 | $15/1M | $60/1M |

**选型建议：**
- 开发测试：GPT-4o-mini / Claude Haiku 4（成本极低）
- 生产环境：GPT-4o / Claude Sonnet 4（能力强）
- 复杂推理/编程：Claude Opus 4 / Codex 5-4
- 长文本处理：Claude Opus 4 (200K) / Gemini 3.1 Pro (2M) / Kimi (1M)
- 国内业务：DeepSeek-V3 / Kimi（成本优势）
- 谷歌生态：Gemini 3.1 Pro / Flash

### 2.4 Temperature（温度）

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

### 2.5 常见面试题

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

### 3.2 工程实现：Python（OpenAI SDK）

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# 使用 GPT-4o-mini 作为默认（性价比高）
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True  # 开启流式
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 3.3 工程实现：Python（Anthropic SDK - Claude）

```python
from anthropic import Anthropic

client = Anthropic(api_key="your-api-key")

# Claude 流式输出
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "解释什么是RAG"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### 3.4 工程实现：Go

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

### 任务1：Token计算器（支持多模型成本计算）

```python
"""
Token计算器 - 支持主流模型成本计算
"""

import tiktoken
from typing import Optional

def calculate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    计算文本的Token数

    注意：不同模型使用不同的编码
    - GPT系列: cl100k_base
    - Claude: 估算公式（中英文不同）
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = len(enc.encode(text))

    # Claude 的 tiktoken 估算（中英文比例不同）
    if "claude" in model.lower():
        # Claude 对中文更友好，token数约为 GPT 的 60-70%
        tokens = int(tokens * 0.65)

    return tokens


def calculate_cost(
    input_text: str,
    output_text: str = "",
    model: str = "gpt-4o-mini"
) -> dict:
    """
    计算API调用成本

    模型定价（$/1M tokens）：
    - gpt-4o-mini: input=$0.15, output=$0.60
    - gpt-4o: input=$2.50, output=$10
    - gpt-4.5: input=$75, output=$150
    - claude-opus-4: input=$15, output=$75
    - claude-sonnet-4: input=$3, output=$15
    - claude-haiku-4: input=$0.25, output=$1.25
    - gemini-3.1-pro: input=$1.25, output=$5
    - deepseek-v3: ¥1/1M input, ¥2/1M output
    """
    pricing = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.0),
        "gpt-4.5": (75.0, 150.0),
        "claude-opus-4": (15.0, 75.0),
        "claude-sonnet-4": (3.0, 15.0),
        "claude-haiku-4": (0.25, 1.25),
        "gemini-3.1-pro": (1.25, 5.0),
        "deepseek-v3": (1.0, 2.0),  # ¥/1M
    }

    input_tokens = calculate_tokens(input_text, model)
    output_tokens = calculate_tokens(output_text, model) if output_text else int(input_tokens * 0.3)

    if model in pricing:
        input_price, output_price = pricing[model]
    else:
        input_price, output_price = 1.0, 2.0

    total_cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": total_cost,
        "cost_cny": total_cost * 7.2 if "deepseek" not in model else total_cost,
    }


# 测试
if __name__ == "__main__":
    text = """
    LLM（Large Language Model，大语言模型）是人工智能领域的重要突破。
    它能够理解和生成人类语言，在各种任务中表现出色。
    Agent 是大语言模型的重要应用方向。
    """

    models = [
        "gpt-4o-mini",
        "gpt-4o",
        "claude-sonnet-4",
        "claude-haiku-4",
        "gemini-3.1-pro",
        "deepseek-v3",
    ]

    print("=" * 60)
    print("Token 计算对比")
    print("=" * 60)

    for model in models:
        result = calculate_cost(text, model=model)
        print(f"\n{model}:")
        print(f"  输入 tokens: {result['input_tokens']}")
        print(f"  预估输出 tokens: {result['output_tokens']}")
        print(f"  总 tokens: {result['total_tokens']}")
        if "deepseek" in model:
            print(f"  预估成本: ¥{result['cost_cny']:.6f}")
        else:
            print(f"  预估成本: ${result['cost_usd']:.6f}")
```

### 任务2：带Streaming的多模型对话CLI

```python
"""
多模型对话 CLI - 支持 GPT-4o / Claude / Gemini
"""

from openai import OpenAI
from anthropic import Anthropic
import os

# 初始化客户端
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def chat_openai(model: str = "gpt-4o-mini"):
    """OpenAI 模型对话"""
    print(f"=== Chat with {model} ===")
    print("输入 'quit' 退出\n")

    messages = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        stream = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                print(token, end="", flush=True)
                full_response += token

        messages.append({"role": "assistant", "content": full_response})
        print("\n")


def chat_claude(model: str = "claude-sonnet-4-20250514"):
    """Claude 模型对话"""
    print(f"=== Chat with {model} ===")
    print("输入 'quit' 退出\n")

    messages = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        full_response = ""

        with anthropic_client.messages.stream(
            model=model,
            max_tokens=1024,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response += text

        messages.append({"role": "assistant", "content": full_response})
        print("\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        provider = sys.argv[1].lower()
        model = sys.argv[2] if len(sys.argv) > 2 else None

        if provider == "openai":
            chat_openai(model or "gpt-4o-mini")
        elif provider == "claude":
            chat_claude(model or "claude-sonnet-4-20250514")
        else:
            print("Usage: python chat_cli.py [openai|claude] [model]")
    else:
        print("Usage: python chat_cli.py [openai|claude] [model]")
        print("Examples:")
        print("  python chat_cli.py openai gpt-4o-mini")
        print("  python chat_cli.py claude claude-sonnet-4-20250514")
```

---

## 5. 自测题

| # | 问题 | 答案 |
|---|------|------|
| 1 | Token是什么？1个中文token约等于多少汉字？ | 文本最小处理单元，约1.5-2个汉字 |
| 2 | Temperature=0和1的区别？ | 0=确定输出，1=高随机输出 |
| 3 | Streaming输出的优势？ | 减少等待时间，改善用户体验 |
| 4 | 2026年主流模型有哪些？ | GPT-4.5/4o、Claude Opus 4、Gemini 3.1 Pro 等 |

### 自测结果

| # | 问题 | 我的答案 | 结果 |
|---|------|---------|------|
| 1 | Token是什么？1个中文token约等于多少汉字？ | | ✅/❌ |
| 2 | Temperature=0和1的区别？ | | ✅/❌ |
| 3 | Streaming输出的优势？ | | ✅/❌ |
| 4 | 2026年主流模型有哪些？ | | ✅/❌ |

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
> 以Claude Sonnet 4为例，定价是$3/1M输入Token、$15/1M输出Token。我一般用tiktoken库精确计算：
> ```python
> tokens = len(tiktoken.get_encoding("cl100k_base").encode(prompt))
> cost = tokens * 3 / 1_000_000
> ```
> 另外要考虑：
> 1. 对话历史也要算Token，所以长对话成本会累积
> 2. 可以用缓存+去重减少重复计算
> 3. 要监控单用户/单接口的Token消耗，防止滥用

### Q4: 如何根据场景选择合适的模型？

**回答要点：**
> 2026年的模型选择非常丰富：
> - 简单对话/开发测试：GPT-4o-mini（$0.15/1M输入）或 Claude Haiku 4（$0.25/1M输入）
> - 平衡场景：GPT-4o 或 Claude Sonnet 4（能力强，性价比高）
> - 复杂推理/编程：Claude Opus 4 或 Codex 5-4（最强能力）
> - 超长文档：Gemini 3.1 Pro (2M) 或 Claude Opus 4 (200K)
> - 国内业务：DeepSeek-V3 或 Kimi（成本优势）
>
> 实际经验：70%的任务用中端模型（GPT-4o/Claude Sonnet 4）就够了，只有复杂推理才需要 Opus 级别。

---

## 7. 生产踩坑

| 坑 | 原因 | 解决方案 |
|----|------|---------|
| **Token溢出** | 输入超Context Window | 实现超长文本的自动分片 + 摘要压缩 |
| **Streaming超时** | 网络不稳定/模型慢 | 前端显示loading骨架 + 超时兜底 |
| **成本失控** | 没用缓存+没限流 | 接入Redis缓存 + 按用户/接口限流 |
| **模型幻觉** | Temperature太高 | 生产环境Temperature≤0.7 |
| **模型选型不当** | 用最强模型处理简单任务 | 简单任务用GPT-4o-mini/Claude Haiku 4 |

---

## 8. 检验清单

- [ ] 能口算：1000字中文 ≈ 多少Token（约1333个）
- [ ] 能回答：主流模型的Context Window
- [ ] 能解释：Temperature=0.7 vs 1.0的区别
- [ ] 能实现：带Streaming的LLM调用（OpenAI / Claude API）
- [ ] 能回答：为什么Streaming能改善用户体验
- [ ] 能根据任务选择合适的模型

---

## 9. 本周总结

**学习内容：**
- Token和Tokenizer的原理
- LLM的工作流程和概率本质
- Context Window和Temperature的机制
- Streaming输出的工程实现
- 主流模型选型（GPT-4.5/4o、Claude Opus 4、Gemini 3.1 Pro等）

**关键收获：**
1. LLM本质是"预测下一个词的概率分布"，不是真正的"思考"
2. Token数直接决定成本和Context Window限制
3. Streaming不是炫技，是用户体验和成本控制的基础
4. 模型选型很重要：简单任务用轻量模型，复杂任务用高端模型

**待解决问题：**
1. 如何精确计算中文Token数（不同Tokenizer有差异）
2. 多轮对话的Token累计如何优化

---

## 10. 下周计划

1. 开始 **Week 2：RAG核心概念 + 全链路**
2. 重点加强：RAG为什么能解决LLM幻觉问题
3. 代码练习：实现一个简单的RAG Pipeline

---

*总结时间：2026-04-13*
