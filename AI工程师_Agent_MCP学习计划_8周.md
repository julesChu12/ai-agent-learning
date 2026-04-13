# AI工程师学习计划：Agent/MCP专项（8周）

> 制定依据：Go后端背景 + AI基础 + Agent/MCP待提升
> 时间投入：每天1-2小时
> 
> **知识测试结果：3/5（薄弱点：Q4平台差异、Q5幂等性）**

---

## 📊 测试结果与学习重点

| # | 题目 | 你的答案 | 正确答案 | 学习重点 |
|---|------|---------|---------|---------|
| Q1 | ReAct范式 | B | ✅ B | 已掌握 |
| Q2 | MCP tools定义 | A | ✅ A | 已掌握 |
| Q3 | update_state | B | ✅ B | 已掌握 |
| **Q4** | Coze vs Dify | 知识盲区 | B（Dify私有化强） | ⭐ **Week 8重点** |
| **Q5** | MCP幂等性 | B | ❌ A（唯一请求ID） | ⭐ **Week 7重点** |

---

## 📋 职位核心技能对照

| 职位要求 | 对应学习模块 | 优先级 |
|---------|-------------|-------|
| Agent平台/应用开发 | Week 3-6 LangChain/LangGraph | ⭐⭐⭐ |
| RAG、Agent机制 | Week 1-2 RAG深度 | ⭐⭐⭐ |
| Coze/Dify/LangFlow | Week 7-8 平台实战 | ⭐⭐⭐ |
| LangChain等开源系统 | Week 5-6 LangChain深入 | ⭐⭐⭐ |
| Prompt工程优化 | Week 3 Prompt工程 | ⭐⭐ |
| 架构设计（千万级） | Week 7-8 系统设计 | ⭐⭐ |
| Python开发 | Week 1 Python高级（Go工程师补充） | ⭐ |

---

## 🚀 8周学习路径

### Week 1-2：RAG核心 + 向量数据库

**目标：** 掌握RAG全链路，能独立实现RAG系统

#### Week 1：RAG基础

**知识点：**
- Embedding模型对比（Text2Vec、BGE、M3E）
- 分块策略（Fixed-size、Semantic、Recursive Character）
- 向量检索基础（余弦相似度、ANN算法）
- LangChain RAG Chain

**实战任务：**
```
用 LangChain + Chroma 构建一个 RAG 问答系统
输入：PDF文档
输出：基于文档内容的智能问答

关键代码路径：
1. PDF加载 → 2. 文本分块 → 3. Embedding → 4. 存入向量库
→ 5. 检索 → 6. 上下文组装 → 7. LLM生成
```

**资源：**
- LangChain RAG文档：https://python.langchain.com/docs/tutorials/rag/
- BGE模型：https://github.com/FlagOpen/FlagEmbedding

---

#### Week 2：向量数据库进阶

**知识点：**
- Chroma（轻量级/开发用）vs Milvus（生产级）
- 索引类型：IVF、HNSW、DiskANN
- 重排序：BGE-Reranker
- 混合检索：稀疏 + 稠密

**实战任务：**
```
对比实验（100万向量规模）：
1. Chroma 单机 vs Milvus 集群
2. 测试指标：召回率、QPS、内存占用
3. 选型结论文档
```

**资源：**
- Milvus官方文档：https://milvus.io/docs
- Chroma文档：https://docs.trychroma.com/

---

### Week 3-4：Prompt工程 + Agent范式

**目标：** 掌握主流Prompt范式，理解Agent核心机制

#### Week 3：Prompt工程

**四大范式：**

| 范式 | 核心思想 | 适用场景 |
|------|---------|---------|
| **Few-shot** | 通过示例引导输出 | 格式固定的任务 |
| **CoT (Chain of Thought)** | 展示推理过程 | 复杂逻辑推理 |
| **ReAct** | 推理 + 行动交替 | 工具调用场景 |
| **Role Play** | 设定人设和风格 | 客服、角色扮演 |

**ReAct详解：**
```
Thought: 我需要先查天气
Action: search_weather(city="北京")
Observation: 天气晴，温度25度
Thought: 根据天气，建议穿薄外套
Action: generate_response(text="建议穿薄外套")
Observation: 生成完成
```

**实战任务：**
```python
# 用 ReAct 范式实现天气助手
# 支持：查天气 → 查穿衣建议 → 生成回答
```

**资源：**
- Prompt Engineering Guide：https://www.promptingguide.ai/
- ReAct论文：https://arxiv.org/abs/2210.03629

---

#### Week 4：Agent核心机制

**Agent公式：**
```
Agent = LLM + Planning + Memory + Tools
```

**四大模块：**

| 模块 | 作用 | 关键技术 |
|------|------|---------|
| **Planning** | 子目标分解、任务规划 | ReAct、Reflexion |
| **Memory** | 短期/长期记忆 | ConversationBuffer、VectorStore |
| **Tools** | 调用外部能力 | Function Calling、Tool Call |
| **Multi-Agent** | 多Agent协作 | CrewAI、LangGraph |

**Function Calling原理：**
```
1. LLM 输出结构化 JSON（包含函数名+参数）
2. 解析 JSON → 调用实际函数
3. 函数结果 → 回传给 LLM
4. LLM 继续生成或再次调用
```

**实战任务：**
```
实现一个 "研究助手Agent"：
1. 接收研究主题
2. 搜索 arXiv 论文（工具调用）
3. 提取摘要和关键信息
4. 生成研究综述
```

---

### Week 5-6：LangChain / LangGraph 深入

**目标：** 掌握LCEL表达式，能用LangGraph构建复杂Agent

#### Week 5：LangChain 进阶

**LCEL（LangChain Expression Language）：**
```python
# 链式组合
chain = prompt | model | output_parser

# 并行执行
chain = RunnableParallel({
    "context": retrieval_chain,
    "question": lambda x: x["question"]
}) | combine_contexts | model
```

**核心组件：**
- `LLMChain`：基础链
- `SequentialChain`：顺序执行
- `RouterChain`：条件路由
- `ConversationChain`：对话支持
- `ConversationalRetrievalChain`：RAG对话

**实战任务：**
```python
# 构建多步骤 RAG Chain：
# 1. 检索相关文档
# 2. BGE-Reranker 重排序
# 3. 组装上下文
# 4. LLM 生成答案
# 5. Pydantic OutputParser 验证输出格式
```

**资源：**
- LangChain LCEL文档：https://python.langchain.com/docs/concepts/lcel/
- LangChain GitHub：https://github.com/langchain-ai/langchain

---

#### Week 6：LangGraph 实战

**StateGraph 核心概念：**
```
StateGraph = State + Nodes + Edges + ConditionalEdges

- State: 共享状态字典（dict）
- Node: 处理函数（输入state，输出state）
- Edge: 状态转换（无条件跳转）
- ConditionalEdge: 条件分支（根据state选择下一节点）
```

**示例：ReAct Agent in LangGraph：**
```python
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: List[BaseMessage]
    observation: str

def think_node(state):
    # LLM 推理下一步 action
    ...

def action_node(state):
    # 执行工具调用
    ...

def should_continue(state):
    # 判断是否继续循环
    return "continue" if needs_more_steps else "end"

graph = StateGraph(AgentState)
graph.add_node("think", think_node)
graph.add_node("action", action_node)
graph.add_conditional_edges("think", should_continue, {...})
graph.set_entry_point("think")
graph.set_finish_point("end")
```

**实战任务：**
```
用 LangGraph 实现完整 Agent：
1. 研究助手（搜索 → 阅读 → 摘要 → 报告）
2. 支持暂停/恢复（checkpoint）
3. 支持动态更新状态（update_state）
```

**资源：**
- LangGraph文档：https://langchain-ai.github.io/langgraph/
- LangGraph GitHub：https://github.com/langchain-ai/langgraph

---

### Week 7：MCP协议 + Skill开发

**目标：** 理解MCP协议，能开发自定义Skill

> ⚠️ **Q5薄弱点重点：MCP幂等性 = 唯一请求ID追踪，不是"返回相同结果"**

#### MCP 核心概念

```
MCP (Model Context Protocol) = Transport + Protocol + Session

三层架构：
┌─────────────────────────────────┐
│         Host (LLM App)          │  ← Coze Bot / Dify
├─────────────────────────────────┤
│        MCP Server               │  ← 你的 Skill 实现
├─────────────────────────────────┤
│     External Resources          │  ← 数据库 / API / 文件
└─────────────────────────────────┘
```

**MCP 四大能力：**

| 能力 | 用途 |
|------|------|
| **Tools** | LLM 可调用的外部函数 |
| **Resources** | 文档/文件/数据库访问 |
| **Prompts** | 模板化提示词复用 |
| **二级制** | 图片/音频等二进制数据 |

**Tool 定义示例：**
```json
{
  "name": "search_arxiv",
  "description": "搜索 arXiv 论文",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "limit": {"type": "integer", "default": 10}
    },
    "required": ["query"]
  }
}
```

#### ⚠️ 幂等性设计（Q5重点）

```python
# 错误理解：幂等性 = 每次返回相同结果
# 正确理解：幂等性 = 相同请求ID只处理一次

class IdempotentTool:
    def __init__(self):
        self.processed_ids = set()  # 已处理请求ID缓存
    
    def invoke(self, request_id: str, params: dict) -> dict:
        # 1. 检查是否已处理
        if request_id in self.processed_ids:
            return {"status": "duplicate", "cached": True}
        
        # 2. 首次处理，记录ID
        result = self._do_process(params)
        self.processed_ids.add(request_id)
        
        # 3. 返回结果（可以不同，只要处理逻辑幂等）
        return {"request_id": request_id, "result": result}
```

#### Coze Skill 开发

**开发流程：**
```
1. 编写 Skill Manifest（YAML/JSON）
2. 实现 Tool Handler（Python/JS）
3. 配置认证（OAuth/API Key）
4. 本地测试
5. 发布到 Coze 平台
```

**Dify Tool 开发：**

```python
# Dify Python Tool 节点
class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    description = "搜索学术论文"

    def invoke(self, parameters: dict) -> dict:
        query = parameters.get("query")
        # 实现搜索逻辑
        results = search_arxiv(query)
        return {"results": results}
```

#### 实战任务：**
```
开发一个 "论文助手 MCP Skill"：
1. 搜索 arXiv 论文（tool）
2. 获取论文摘要（resource）
3. 生成引用格式（prompt template）
4. 部署到 Coze Bot 或 Dify 工作流
```

---

### Week 8：平台实战 + 项目整合

**目标：** 独立完成可展示的项目

> ⚠️ **Q4薄弱点重点：Coze vs Dify 核心区别**
> - **Coze**：多Agent编排能力更强，适合企业级AI应用
> - **Dify**：更擅长私有化部署，适合数据敏感场景

#### Coze 平台实战

**核心定位：企业级AI应用平台，多Agent编排更强**

| 特性 | Coze | Dify |
|------|------|------|
| 部署方式 | 云服务为主 | 私有化部署为主 |
| 多Agent | ✅ 强（Bot间协作） | ⚠️ 较弱 |
| 私有化 | ⚠️ 有限支持 | ✅ 完全支持 |
| 插件生态 | ✅ 丰富 | ⚠️ 有限 |
| 数据安全 | ⚠️ 数据上云 | ✅ 完全可控 |

**核心能力：**
- Bot 创建和配置
- 工作流编排（Workflow）
- 插件开发
- Memory（会话记忆）
- 变量和查询参数

**企业级应用场景：**
```
1. 客服机器人 + 知识库 RAG
2. 业务审批助手 + 多Agent协作
3. 数据分析助手 + SQL Tool
```

#### Dify 平台实战

**核心能力：**
- 私有化部署
- 数据集管理（知识库）
- 工作流编排
- API 发布
- 监控和日志

**企业级应用场景：**
```
1. 内部知识库问答
2.合同审查助手
3. 代码审查助手
```

#### 最终项目

**项目选择（任选其一）：**

| 项目 | 技术栈 | 难度 |
|------|--------|------|
| **A. 论文助手Agent** | LangGraph + ArXiv API + MCP | ⭐⭐⭐ |
| **B. 销售数字人** | Coze + 多Agent + 企微 | ⭐⭐⭐ |
| **C. RAG知识库系统** | Dify + Milvus + BGE | ⭐⭐ |

**交付物：**
```
1. 项目代码（GitHub）
2. 架构设计文档
3. 部署指南
4. 效果演示截图/视频
```

---

## 📅 每周时间分配（每天1-2小时）

| 阶段 | 工作日（1h/天） | 周末（2h/天） |
|------|----------------|---------------|
| **上午** | 通勤/午休刷知识点 | 周末可做项目实战 |
| **晚间** | 30min 视频 + 30min 实践 | 1.5h 实战 + 0.5h 总结 |

---

## 🛠️ 学习资源清单

### RAG + 向量库
- LangChain RAG教程：https://python.langchain.com/docs/tutorials/rag/
- Milvus官方文档：https://milvus.io/docs
- RAG综述论文：https://arxiv.org/abs/2312.10997

### Prompt工程
- Prompt Engineering Guide：https://www.promptingguide.ai/
- Anthropic Prompt教程：https://docs.anthropic.com/

### LangChain/LangGraph
- LangGraph文档：https://langchain-ai.github.io/langgraph/
- LangChain中文文档：https://python.langchain.com.cn/

### Agent开发
- OpenAI Function Calling：https://platform.openai.com/docs/guides/function-calling
- LangGraph Agents教程：https://langchain-ai.github.io/langgraph/tutorials/

### MCP协议
- MCP官方文档：https://modelcontextprotocol.io/
- Coze Skill开发：https://www.coze.cn/docs/guides/overview

### 平台
- Dify官方文档：https://dify.ai/docs
- Coze文档：https://www.coze.cn/docs/

---

## ⚠️ 常见误区

| 误区 | 正确做法 |
|------|---------|
| 学完再实践 | 每学一个知识点，立刻动手实践 |
| 追求深度理论 | 以"能用起来"为目标，原理后续补 |
| 忽略Go背景 | Go的并发模型 → 类比到Multi-Agent通信 |
| 只看不写 | 每天必须保证30分钟编码时间 |

---

## 🎯 里程碑检查

| 周次 | 完成标志 | 自测题 |
|------|---------|-------|
| Week 1-2 | 能独立实现完整RAG系统 | 能说出Embedding选型理由 |
| Week 3-4 | 能用ReAct范式实现Tool Call | 能解释Q1-Q3原理 |
| Week 5-6 | 能用LangGraph实现自定义Agent | 能回答update_state作用 |
| Week 7 | 能开发并部署MCP Skill | ⭐ **能解释幂等性=唯一请求ID** |
| Week 8 | 有完整项目可展示 | ⭐ **能说清Coze vs Dify选型** |

---

*计划制定：基于 AI工程师 职位要求*
*最后更新：2026-04-01*
