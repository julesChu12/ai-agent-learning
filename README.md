# AI Agent 学习之路

> 从 Go 后端到 AI Agent 开发者的转型计划

---

## 目标岗位

**大模型应用开发（Agent平台设计、开发、落地）**

- 大模型产品技术方案设计、研发支持、效果优化
- 大模型应用架构稳定性、性能、可观测性建设

---

## 学习进度

### 12周学习计划进度

| 阶段 | 周次 | 主题 | 状态 |
|------|------|------|------|
| 基础夯实 | Week 1 | AI基础 + LLM机制 | ✅ 已完成 |
| 基础夯实 | Week 2 | RAG核心 + 全链路 | ⬜ 待开始 |
| 基础夯实 | Week 3 | Embedding + 向量数据库 | ⬜ 待开始 |
| 基础夯实 | Week 4 | Prompt工程 + Agent基础 | ⬜ 待开始 |
| 开发能力 | Week 5 | LangChain 基础 | ⬜ 待开始 |
| 开发能力 | Week 6 | LangChain 进阶 + LCEL | ⬜ 待开始 |
| 开发能力 | Week 7 | LangGraph + Agent循环 | ⬜ 待开始 |
| 开发能力 | Week 8 | Function Calling + Tool开发 | ⬜ 待开始 |
| 平台协议 | Week 9 | MCP协议 + Skill开发 | ⬜ 待开始 |
| 平台协议 | Week 10 | Coze/Dify平台实战 | ⬜ 待开始 |
| 项目整合 | Week 11 | 系统设计 + 架构入门 | ⬜ 待开始 |
| 项目整合 | Week 12 | 完整项目实战 | ⬜ 待开始 |

### 当前：Week 1 ✅ 已完成

**Week 1 学习内容：**
- Token & Tokenizer 原理（BPE/WordPiece）
- LLM 工作流程（概率预测本质）
- Context Window、Temperature 机制
- Streaming 流式输出实现（Python + Go）
- 主流模型选型（GPT-4o/GPT-4o-mini/Claude/Kimi/DeepSeek）

**主流模型对比（2025年）：**

| 模型 | Context Window | 输入价格 | 输出价格 | 特点 |
|------|---------------|---------|---------|------|
| GPT-4o-mini | 128K | $0.15/1M | $0.60/1M | 性价比之王 |
| GPT-4o | 128K | $2.50/1M | $10/1M | 通用强 |
| Claude 3.5 Sonnet | 200K | $3/1M | $15/1M | 长上下文 |
| Kimi | 128K/1M | ¥0.5/1K | ¥2/1K | 超长上下文 |
| DeepSeek-V3 | 128K | ¥1/1M | ¥2/1M | 开源高性价比 |

**检验清单：**
- [x] 能口算：1000字中文 ≈ 1333 Token
- [x] 能回答：GPT-4o 的 Context Window（128K）
- [x] 能解释：Temperature=0.7 vs 1.0 的区别
- [x] 能实现：带 Streaming 的 LLM 调用
- [x] 能根据任务选择合适的模型

---

## 技术栈

**核心技能：** Python、Agent、RAG、Prompt Engineering、LangChain/LangGraph、Dify/Coze/Langflow

**背景优势：** Go 后端开发经验

**重点提升：** Agent 工程落地能力、RAG、工作流编排、评测优化、系统设计

---

## 项目结构

```
ai-agent-learning/
├── README.md                              # 本文件
├── .claude.md                             # 项目规则
├── AI工程师_夯实基础式学习计划_12周.md     # 详细学习计划
├── AI工程师_Agent_MCP学习计划_8周.md       # 快速学习路径
├── requirements.txt                       # Python 依赖
├── learning-progress/
│   ├── README.md                          # 进度追踪总览
│   ├── week01-学习笔记.md                 # Week 1 详细笔记
│   └── week01-总结模板.md                # 周总结模板
└── api.py, time.pulm                     # 学习过程中的代码练习
```

---

## 面试目标

针对 AI Agent 相关岗位的面试准备：

| 能力维度 | 具体要求 |
|---------|---------|
| 应用开发 | Python、API、工作流、工具调用、RAG、Agent 框架 |
| 工程落地 | 模块拆分、服务边界、可测试性、稳定性、性能优化 |
| 平台思维 | 通用组件、Prompt 管理、评测体系、配置化、可复用 |
| 产品意识 | 场景定义、指标设计、用户价值、迭代优先级 |
| 架构能力 | 检索、存储、缓存、队列、流式处理、监控告警、容灾 |

---

## 联系方式

- GitHub: [julesChu12/ai-agent-learning](https://github.com/julesChu12/ai-agent-learning)

---

*学习开始日期：2026-04-01*
*最后更新：2026-04-13*
