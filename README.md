> 本项目由本人的另外一个项目：[DesignTree](https://github.com/new-tonAA/DesignTree) 改编而来

<div align="center">

# 🌳 OfferTree — AI求职智能匹配

**基于 OpenAI Agents SDK 的智能求职匹配与简历优化系统**

融合岗位匹配、简历评估优化、决策树管理的AI求职辅助平台

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![OpenAI Agents SDK](https://img.shields.io/badge/OpenAI_Agents_SDK-0.17+-orange.svg)](https://openai-agents-sdk.doczh.com/streaming/)

</div>

---

## 一、背景

学生在求职场景时会面临以下痛点：

1. 在海量岗位中，搜寻与自己背景、能力专长、职业兴趣匹配度高的工作机会会花费大量的时间。
2. 明确感兴趣的岗位后，不确定自己的简历与岗位的匹配度，同时也想知道简历需要做哪些优化，能提升通过简历初筛的命中率。

## 二、任务

结合学生的真实求职场景，设计一个AI求职智能匹配智能体，帮助学生有效匹配合适岗位，并提升对心仪岗位的初筛命中率。

## 三、实验环境

| 项目 | 说明 |
|------|------|
| **OpenAI Agents SDK 流式传输 API** | [https://openai-agents-sdk.doczh.com/streaming/](https://openai-agents-sdk.doczh.com/streaming/) |
| **DeepSeek API** | 创建 API-Key：[https://platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys) |
| **Python** | ≥ 3.10（推荐 3.11） |
| **Conda 环境名** | `offertree` |
| **安装依赖** | `pip install -r requirements.txt`（含 `openai-agents>=0.17.2`） |
| **后端框架** | FastAPI + Uvicorn |
| **前端** | 原生 HTML/CSS/JS（无框架依赖） |

## 四、核心功能

### 4.1 🎯 智能岗位匹配

根据用户的专业、爱好、技能等画像信息，AI推荐最匹配的岗位方向和具体职位：
- 匹配度分析（技能、经验、教育背景）
- 技能差距分析
- 求职策略建议

### 4.2 📝 简历评估与优化

- **简历评估**：多维度评估简历与目标岗位的匹配度（整体评分、技能匹配、优势亮点、不足之处）
- **简历优化**：AI自动优化简历，提升ATS系统通过率（关键词优化、量化成果、针对性调整）

### 4.3 🌳 决策树管理

```
root (初始求职需求)
 └── v1 (第一次匹配 + 回答)
      ├── v2 (继续追问，细化方向)  ← 当前节点
      └── v1b (回退后探索另一方向)
```

- **路径记忆**：自动收集从叶子节点到根节点的所有问答，组成上下文
- **分支回退**：点击任意历史节点即可回退，支持分支探索
- **偏好学习**：自动学习用户的求职偏好（行业方向、技能方向等）
- **状态持久化**：所有节点、对话、选中状态自动保存至 JSON 文件

### 4.4 💬 流式问答 + Markdown 渲染

- 流式输出，逐字呈现，打字机效果
- 完整 Markdown 渲染：标题、列表、代码高亮、表格、引用

### 4.5 🔄 多平台支持

| 平台 | 文字模型 | 图片模型 |
|------|----------|----------|
| DeepSeek | deepseek-chat | - |
| V3.CM | gpt-4o-mini | gpt-image-1, gpt-image-2 |
| OpenAI | gpt-4o | gpt-image-1, gpt-image-2 |
| OpenRouter | Claude, Gemini | Gemini Image |
| 火山引擎 | Doubao-Seed 系列 | doubao-seedream 系列 |

## 五、快速开始

### 方式一：Conda 环境（推荐）

```bash
# 1. 创建 conda 环境
conda create -n offertree python=3.11 -y

# 2. 激活环境
conda activate offertree

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务器
python server.py

# 5. 打开浏览器
# http://localhost:8000
```

### 方式二：pip 直接安装

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务器
python server.py

# 3. 打开浏览器
# http://localhost:8000
```

### API Key 配置

**方式一：环境变量**
```bash
export OPENAI_API_KEY=sk-...
python server.py
```

**方式二：UI 设置**

打开 http://localhost:8000，点击左上角logo，输入各平台的 API Key。

> 💡 命令行Agent `job_match_agent.py` 会自动读取 OfferTree 的 `config.json` 中保存的 Key。

### 命令行Agent运行方式

```bash
# 激活 conda 环境
conda activate offertree

# 设置 API Key（支持 DeepSeek / OpenAI）
export OPENAI_API_KEY=sk-...

# 运行命令行Agent
python job_match_agent.py
```

运行示例：
```
AI求职智能匹配 Agent 已启动。输入问题后回车，输入 N/Exit/退出 结束。

求职问题 > 我是计算机专业大三学生，想找互联网公司前端开发实习
根据你的背景，推荐以下岗位方向：
1. 前端开发实习生（匹配度85%）- 互联网公司...
2. 全栈开发实习生（匹配度70%）...

求职问题 > N
已退出AI求职智能匹配。
```

## 六、文件结构

```
OfferTree/
├── job_match_agent.py      # 命令行求职匹配 Agent
├── state_manager.py         # 决策树核心逻辑（节点管理、路径记忆、偏好学习）
├── agent.py                 # AI 调用层（prompt优化 / 图片生成 / 语音转文字）
├── server.py                # FastAPI 后端（API 路由、简历评估、岗位匹配）
├── state_tree.py            # 决策树数据结构（Pydantic 模型）
├── requirements.txt         # Python 依赖
├── config.json              # API Keys 配置（自动生成）
├── sessions/                # 会话 JSON 文件（自动保存）
├── static/
│   ├── index.html           # 前端界面
│   └── uploads/             # 生成的图片
└── Readme_image/            # README 截图
```

## 七、API 接口

| 功能 | 接口 | 说明 |
|------|------|------|
| 流式问答 | `POST /api/generate` | 求职问题流式回答 |
| 岗位匹配 | `POST /api/job_match` | 根据用户画像推荐岗位 |
| 简历评估 | `POST /api/evaluate_resume` | 评估简历匹配度 |
| 简历优化 | `POST /api/enhance_resume` | 优化简历提升通过率 |
| 图片生成 | `POST /api/generate_image` | 生成示意图（可选） |
| 项目管理 | `POST /api/project/new` | 新建求职项目 |

---

<div align="center">

**OfferTree** — 让求职决策像树一样清晰 🌳

</div>
