"""
server.py — FastAPI 后端

启动：
  cd arch_ai
  python server.py

浏览器访问 http://localhost:8000
"""

import os
import json
import tempfile
import time
import base64
import re
from pathlib import Path

import uvicorn
import httpx
import contextvars
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

import state_manager as sm
import agent
import guided_tree as gt

app = FastAPI(title="OfferTree - AI求职智能匹配")

# 获取exe所在目录（兼容开发环境和打包后）
def get_app_dir() -> Path:
    """获取应用程序所在目录"""
    import sys
    if getattr(sys, 'frozen', False):
        # PyInstaller打包后
        return Path(sys.executable).parent
    else:
        # 开发环境
        return Path(__file__).parent

APP_DIR = get_app_dir()
STATIC_DIR = APP_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "uploads").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

CONFIG_FILE = APP_DIR / "config.json"

# 多用户隔离：每个客户端独立会话
_sessions_by_client: dict[str, dict] = {}
_last_session_paths: dict[str, str] = {}
_client_id_var = contextvars.ContextVar('client_id', default='default')


def _get_session() -> dict:
    """获取当前客户端的会话"""
    cid = _client_id_var.get()
    return _sessions_by_client.get(cid, {})

def _set_session(session: dict) -> None:
    """设置当前客户端的会话"""
    cid = _client_id_var.get()
    _sessions_by_client[cid] = session

def _get_last_path() -> str:
    """获取当前客户端的上次会话路径"""
    cid = _client_id_var.get()
    return _last_session_paths.get(cid, "")

def _set_last_path(path: str) -> None:
    """设置当前客户端的上次会话路径"""
    cid = _client_id_var.get()
    _last_session_paths[cid] = path


@app.middleware("http")
async def client_id_middleware(request: Request, call_next):
    """多用户隔离中间件：从请求头提取客户端ID，实现会话隔离"""
    client_id = request.headers.get("X-Client-ID", "default")
    _client_id_var.set(client_id)
    response = await call_next(request)
    return response


def load_config():
    """加载配置文件"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}


def save_config(config: dict):
    """保存配置文件"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_api_keys_from_config():
    """从配置文件加载API keys"""
    config = load_config()
    keys = config.get('api_keys', {})
    openai_key = None
    if keys.get('openai'):
        openai_key = agent._sanitize_api_key(keys['openai'])
    elif config.get('openai_api_key'):
        openai_key = agent._sanitize_api_key(config.get('openai_api_key'))

    if openai_key:
        agent.PLATFORMS["openai"]["api_key"] = openai_key
        os.environ["OPENAI_API_KEY"] = openai_key

    if keys.get('openrouter'):
        agent.PLATFORMS["openrouter"]["api_key"] = agent._sanitize_api_key(keys['openrouter'])
    if keys.get('v3'):
        agent.PLATFORMS["v3"]["api_key"] = agent._sanitize_api_key(keys['v3'])
    if keys.get('deepseek'):
        agent.PLATFORMS["deepseek"]["api_key"] = agent._sanitize_api_key(keys['deepseek'])
    if keys.get('volcengine'):
        agent.PLATFORMS["volcengine"]["api_key"] = agent._sanitize_api_key(keys['volcengine'])
    if openai_key or keys:
        print(f"[info] Loaded API keys from config: {CONFIG_FILE}")


# 启动时加载配置
load_api_keys_from_config()


def _auto_load_latest_session():
    """启动时自动加载最近的会话（为默认客户端）"""
    _client_id_var.set("default")
    sessions = sm.list_sessions()
    if sessions:
        latest = sessions[0]["path"]
        try:
            session = sm.load_session(latest)
            _set_session(session)
            _set_last_path(latest)
            print(f"[info] Auto-loaded session: {latest}")
        except Exception as e:
            print(f"[warn] Failed to auto-load session: {e}")


# 启动时自动加载
_auto_load_latest_session()


def _require_session():
    session = _get_session()
    if not session:
        # 尝试自动恢复
        last_path = _get_last_path()
        if last_path and Path(last_path).exists():
            try:
                session = sm.load_session(last_path)
                _set_session(session)
                return session
            except:
                pass
        raise HTTPException(400, "请先创建或加载一个项目")
    return session


# ── 页面 ─────────────────────────────────────

def get_resource_path(relative_path: str) -> Path:
    """获取资源文件路径（兼容开发环境和打包后）"""
    import sys
    if getattr(sys, 'frozen', False):
        # PyInstaller打包后，资源在临时目录
        base_path = Path(sys._MEIPASS)
    else:
        # 开发环境
        base_path = Path(__file__).parent
    return base_path / relative_path

@app.get("/", response_class=HTMLResponse)
def index():
    return get_resource_path("static/index.html").read_text(encoding="utf-8")


# ── 项目管理 ─────────────────────────────────

class NewProjectReq(BaseModel):
    project_name: str

@app.post("/api/project/new")
def new_project(req: NewProjectReq):
    client_id = _client_id_var.get()
    session = sm.new_session(req.project_name, client_id=client_id)
    _set_session(session)
    _set_last_path(session["save_path"])
    return {"ok": True, "save_path": session["save_path"]}

@app.post("/api/project/load")
def load_project(path: str = Form(...)):
    try:
        session = sm.load_session(path)
        # 权限检查：只能加载自己的项目或旧的无归属项目
        client_id = _client_id_var.get()
        session_cid = session.get("client_id", "default")
        if session_cid != client_id and session_cid != "default":
            raise HTTPException(403, "无权访问该项目")
        # 认领旧项目：将default项目绑定到当前客户端
        if session_cid == "default" and client_id != "default":
            session["client_id"] = client_id
            sm._save(session)
        _set_session(session)
        _set_last_path(path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "project": session["project"]}

@app.get("/api/project/list")
def list_projects():
    client_id = _client_id_var.get()
    return sm.list_sessions(client_id=client_id)

class RenameProjectReq(BaseModel):
    path: str
    new_name: str

@app.post("/api/project/rename")
def rename_project(req: RenameProjectReq):
    """重命名项目"""
    try:
        s = sm.load_session(req.path)
        # 权限检查：只能操作自己的项目（旧的无归属项目允许操作）
        client_id = _client_id_var.get()
        session_cid = s.get("client_id", "default")
        if session_cid != client_id and session_cid != "default":
            raise HTTPException(403, "无权操作该项目")
        s["project"] = req.new_name
        sm._save(s)
        # 如果是当前会话，更新内存
        session = _get_session()
        if session and session.get("save_path") == req.path:
            session["project"] = req.new_name
            _set_session(session)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))

class DeleteProjectReq(BaseModel):
    path: str

@app.post("/api/project/delete")
def delete_project(req: DeleteProjectReq):
    """删除项目"""
    try:
        import os
        s = sm.load_session(req.path)
        # 权限检查：只能操作自己的项目（旧的无归属项目允许操作）
        client_id = _client_id_var.get()
        session_cid = s.get("client_id", "default")
        if session_cid != client_id and session_cid != "default":
            raise HTTPException(403, "无权操作该项目")
        # 如果删除的是当前会话，清空内存
        session = _get_session()
        if session and session.get("save_path") == req.path:
            _set_session({})
            _set_last_path("")
        os.remove(req.path)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/project/state")
def get_state():
    s = _require_session()
    cur = s["nodes"][s["current_node"]]
    current_selected_urls = sm.get_node_selected_urls(s)
    current_attachments = sm.get_node_attachments(s)
    current_path_node_ids = [n["id"] for n in sm.get_path_to_root(s)]
    # 收集路径上的问答对（用于文字模式恢复）
    path_qa = []
    for n in sm.get_path_to_root(s):
        if n["id"] != "root" and n.get("answer"):
            path_qa.append({"question": n["user_input"], "answer": n["answer"]})
    # 检测是否在引导模式：树存在且有question（根或子节点），或者已有guided_answers
    guided_tree = s.get("guided_tree", {})
    has_guided_content = (
        guided_tree.get("question")
        or (guided_tree.get("children") and len(guided_tree.get("children", {})) > 0)
        or (s.get("guided_answers") and len(s.get("guided_answers", [])) > 0)
    )
    guided_active = bool(guided_tree and has_guided_content)

    return {
        "project":       s["project"],
        "current_node":  s["current_node"],
        "current_path_node_ids": current_path_node_ids,
        "tree":          sm.get_tree_for_ui(s),
        "style_summary": sm.get_style_summary(s),
        "style_candidates": sm.get_style_candidates(s),
        "ref_images":    s["reference_images"],
        "save_path":     s["save_path"],
        "history":       _history_for_ui(s),
        "path_tags":     _path_tags(s),
        "current_images": cur["images"],
        "current_prompt": cur["prompt"],
        "current_answer": cur.get("answer", ""),
        "current_answer_selected": cur.get("answer_selected", False),
        "current_selected": cur.get("selected"),  # legacy
        "current_selecteds": current_selected_urls,
        "current_attached_images": current_attachments,  # legacy alias
        "current_attachments": current_attachments,
        "generating":    cur.get("generating", False) and not (bool(cur.get("images")) or bool(cur.get("answer"))),
        "path_qa":              path_qa,
        "guided_active":        guided_active,
        "guided_is_result":     s.get("guided_is_result", False),
        "guided_match_result":  s.get("guided_match_result", ""),
        "guided_exited":        s.get("guided_exited", False),
        "guided_free_chat_qa":  s.get("guided_free_chat_qa", []),
    }


# ── 核心：生成图片 ───────────────────────────

class GenerateReq(BaseModel):
    user_input: str
    history: Optional[list[dict]] = None
    model_memory: Optional[str] = None
    n: Optional[int] = 1
    parent_node_id: Optional[str] = None
    optimize_prompt: bool = True
    prompt_images: Optional[list[dict]] = None  # 附带图片

class PolishReq(BaseModel):
    user_input: str

@app.post("/api/polish_prompt")
def polish_prompt(req: PolishReq):
    """润色提示词（不生成图片）"""
    s = _require_session()
    # 校验 API Key
    text_api_key = agent._sanitize_api_key(agent._get_text_api_key())
    text_platform_name = agent.PLATFORMS[agent._current_text_platform]["name"]
    if not text_api_key:
        raise HTTPException(400, f"未设置 {text_platform_name} 的 API Key，请在设置页面配置")
    try:
        context = sm.build_context_for_agent(s, req.user_input, None)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        polished = agent.prompt_agent(context)
        return {"polished_prompt": polished, "optimized": True, "warning": None}
    except Exception as e:
        polished = agent.compose_prompt_from_context(context)
        return {
            "polished_prompt": polished,
            "optimized": False,
            "warning": f"AI润色失败，已切换为路径记忆拼接：{e}",
        }

@app.post("/api/generate")
async def generate(req: GenerateReq):
    """文本问答（流式），使用配置的文本平台，同时保存到树节点"""
    s = _require_session()

    # 先创建节点
    node = sm.add_node(s, req.user_input, req.parent_node_id)
    node_id = node["id"]

    input_items = []
    if req.history:
        # 只保留最近10条历史（5轮QA），避免token过多导致慢
        recent_history = req.history[-10:] if len(req.history) > 10 else req.history
        for item in recent_history:
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "").strip()[:2000]  # 限制单条长度
            if role in {"user", "assistant"} and content:
                input_items.append({"role": role, "content": content})

    input_items.append({"role": "user", "content": req.user_input})

    instructions = (
        "你是一名AI求职智能匹配助手，擅长帮助学生匹配合适的岗位、评估简历与岗位的匹配度、优化简历提升通过初筛的命中率。"
        "你可以根据用户的专业、爱好、技能等个人信息，推荐合适的职业方向和具体岗位。"
        "你还可以分析用户简历与目标岗位的匹配度，给出具体的简历优化建议。"
        "回答要结构清晰、逻辑完整、表述准确，尽量用中文回答。"
        "如果用户提出具体问题，直接给出完整答案，不要只给片段。"
        "请保持上下文连贯，并在回答中使用分点或示例来提升可读性。"
    )

    api_key = agent._sanitize_api_key(agent._get_text_api_key())
    text_platform = agent._current_text_platform
    config = agent.PLATFORMS[text_platform]
    base_url = config["base_url"]

    # 校验 API Key
    if not api_key:
        sm.remove_node(s, node_id)
        raise HTTPException(400, f"未设置 {config['name']} 的 API Key，请在设置页面配置")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if text_platform == "openrouter":
        headers["HTTP-Referer"] = "https://localhost:8000"
        headers["X-Title"] = "OfferTree"

    endpoint = "/v1/chat/completions"
    if text_platform == "volcengine":
        endpoint = "/chat/completions"

    payload = {
        "model": agent._current_text_model,
        "messages": [{"role": "system", "content": instructions}] + input_items,
        "max_tokens": 2048,
        "temperature": 0.3,
        "stream": True,
    }

    # 用闭包捕获 session 引用
    _session_ref = s

    async def event_stream():
        full_text = ""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", f"{base_url}{endpoint}", headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        err_msg = f"\n[错误] API 返回 {resp.status_code}: {error_body.decode('utf-8', errors='replace')[:300]}"
                        sm.set_node_answer(_session_ref, node_id, err_msg)
                        yield err_msg
                        return
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content
                                yield content
                        except Exception:
                            pass
        except Exception as e:
            err_msg = f"\n[错误] {e}"
            sm.set_node_answer(_session_ref, node_id, err_msg)
            yield err_msg
            return
        # 流式完成，保存回答到节点
        sm.set_node_answer(_session_ref, node_id, full_text)

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")


# ── 画图（可选功能） ────────────────────────────

class GenerateImageReq(BaseModel):
    user_input: str
    n: int = 1

@app.post("/api/generate_image")
async def generate_image(req: GenerateImageReq):
    """生成示意图，同时保存到树节点"""
    s = _require_session()

    # 校验 API Key
    image_api_key = agent._sanitize_api_key(agent._get_api_key())
    image_platform_name = agent.PLATFORMS[agent._current_platform]["name"]
    if not image_api_key:
        raise HTTPException(400, f"未设置 {image_platform_name} 的 API Key，请在设置页面配置")

    # 创建节点
    node = sm.add_node(s, req.user_input)
    node_id = node["id"]

    try:
        images = agent.generate_images(
            prompt=req.user_input + ", network diagram, educational illustration, clean and clear",
            n=req.n,
            size="1792x1024",
        )
        sm.add_images(s, node_id, images)
        sm.set_node_prompt(s, node_id, req.user_input)
        return {"images": images, "node_id": node_id}
    except Exception as e:
        # 生成失败，移除节点
        sm.remove_node(s, node_id)
        raise HTTPException(500, f"画图失败: {e}")


# ── 选图 ─────────────────────────────────────

class SelectImageReq(BaseModel):
    image_url: str

@app.post("/api/select_image")
def select_image(req: SelectImageReq):
    s = _require_session()
    try:
        sm.select_image(s, req.image_url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "style_summary": sm.get_style_summary(s)}


class StyleSelectReq(BaseModel):
    candidate_keywords: list[str]
    selected_keywords: list[str]


@app.post("/api/style/select")
def style_select(req: StyleSelectReq):
    s = _require_session()
    sm.apply_style_selection(s, req.candidate_keywords, req.selected_keywords)
    return {
        "ok": True,
        "style_summary": sm.get_style_summary(s),
        "style_candidates": sm.get_style_candidates(s),
    }


# ── 切换节点 ─────────────────────────────────

class SwitchNodeReq(BaseModel):
    node_id: str

@app.post("/api/switch_node")
def switch_node(req: SwitchNodeReq):
    s = _require_session()
    try:
        sm.switch_node(s, req.node_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "current_node": s["current_node"]}


class ToggleAnswerSelectedReq(BaseModel):
    node_id: Optional[str] = None


# ── 引导式求职匹配 ─────────────────────────────

def _get_guided_tree(s: dict) -> dict:
    """获取引导树，不存在则初始化"""
    if "guided_tree" not in s:
        s["guided_tree"] = {
            "id": "root",
            "question": "",
            "options": [],
            "selected_option": "",
            "selected_options": [],
            "multi_select": False,
            "max_select": 1,
            "children": {},
        }
    return s["guided_tree"]


def _find_node_by_id(tree: dict, node_id: str) -> dict | None:
    """在树中递归查找指定ID的节点"""
    if not tree:
        return None
    if tree.get("id") == node_id:
        return tree
    for child in tree.get("children", {}).values():
        found = _find_node_by_id(child, node_id)
        if found:
            return found
    return None


def _rebuild_guided_messages(s: dict):
    """根据当前树的选中路径，重建AI对话历史"""
    tree = _get_guided_tree(s)
    messages = [{"role": "system", "content": gt.SYSTEM_PROMPT}] + gt.get_path_messages(tree)
    s["guided_messages"] = messages


class GuidedStartReq(BaseModel):
    project_name: str = "求职匹配"

@app.post("/api/guided/start")
async def guided_start(req: GuidedStartReq):
    """开始引导式求职匹配，AI生成第一个问题"""

    # 创建新项目
    client_id = _client_id_var.get()
    session = sm.new_session(req.project_name, client_id=client_id)
    _set_session(session)
    _set_last_path(session["save_path"])

    # 初始化引导树和对话历史
    session["guided_answers"] = []
    session["guided_exited"] = False
    session["guided_free_chat_qa"] = []
    session["guided_messages"] = [
        {"role": "system", "content": gt.SYSTEM_PROMPT},
    ]
    session["guided_tree"] = {
        "id": "root",
        "question": "",
        "options": [],
        "selected_option": "",
        "selected_options": [],
        "multi_select": False,
        "max_select": 1,
        "children": {},
    }

    # 调用AI生成第一个问题
    return await _ai_generate_next_question(session)


class GuidedRespondReq(BaseModel):
    current_node_id: Optional[str] = None
    user_answer: str       # 用户选择的选项或输入的文本
    multi_answers: Optional[list[str]] = None
    project_name: Optional[str] = None

@app.post("/api/guided/respond")
async def guided_respond(req: GuidedRespondReq):
    """用户回答后，将回答添加到树节点，AI动态生成下一个问题或匹配结果"""
    s = _require_session()

    tree = _get_guided_tree(s)

    # 找到当前叶子节点（用户正在回答的节点）
    leaf = tree
    while leaf.get("selected_option") and leaf.get("selected_option") in leaf.get("children", {}):
        leaf = leaf["children"][leaf["selected_option"]]

    # 如果用户提供了node_id，优先使用它定位
    if req.current_node_id:
        found = _find_node_by_id(tree, req.current_node_id)
        if found:
            leaf = found

    # 记录用户选择到树的当前叶子节点
    answers_to_record = req.multi_answers or [req.user_answer]
    user_text = "、".join([a for a in answers_to_record if a])

    leaf["selected_option"] = user_text
    leaf["selected_options"] = [a for a in answers_to_record if a]

    # 如果该选择还没有子节点，创建一个占位子节点
    if user_text not in leaf.get("children", {}):
        if "children" not in leaf:
            leaf["children"] = {}
        leaf["children"][user_text] = {
            "id": f"node_{int(time.time()*1000)}",
            "question": "",
            "options": [],
            "selected_option": "",
            "selected_options": [],
            "multi_select": False,
            "max_select": 1,
            "children": {},
        }

    # 重建对话历史（基于树的选中路径）
    _rebuild_guided_messages(s)

    # 更新guided_answers（基于路径）
    s["guided_answers"] = gt.get_path_answers(tree)

    # 用户继续回答时清除旧的匹配结果标记
    s["guided_is_result"] = False
    s["guided_exited"] = False
    s.pop("guided_match_result", None)
    s["guided_free_chat_qa"] = []

    # 同时在状态树中添加节点
    node = sm.add_node(s, user_text)
    sm._save(s)

    # 调用AI生成下一个问题
    return await _ai_generate_next_question(s)


class GuidedBranchReq(BaseModel):
    """用户点击树中的节点，选择一个不同选项来分支"""
    node_id: str
    new_option: str  # 用户新选择的选项

@app.post("/api/guided/branch")
async def guided_branch(req: GuidedBranchReq):
    """用户在树中切换到不同分支，AI生成该分支的下一个问题"""
    s = _require_session()

    tree = _get_guided_tree(s)

    # 找到目标节点
    target = _find_node_by_id(tree, req.node_id)
    if not target:
        raise HTTPException(400, f"节点 {req.node_id} 不存在")

    # 更新选择（清除旧的子树选中路径）
    target["selected_option"] = req.new_option
    target["selected_options"] = [req.new_option]

    # 切换分支时清除匹配结果标记
    s["guided_is_result"] = False
    s["guided_exited"] = False
    s.pop("guided_match_result", None)
    s["guided_free_chat_qa"] = []

    # 如果该选择还没有子节点，创建占位
    if req.new_option not in target.get("children", {}):
        if "children" not in target:
            target["children"] = {}
        target["children"][req.new_option] = {
            "id": f"node_{int(time.time()*1000)}",
            "question": "",
            "options": [],
            "selected_option": "",
            "selected_options": [],
            "multi_select": False,
            "max_select": 1,
            "children": {},
        }

    # 重建对话历史
    _rebuild_guided_messages(s)

    # 更新guided_answers
    s["guided_answers"] = gt.get_path_answers(tree)

    # 如果新分支的子节点已有问题（之前问过），直接返回
    child = target["children"][req.new_option]
    if child.get("question"):
        sm._save(s)
        # 如果子节点已经有selected_option，继续沿着路径走
        cur = child
        while cur.get("selected_option") and cur["selected_option"] in cur.get("children", {}):
            cur = cur["children"][cur["selected_option"]]

        if cur.get("question"):
            return {
                "ok": True,
                "node_id": cur["id"],
                "question": cur["question"],
                "options": cur.get("options", []),
                "input_type": "select_or_text",
                "multi_select": cur.get("multi_select", False),
                "max_select": cur.get("max_select", 1),
                "is_result": False,
                "ai_generate": True,
                "tree": gt.build_tree_for_ui(tree, s.get("guided_free_chat_qa")),
            }



    # 新分支没有问题，调用AI生成
    return await _ai_generate_next_question(s)


class GuidedNavigateReq(BaseModel):
    """用户点击树中的中间节点，回退到该节点重新选择"""
    node_id: str


@app.post("/api/guided/navigate")
async def guided_navigate(req: GuidedNavigateReq):
    """用户点击树中的中间节点，回退到该节点重新选择"""
    s = _require_session()

    tree = _get_guided_tree(s)

    # 找到目标节点
    target = _find_node_by_id(tree, req.node_id)
    if not target:
        raise HTTPException(400, f"节点 {req.node_id} 不存在")

    # 清除目标节点的selected_option，使其成为当前叶子
    target["selected_option"] = ""
    target["selected_options"] = []

    # 回溯时清除匹配结果标记
    s["guided_is_result"] = False
    s["guided_exited"] = False
    s.pop("guided_match_result", None)

    # 重建对话历史和答案
    _rebuild_guided_messages(s)
    s["guided_answers"] = gt.get_path_answers(tree)
    sm._save(s)

    # 返回该节点的问题和选项
    return {
        "ok": True,
        "node_id": target["id"],
        "question": target.get("question", ""),
        "options": target.get("options", []),
        "input_type": "select_or_text",
        "multi_select": target.get("multi_select", False),
        "max_select": target.get("max_select", 1),
        "is_result": False,
        "ai_generate": True,
        "tree": gt.build_tree_for_ui(tree, s.get("guided_free_chat_qa")),
    }





async def _ai_generate_next_question(s: dict):
    """调用AI根据对话历史动态生成下一个问题或触发匹配结果"""
    api_key = agent._sanitize_api_key(agent._get_text_api_key())
    text_platform = agent._current_text_platform
    config = agent.PLATFORMS[text_platform]
    base_url = config["base_url"]

    if not api_key:
        raise HTTPException(400, f"未设置 {config['name']} 的 API Key，请在设置页面配置")

    messages = s.get("guided_messages", [])

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if text_platform == "openrouter":
        headers["HTTP-Referer"] = "https://localhost:8000"
        headers["X-Title"] = "OfferTree"

    endpoint = "/v1/chat/completions"
    if text_platform == "volcengine":
        endpoint = "/chat/completions"

    payload = {
        "model": agent._current_text_model,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.5,
        "stream": False,
    }

    # 默认值（AI调用失败时使用）
    question = "你目前感兴趣或学习的领域是什么？"
    options = ["计算机/IT", "金融/经济", "法律", "教育", "医疗/生物", "其他"]
    action = "ask"
    multi_select = False
    max_select = 1

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{base_url}{endpoint}", headers=headers, json=payload)
            if resp.status_code != 200:
                pass
            else:
                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                try:
                    clean_text = re.sub(r'```(?:json)?\s*', '', text).strip().rstrip('`').strip()
                    json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        question = parsed.get("question", question)
                        options = parsed.get("options", options)
                        action = parsed.get("action", "ask")
                        multi_select = parsed.get("multi_select", False)
                        max_select = parsed.get("max_select", 1)
                    else:
                        parsed = json.loads(clean_text)
                        question = parsed.get("question", question)
                        options = parsed.get("options", options)
                        action = parsed.get("action", "ask")
                        multi_select = parsed.get("multi_select", False)
                        max_select = parsed.get("max_select", 1)
                except (json.JSONDecodeError, AttributeError):
                    if text and len(text) < 200:
                        question = text
    except Exception:
        pass

    # 将AI生成的问题写入树的当前叶子节点
    tree = _get_guided_tree(s)
    leaf = tree
    while leaf.get("selected_option") and leaf["selected_option"] in leaf.get("children", {}):
        leaf = leaf["children"][leaf["selected_option"]]

    if not leaf.get("id"):
        leaf["id"] = f"node_{int(time.time()*1000)}"
    leaf["question"] = question
    leaf["options"] = options
    leaf["multi_select"] = multi_select
    leaf["max_select"] = max_select
    # selected_option 已经在 respond 时设置过了（如果是新问题则是空的）

    # 记录AI的回复到对话历史
    ai_response = json.dumps({"question": question, "options": options, "action": action}, ensure_ascii=False)
    s["guided_messages"].append({"role": "assistant", "content": ai_response})
    sm._save(s)

    # 构建前端树
    ui_tree = gt.build_tree_for_ui(tree, s.get("guided_free_chat_qa"))

    if action == "result":
        return await _generate_match_result(s, "match_result")

    return {
        "ok": True,
        "node_id": leaf["id"],
        "question": question,
        "options": options,
        "input_type": "select_or_text",
        "multi_select": multi_select,
        "max_select": max_select,
        "is_result": False,
        "ai_generate": True,
        "tree": ui_tree,
    }


async def _generate_match_result(s: dict, node_id: str):
    """调用AI根据用户画像生成匹配结果（流式）"""
    api_key = agent._sanitize_api_key(agent._get_text_api_key())
    text_platform = agent._current_text_platform
    config = agent.PLATFORMS[text_platform]
    base_url = config["base_url"]

    if not api_key:
        raise HTTPException(400, f"未设置 {config['name']} 的 API Key，请在设置页面配置")

    profile = gt.build_user_profile(s.get("guided_answers", []))

    match_prompt = f"""你是一名AI求职匹配专家。请根据以下用户画像，推荐最匹配的岗位方向和具体职位。

用户画像：
{profile}

请输出：
1. **推荐岗位方向**（按匹配度从高到低排列，至少3个方向）
   每个方向包含：
   - 岗位名称
   - 匹配度（0-100%）
   - 匹配原因（结合用户画像说明）
   - 适合的具体职位举例

2. **技能差距分析**
   - 用户已具备的核心技能
   - 需要补充的技能
   - 推荐的学习路径

3. **求职策略建议**
   - 简历重点突出的方向
   - 目标公司类型建议
   - 面试准备重点

请确保推荐具体、有针对性、可操作。"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if text_platform == "openrouter":
        headers["HTTP-Referer"] = "https://localhost:8000"
        headers["X-Title"] = "OfferTree"

    endpoint = "/v1/chat/completions"
    if text_platform == "volcengine":
        endpoint = "/chat/completions"

    payload = {
        "model": agent._current_text_model,
        "messages": [
            {"role": "system", "content": gt.MATCH_SYSTEM_PROMPT},
            {"role": "user", "content": match_prompt},
        ],
        "max_tokens": 2500,
        "temperature": 0.4,
        "stream": True,
    }

    _session_ref = s

    async def match_stream():
        full_text = ""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", f"{base_url}{endpoint}", headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        yield f"\n[错误] API 返回 {resp.status_code}: {error_body.decode('utf-8', errors='replace')[:300]}"
                        return
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content
                                yield content
                        except Exception:
                            pass
        except Exception as e:
            yield f"\n[错误] {e}"
            return
        # 流式完成，保存匹配结果到session
        _session_ref["guided_match_result"] = full_text
        _session_ref["guided_is_result"] = True
        sm._save(_session_ref)

    # 标记这是一个流式匹配结果
    # 使用特殊前缀让前端识别
    return StreamingResponse(match_stream(), media_type="text/plain; charset=utf-8",
                             headers={"X-Guided-Result": "true"})


class GuidedFinishReq(BaseModel):
    """用户在AI深入了解阶段自由输入后，请求生成匹配结果"""

@app.post("/api/guided/finish")
async def guided_finish():
    """结束引导问答，生成最终匹配结果（流式）"""
    s = _require_session()
    return await _generate_match_result(s, "match_result")


@app.get("/api/guided/tree")
def guided_tree():
    """获取当前引导树结构（供前端渲染侧边栏树）"""
    s = _require_session()
    tree = _get_guided_tree(s)
    return {
        "ok": True,
        "tree": gt.build_tree_for_ui(tree, s.get("guided_free_chat_qa")),
    }





@app.post("/api/guided/clear_result")
def guided_clear_result():
    """清除匹配结果标记（用户点击"继续追问"时调用）"""
    s = _require_session()
    s["guided_is_result"] = False
    s["guided_exited"] = True
    s.pop("guided_match_result", None)
    # 初始化自由对话QA列表
    if "guided_free_chat_qa" not in s:
        s["guided_free_chat_qa"] = []
    sm._save(s)
    return {"ok": True}


class GuidedFreeChatQAReq(BaseModel):
    question: str
    answer: str

@app.post("/api/guided/free_chat_qa")
def guided_add_free_chat_qa(req: GuidedFreeChatQAReq):
    """追加一条自由对话QA到后端session"""
    s = _require_session()
    if "guided_free_chat_qa" not in s:
        s["guided_free_chat_qa"] = []
    s["guided_free_chat_qa"].append({
        "question": req.question,
        "answer": req.answer,
    })
    sm._save(s)
    return {"ok": True}


# ── 简历评估 ─────────────────────────────────

class EvaluateResumeReq(BaseModel):
    resume_text: str
    target_position: str = ""

@app.post("/api/evaluate_resume")
async def evaluate_resume(req: EvaluateResumeReq):
    """评估简历与目标岗位的匹配度，返回评估结果和优化建议"""
    s = _require_session()

    api_key = agent._sanitize_api_key(agent._get_text_api_key())
    text_platform = agent._current_text_platform
    config = agent.PLATFORMS[text_platform]
    base_url = config["base_url"]

    if not api_key:
        raise HTTPException(400, f"未设置 {config['name']} 的 API Key，请在设置页面配置")

    target_part = f"目标岗位：{req.target_position}\n" if req.target_position else ""

    eval_prompt = f"""你是一名资深HR和简历评估专家。请对以下简历进行详细评估。

{target_part}简历内容：
{req.resume_text}

请从以下维度进行评估，并以结构化的方式输出：

1. **整体评分**（0-100分）
2. **匹配度分析**
   - 技能匹配度（0-100%）
   - 经验匹配度（0-100%）
   - 教育背景匹配度（0-100%）
3. **优势亮点**（列出3-5个）
4. **不足之处**（列出3-5个）
5. **具体优化建议**（每条建议要具体、可操作）
6. **关键词缺失**（该岗位常见但简历中缺失的关键词）

请确保评估客观、专业、有建设性。"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if text_platform == "openrouter":
        headers["HTTP-Referer"] = "https://localhost:8000"
        headers["X-Title"] = "OfferTree"

    endpoint = "/v1/chat/completions"
    if text_platform == "volcengine":
        endpoint = "/chat/completions"

    payload = {
        "model": agent._current_text_model,
        "messages": [
            {"role": "system", "content": "你是一名资深HR和简历评估专家，擅长评估简历与岗位的匹配度并给出专业优化建议。"},
            {"role": "user", "content": eval_prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
        "stream": True,
    }

    _session_ref = s

    async def eval_stream():
        full_text = ""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", f"{base_url}{endpoint}", headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        yield f"\n[错误] API 返回 {resp.status_code}: {error_body.decode('utf-8', errors='replace')[:300]}"
                        return
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content
                                yield content
                        except Exception:
                            pass
        except Exception as e:
            yield f"\n[错误] {e}"
            return

    return StreamingResponse(eval_stream(), media_type="text/plain; charset=utf-8")


class EnhanceResumeReq(BaseModel):
    resume_text: str
    target_position: str = ""
    weak_points: str = ""

@app.post("/api/enhance_resume")
async def enhance_resume(req: EnhanceResumeReq):
    """根据评估结果增强简历，返回优化后的简历文本"""
    s = _require_session()

    api_key = agent._sanitize_api_key(agent._get_text_api_key())
    text_platform = agent._current_text_platform
    config = agent.PLATFORMS[text_platform]
    base_url = config["base_url"]

    if not api_key:
        raise HTTPException(400, f"未设置 {config['name']} 的 API Key，请在设置页面配置")

    target_part = f"目标岗位：{req.target_position}\n" if req.target_position else ""
    weak_part = f"已知不足：{req.weak_points}\n" if req.weak_points else ""

    enhance_prompt = f"""你是一名资深简历优化专家。请根据以下信息优化简历，提升通过初筛的命中率。

{target_part}{weak_part}原始简历：
{req.resume_text}

请输出优化后的完整简历，要求：
1. 针对目标岗位优化关键词，提高ATS系统通过率
2. 量化成果描述（用数据说话）
3. 突出与目标岗位相关的经验和技能
4. 使用更有力的动词和描述
5. 保持真实，不编造经历
6. 输出完整的优化后简历（而不是只给修改建议）

请直接输出优化后的简历内容，使用Markdown格式。"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if text_platform == "openrouter":
        headers["HTTP-Referer"] = "https://localhost:8000"
        headers["X-Title"] = "OfferTree"

    endpoint = "/v1/chat/completions"
    if text_platform == "volcengine":
        endpoint = "/chat/completions"

    payload = {
        "model": agent._current_text_model,
        "messages": [
            {"role": "system", "content": "你是一名资深简历优化专家，擅长根据目标岗位优化简历，提高ATS系统通过率和面试机会。"},
            {"role": "user", "content": enhance_prompt},
        ],
        "max_tokens": 3000,
        "temperature": 0.4,
        "stream": True,
    }

    async def enhance_stream():
        full_text = ""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", f"{base_url}{endpoint}", headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        yield f"\n[错误] API 返回 {resp.status_code}: {error_body.decode('utf-8', errors='replace')[:300]}"
                        return
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content
                                yield content
                        except Exception:
                            pass
        except Exception as e:
            yield f"\n[错误] {e}"
            return

    return StreamingResponse(enhance_stream(), media_type="text/plain; charset=utf-8")


class JobMatchReq(BaseModel):
    user_profile: str  # 用户画像：专业、爱好、技能等

@app.post("/api/job_match")
async def job_match(req: JobMatchReq):
    """根据用户画像推荐匹配的岗位方向"""
    s = _require_session()

    api_key = agent._sanitize_api_key(agent._get_text_api_key())
    text_platform = agent._current_text_platform
    config = agent.PLATFORMS[text_platform]
    base_url = config["base_url"]

    if not api_key:
        raise HTTPException(400, f"未设置 {config['name']} 的 API Key，请在设置页面配置")

    match_prompt = f"""你是一名AI求职匹配专家。请根据以下用户画像，推荐最匹配的岗位方向和具体职位。

用户画像：
{req.user_profile}

请输出：
1. **推荐岗位方向**（按匹配度从高到低排列，至少3个方向）
   每个方向包含：
   - 岗位名称
   - 匹配度（0-100%）
   - 匹配原因（结合用户画像说明）
   - 适合的具体职位举例

2. **技能差距分析**
   - 用户已具备的核心技能
   - 需要补充的技能
   - 推荐的学习路径

3. **求职策略建议**
   - 简历重点突出的方向
   - 目标公司类型建议
   - 面试准备重点

请确保推荐具体、有针对性、可操作。"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if text_platform == "openrouter":
        headers["HTTP-Referer"] = "https://localhost:8000"
        headers["X-Title"] = "OfferTree"

    endpoint = "/v1/chat/completions"
    if text_platform == "volcengine":
        endpoint = "/chat/completions"

    payload = {
        "model": agent._current_text_model,
        "messages": [
            {"role": "system", "content": gt.MATCH_SYSTEM_PROMPT},
            {"role": "user", "content": match_prompt},
        ],
        "max_tokens": 2500,
        "temperature": 0.4,
        "stream": True,
    }

    async def match_stream():
        full_text = ""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", f"{base_url}{endpoint}", headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        yield f"\n[错误] API 返回 {resp.status_code}: {error_body.decode('utf-8', errors='replace')[:300]}"
                        return
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content
                                yield content
                        except Exception:
                            pass
        except Exception as e:
            yield f"\n[错误] {e}"
            return

    return StreamingResponse(match_stream(), media_type="text/plain; charset=utf-8")

@app.post("/api/toggle_answer_selected")
def toggle_answer_selected(req: ToggleAnswerSelectedReq):
    """切换当前节点的文字回答选中状态"""
    s = _require_session()
    node_id = req.node_id or s["current_node"]
    if node_id not in s["nodes"]:
        raise HTTPException(400, f"节点 {node_id} 不存在")
    if not s["nodes"][node_id].get("answer"):
        raise HTTPException(400, "该节点没有文字回答")
    new_state = sm.toggle_answer_selected(s, node_id)
    return {"ok": True, "answer_selected": new_state}


# ── 语音转文字 ───────────────────────────────

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    suffix = Path(audio.filename or "audio.webm").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        text = agent.transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)
    return {"text": text}


# ── 上传参考图 ───────────────────────────────

@app.post("/api/upload_reference")
async def upload_reference(file: UploadFile = File(...), label: str = Form("")):
    s = _require_session()
    dest = Path(__file__).parent / "static" / "uploads" / f"ref_{file.filename}"
    dest.write_bytes(await file.read())
    url = f"/static/uploads/ref_{file.filename}"
    sm.add_reference_image(s, url, label or file.filename)
    return {"ok": True, "url": url}


# ── 保存 API Key ─────────────────────────────

class SaveKeyReq(BaseModel):
    api_key: str

@app.post("/api/save_key")
def save_key(req: SaveKeyReq):
    config_path = Path(__file__).parent / "config.json"
    data = json.loads(config_path.read_text()) if config_path.exists() else {}
    data["openai_api_key"] = req.api_key
    config_path.write_text(json.dumps(data))
    os.environ["OPENAI_API_KEY"] = req.api_key
    return {"ok": True}


# ── 平台选择 ───────────────────────────────

class SetPlatformReq(BaseModel):
    platform: Optional[str] = None
    image_model: Optional[str] = None
    text_model: Optional[str] = None
    text_platform: Optional[str] = None  # 独立的文本平台

@app.get("/api/platforms")
def get_platforms():
    """获取所有支持的平台和模型列表"""
    return agent.get_platforms()

@app.post("/api/set_platform")
def set_platform(req: SetPlatformReq):
    """设置当前使用的平台和模型"""
    try:
        agent.set_platform(req.platform, req.image_model, req.text_model, req.text_platform)
        return {"ok": True, "config": agent.get_current_config()}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/api/current_platform")
def get_current_platform():
    """获取当前平台配置"""
    return agent.get_current_config()


class SetApiKeysReq(BaseModel):
    openai: Optional[str] = None
    openrouter: Optional[str] = None
    v3: Optional[str] = None
    deepseek: Optional[str] = None
    volcengine: Optional[str] = None

@app.post("/api/set_api_keys")
def set_api_keys(req: SetApiKeysReq):
    """设置API Keys，覆盖默认值并保存到配置文件"""
    keys = {}
    if req.openai:
        agent.PLATFORMS["openai"]["api_key"] = agent._sanitize_api_key(req.openai)
        keys["openai"] = req.openai
    if req.openrouter:
        agent.PLATFORMS["openrouter"]["api_key"] = agent._sanitize_api_key(req.openrouter)
        keys["openrouter"] = req.openrouter
    if req.v3:
        agent.PLATFORMS["v3"]["api_key"] = agent._sanitize_api_key(req.v3)
        keys["v3"] = req.v3
    if req.deepseek:
        agent.PLATFORMS["deepseek"]["api_key"] = agent._sanitize_api_key(req.deepseek)
        keys["deepseek"] = req.deepseek
    if req.volcengine:
        agent.PLATFORMS["volcengine"]["api_key"] = agent._sanitize_api_key(req.volcengine)
        keys["volcengine"] = req.volcengine
    
    # 保存到配置文件
    config = load_config()
    config['api_keys'] = keys
    save_config(config)
    print(f"[info] Saved API keys to config: {CONFIG_FILE}")
    
    return {"ok": True}


# ── 辅助格式化 ───────────────────────────────

def _history_for_ui(s: dict) -> list:
    out = []
    for n in s["nodes"].values():
        selected_images = []
        selected_urls = []
        if isinstance(n.get("selected_list"), list):
            selected_urls = [u for u in n.get("selected_list", []) if isinstance(u, str) and u]
        elif n.get("selected"):
            selected_urls = [n.get("selected")]

        if selected_urls:
            by_url = {
                img.get("url"): img
                for img in n.get("images", [])
                if isinstance(img, dict) and img.get("url")
            }
            selected_images = [by_url[u] for u in selected_urls if u in by_url]
        selected_image = selected_images[0] if selected_images else None

        # 获取节点的附带图片
        attachments = []
        if isinstance(n.get("attachments"), list):
            attachments = [img for img in n.get("attachments", []) if isinstance(img, dict) and img.get("url")]

        out.append({
            "id":         n["id"],
            "user_input": n["user_input"],
            "prompt":     n["prompt"],
            "answer":     n.get("answer", ""),
            "answer_selected": n.get("answer_selected", False),
            "selected":   bool(selected_urls),
            "selected_count": len(selected_images),
            "selected_urls": selected_urls,
            "selected_images": selected_images,
            "selected_image": selected_image,
            "images":     len(n["images"]),
            "image_list": [img for img in n.get("images", []) if isinstance(img, dict) and img.get("url")],
            "is_current": n["id"] == s["current_node"],
            "parent":     n["parent"],
            "attachments": attachments,
        })
    return out


def _save_prompt_attachments(prompt_images: list[dict]) -> list[dict]:
    out: list[dict] = []
    upload_dir = Path(__file__).parent / "static" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext_map = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
    }

    for i, item in enumerate(prompt_images or []):
        if not isinstance(item, dict):
            continue
        raw_url = str(item.get("url") or "").strip()
        if not raw_url:
            continue

        # 已经是本地静态路径时，直接收录
        if raw_url.startswith("/static/uploads/"):
            out.append({"url": raw_url, "name": item.get("name", "")})
            continue

        if not raw_url.startswith("data:image"):
            continue

        try:
            header, b64_data = raw_url.split(",", 1)
            mime = "image/png"
            if ";" in header:
                mime = header[5:].split(";", 1)[0]
            ext = ext_map.get(mime, ".png")
            filename = f"attach_{int(time.time() * 1000)}_{i}{ext}"
            local_path = upload_dir / filename
            local_path.write_bytes(base64.b64decode(b64_data))
            out.append({
                "url": f"/static/uploads/{filename}",
                "local_path": str(local_path),
                "name": item.get("name", ""),
            })
        except Exception:
            continue
    return out


def _cleanup_attachment_files(attachments: list[dict]) -> None:
    for item in attachments or []:
        if not isinstance(item, dict):
            continue
        local_path = item.get("local_path")
        if not local_path:
            continue
        try:
            p = Path(local_path)
            if p.exists() and p.is_file():
                p.unlink()
        except Exception:
            pass

def _path_tags(s: dict) -> list:
    path    = sm.get_path_to_root(s)
    cur_id  = s["current_node"]
    tags    = []
    for node in path:
        words = [w for w in node["user_input"].replace("，", " ").replace(",", " ").split() if len(w) >= 2]
        for w in words[:3]:
            tags.append({"text": w, "is_new": node["id"] == cur_id})
    return tags


if __name__ == "__main__":
    import webbrowser
    import threading
    import time
    import socket

    def find_available_port(start_port=8000, max_attempts=10):
        """查找可用端口"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("0.0.0.0", port))
                    return port
            except OSError:
                continue
        return start_port + max_attempts

    port = find_available_port()

    def open_browser():
        time.sleep(1.5)  # 等待服务器启动
        webbrowser.open(f"http://localhost:{port}")

    # 启动后自动打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()

    print(f"\n  OfferTree AI求职智能匹配 → http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
