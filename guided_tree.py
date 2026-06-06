"""
guided_tree.py — AI求职智能匹配引导式问题系统

交互模式：AI主动提问 → 用户选择/回答 → AI继续引导
所有问题和选项均由AI动态生成，形成一棵对话树。
用户可以在树中回溯选择不同分支，生长出新的子树。
从根到当前叶子的路径就是给AI的对话历史。
"""

import json


# 系统提示词：指导AI如何进行引导式提问
SYSTEM_PROMPT = """你是OfferTree求职智能匹配助手。你的任务是通过逐步提问了解用户的求职背景，然后给出精准的岗位匹配建议。

提问策略（按顺序逐步推进，根据用户回答灵活调整）：
1. 先了解用户感兴趣的领域/行业
2. 再深入了解具体方向/细分领域
3. 了解用户目前的技能和技术栈
4. 了解经验水平和项目经验
5. 了解求职偏好（公司类型、优先因素等）
6. 根据以上信息针对性地深入了解
7. 当你认为已经收集到足够信息（通常5-7轮对话后），给出匹配结果

每次回复必须是JSON格式：
{
  "question": "你要问的问题",
  "options": ["选项1", "选项2", ...],
  "multi_select": false,
  "max_select": 1,
  "action": "ask" 或 "result"
}

要求：
- 只有当问题能明确列举出互斥的备选答案时，才提供options（如行业选择、是/否、技能类别等）
- 对于开放式/描述性问题（如"请描述你的项目经验"、"你的优势是什么"），options设为空数组[]，让用户自由输入
- 绝不要为开放式问题编造占位选项（如["项目1","项目2","项目3"]），这不提供任何价值
- 选项要具体、有区分度，每个选项代表真正不同的选择方向
- 有时候问"是"或"否"也很有价值，例如"你是否有过实习经验？"选项可以是["是","否"]
- 问题要简洁明了，一次只问一个方面
- 当问题允许多个答案时（如"你掌握哪些编程语言？"），设置multi_select为true，max_select为最多可选数量
- 当问题只允许一个答案时，multi_select设为false，max_select设为1
- 根据用户前面的回答动态调整后续问题，做到个性化
- 当收集到足够信息后，action设为"result"
- 用中文提问和回答

只输出JSON，不要其他内容。"""


# 生成匹配结果的系统提示词
MATCH_SYSTEM_PROMPT = """你是一名AI求职匹配专家，擅长根据用户的专业、爱好、技能等信息，精准推荐匹配的岗位方向和具体职位。"""


def build_user_profile(answers: list[dict]) -> str:
    """根据用户的所有回答，构建用户画像文本（用于AI生成匹配结果）

    answers 格式: [{"question": "...", "answer": "..."}, ...]
    """
    parts = []
    for a in answers:
        q = a.get("question", "")
        ans = a.get("answer", "")
        if q and ans:
            parts.append(f"- {q} → {ans}")
    return "\n".join(parts)


def build_tree_for_ui(tree: dict, free_chat_qa: list = None) -> dict:
    """将后端树结构转换为前端可渲染的树结构（选项作为独立子节点）

    free_chat_qa: 继续追问的自由对话QA列表 [{"question":"...","answer":"..."}, ...]
                  会作为叶子节点下的子节点追加到树中。

    前端树结构:
      Question Node:
        {
          "id": "...", "type": "question",
          "label": "问题文本",
          "options": [...], "multi_select": False, "max_select": 1,
          "selected_option": "...", "selected_options": [...],
          "is_current": True/False,
          "children": [Option Node, ...],   # 选项节点列表
          "multi_next": Question Node       # 多选时的下一题
        }

      Option Node:
        {
          "id": "...", "type": "option",
          "label": "选项文本",
          "selected": True/False,
          "is_custom": True/False,          # 是否为用户自定义输入
          "question_id": "...",             # 父问题节点ID（用于分支）
          "children": [Question Node, ...]  # 选中后展开的下一题
        }

      FreeChatQA Node (type="free_chat_qa"):
        {
          "id": "...", "type": "free_chat_qa",
          "label": "问题文本",
          "answer": "回答文本",
          "is_current": True/False
        }

      FreeChatSeparator Node (type="free_chat_separator"):
        {
          "id": "...", "type": "free_chat_separator",
          "label": "继续追问"
        }
    """
    if not tree:
        return None

    def _convert(node: dict, current_leaf_id: str = None) -> dict:
        if not node:
            return None

        node_id = node.get("id", "")
        question = node.get("question", "")
        options = node.get("options", [])
        selected = node.get("selected_option", "")
        selected_options = node.get("selected_options", [])
        multi_select = node.get("multi_select", False)
        max_select = node.get("max_select", 1)
        children = node.get("children", {})

        # 收集所有可见选项（AI提供的 + 用户自定义输入的）
        all_option_labels = list(options)
        for child_key in children:
            if child_key not in all_option_labels:
                all_option_labels.append(child_key)

        # 构建选项子节点
        option_children = []
        for i, opt in enumerate(all_option_labels):
            # 兼容旧数据：如果没有selected_options，从selected_option中拆分
            effective_selected = selected_options if (multi_select and selected_options) else (
                selected.split("、") if (multi_select and selected and "、" in selected) else []
            )
            is_selected = (opt in effective_selected) if effective_selected else (opt == selected)

            # 多选时：只保留选中的预设选项和自定义选项；未选中的预设选项不显示在树中
            # 单选时：保留所有选项（允许在树中切换分支）
            if multi_select and not is_selected and opt in options:
                continue

            opt_node = {
                "id": f"{node_id}_opt{i}",
                "type": "option",
                "label": opt,
                "selected": is_selected,
                "is_custom": opt not in options,
                "question_id": node_id,
                "children": [],
            }
            # 单选：选中的选项展开子树
            if not multi_select and is_selected and opt in children:
                converted = _convert(children[opt], current_leaf_id)
                if converted:
                    opt_node["children"].append(converted)
            option_children.append(opt_node)

        is_current = node_id == current_leaf_id

        result = {
            "id": node_id,
            "type": "question",
            "label": question,
            "options": options,
            "multi_select": multi_select,
            "max_select": max_select,
            "selected_option": selected,
            "selected_options": selected_options,
            "is_current": is_current,
            "children": option_children,
        }

        # 多选：下一题在组合键下
        if multi_select and selected and selected in children:
            converted = _convert(children[selected], current_leaf_id)
            if converted:
                result["multi_next"] = converted

        return result

    # 找到当前叶子节点ID
    leaf_id = _find_current_leaf_id(tree)
    ui_tree = _convert(tree, leaf_id)

    # 如果有自由对话QA，追加到叶子节点下
    if free_chat_qa and ui_tree:
        _append_free_chat_qa(ui_tree, free_chat_qa)

    return ui_tree


def _append_free_chat_qa(ui_tree: dict, free_chat_qa: list) -> None:
    """将自由对话QA作为子节点追加到UI树的叶子节点下

    找到最深的叶子选项节点，在其children中追加：
    1. "继续追问"分隔节点
    2. 每个QA对作为一个 free_chat_qa 类型的节点
    """
    leaf_option = _find_leaf_option_node(ui_tree)
    if not leaf_option:
        # 没有选中选项的叶子，直接挂在question节点上
        leaf_question = _find_leaf_question_node(ui_tree)
        if not leaf_question:
            return
        _attach_free_chat_to_node(leaf_question, free_chat_qa)
    else:
        _attach_free_chat_to_node(leaf_option, free_chat_qa)


def _find_leaf_option_node(node: dict) -> dict:
    """找到叶子节点的最后一个选中选项节点（沿着选中路径走到底）"""
    if node.get("type") == "option":
        if node.get("selected") and node.get("children"):
            for child in node["children"]:
                deeper = _find_leaf_option_node(child)
                if deeper:
                    return deeper
        if node.get("selected"):
            return node
        return None

    if node.get("type") == "question":
        if node.get("children"):
            for child in node["children"]:
                if child.get("type") == "option" and child.get("selected"):
                    deeper = _find_leaf_option_node(child)
                    if deeper:
                        return deeper
        if node.get("multi_next"):
            deeper = _find_leaf_option_node(node["multi_next"])
            if deeper:
                return deeper
    return None


def _find_leaf_question_node(node: dict) -> dict:
    """找到最深的叶子问题节点"""
    if node.get("type") == "question":
        if node.get("children"):
            for child in node["children"]:
                if child.get("type") == "option" and child.get("selected"):
                    if child.get("children"):
                        for sub in child["children"]:
                            deeper = _find_leaf_question_node(sub)
                            if deeper:
                                return deeper
        if node.get("multi_next"):
            deeper = _find_leaf_question_node(node["multi_next"])
            if deeper:
                return deeper
        return node
    return None


def _attach_free_chat_to_node(node: dict, free_chat_qa: list) -> None:
    """在指定节点的children中追加自由对话QA节点"""
    if "children" not in node:
        node["children"] = []

    # 追加"继续追问"分隔节点
    sep_node = {
        "id": "free_chat_separator",
        "type": "free_chat_separator",
        "label": "继续追问",
    }
    node["children"].append(sep_node)

    # 追加每个QA对
    for i, qa in enumerate(free_chat_qa):
        qa_node = {
            "id": f"free_chat_qa_{i}",
            "type": "free_chat_qa",
            "label": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "is_current": (i == len(free_chat_qa) - 1),
        }
        node["children"].append(qa_node)


def _find_current_leaf_id(node: dict) -> str:
    """沿着selected_option路径找到当前叶子节点"""
    if not node:
        return ""
    selected = node.get("selected_option", "")
    children = node.get("children", {})
    if selected and selected in children:
        return _find_current_leaf_id(children[selected])
    return node.get("id", "")


def get_path_answers(tree: dict) -> list[dict]:
    """从根到当前叶子节点，收集路径上所有的问答对"""
    answers = []
    node = tree
    while node:
        q = node.get("question", "")
        selected = node.get("selected_option", "")
        if q and selected:
            answers.append({"question": q, "answer": selected})
        children = node.get("children", {})
        if selected and selected in children:
            node = children[selected]
        else:
            break
    return answers


def get_path_messages(tree: dict) -> list[dict]:
    """从根到当前叶子节点，构建AI对话历史（用于继续提问）

    返回格式: [{"role": "user/assistant", "content": "..."}, ...]
    """
    messages = []
    node = tree
    while node:
        q = node.get("question", "")
        selected = node.get("selected_option", "")
        options = node.get("options", [])

        if q:
            # AI的提问
            ai_msg = json.dumps({"question": q, "options": options, "action": "ask"}, ensure_ascii=False)
            messages.append({"role": "assistant", "content": ai_msg})

        if selected:
            # 用户的回答
            messages.append({"role": "user", "content": selected})

        children = node.get("children", {})
        if selected and selected in children:
            node = children[selected]
        else:
            break
    return messages
