"""计算机网络助教 Agent 命令行程序。

要求:
- Python 3.10+
- OPENAI_API_KEY 环境变量已设置
- pip install openai-agents

运行:
    python network_tutor_agent.py

输入 N、n、Exit、exit 或 退出 可结束对话。
"""

import asyncio
import os
import sys
from typing import Any

from agents import Agent, ModelSettings, Runner
from agents.repl import RawResponsesStreamEvent, RunItemStreamEvent, AgentUpdatedStreamEvent
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key

    # 尝试从 DesignTree 项目 config.json 读取保存的 Key
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            api_key = str(cfg.get("openai_api_key") or cfg.get("api_keys", {}).get("openai", "")).strip()
            if api_key:
                return api_key
        except Exception:
            pass

    raise RuntimeError(
        "请先通过环境变量设置 OPENAI_API_KEY，或通过 DesignTree 页面设置 OpenAI API Key。"
    )


def build_network_tutor_agent() -> Agent[Any]:
    instructions = (
        "你是一名计算机网络课程助教 Agent，擅长解释计算机网络基本概念、协议机制、网络分层、路由与交换、拥塞控制、Socket 编程、HTTP/SSE 等内容。"
        "回答要结构清晰、逻辑完整、表达准确，尽量用中文回答。"
        "如果用户提出具体问题，直接给出完整答案，不要只给片段或简单复述问题。"
        "如果用户希望继续问同一个主题，请保持上下文连贯性。"
        "在回答中可以使用编号、分点、示例代码片段或对比表格来提升可读性。"
    )

    model_settings = ModelSettings(temperature=0.2, top_p=0.95)
    return Agent(
        name="NetworkTutorAgent",
        instructions=instructions,
        model="gpt-4o-mini",
        model_settings=model_settings,
    )


async def run_tutor(agent: Agent[Any]) -> None:
    print("计算机网络助教 Agent 已启动。输入问题后回车，输入 N/Exit/退出 结束。\n")
    current_agent = agent
    input_items = []

    while True:
        try:
            question = input("网络问题 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return

        if not question:
            continue

        if question.lower() in {"n", "exit", "quit", "退出"}:
            print("已退出计算机网络助教。")
            return

        input_items.append({"role": "user", "content": question})

        try:
            response = Runner.run_streamed(
                current_agent,
                input=input_items,
                max_turns=None,
            )

            async for event in response.stream_events():
                if isinstance(event, RawResponsesStreamEvent):
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        sys.stdout.write(event.data.delta)
                        sys.stdout.flush()
                elif isinstance(event, RunItemStreamEvent):
                    if event.item.type == "tool_call_item":
                        print("\n[Agent 调用工具...]", flush=True)
                    elif event.item.type == "tool_call_output_item":
                        print(f"\n[工具输出] {event.item.output}", flush=True)
                elif isinstance(event, AgentUpdatedStreamEvent):
                    print(f"\n[Agent 已更新: {event.new_agent.name}]", flush=True)

            print("\n")
            current_agent = response.last_agent
            input_items = response.to_input_list()

        except Exception as exc:
            print(f"\n[错误] {exc}\n")
            break


def main() -> None:
    try:
        api_key = get_api_key()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    os.environ["OPENAI_API_KEY"] = api_key
    agent = build_network_tutor_agent()
    asyncio.run(run_tutor(agent))


if __name__ == "__main__":
    main()
