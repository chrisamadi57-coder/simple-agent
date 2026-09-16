"""
agent.py

The agent: sends messages to the model, runs any tools the model requests,
feeds results back, and returns the final answer.

Optional `on_tool_call` callback lets a UI display tool activity.
"""

import json
import os
from typing import Callable, Optional
from openai import OpenAI
from dotenv import load_dotenv

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

load_dotenv()

_base_url = os.getenv("OPENAI_BASE_URL") or None
client = OpenAI(base_url=_base_url) if _base_url else OpenAI()

MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a helpful AI agent. "
    "You have access to tools. Use them whenever they help answer the user. "
    "Think step by step, but keep your final answers concise."
)


def run_agent(
    messages: list,
    on_tool_call: Optional[Callable[[str, dict, str], None]] = None,
) -> str:
    """
    Run the agent loop until the model produces a final text answer.

    Args:
        messages: The message history (list of dicts). Mutated in place.
        on_tool_call: Optional callback(name, args, result) invoked for each
                      tool execution. Used by UIs to display activity.

    Returns:
        The assistant's final text reply.
    """
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                raw_args = tc.function.arguments or "{}"

                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}

                fn = TOOL_FUNCTIONS.get(fn_name)
                if fn is None:
                    result = f"Error: unknown tool '{fn_name}'"
                else:
                    try:
                        result = fn(**args)
                    except Exception as e:
                        result = f"Error running {fn_name}: {e}"

                # Notify the UI, if any
                if on_tool_call:
                    on_tool_call(fn_name, args, str(result))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

            continue

        reply = msg.content or ""
        messages.append({"role": "assistant", "content": reply})
        return reply


def new_conversation() -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT}]