"""
agent.py

The agent: sends messages to the model, runs any tools the model requests,
feeds results back, and returns the final answer.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

load_dotenv()

# --- Client setup (works for OpenAI or OpenRouter) ------------------------

_base_url = os.getenv("OPENAI_BASE_URL") or None
client = OpenAI(base_url=_base_url) if _base_url else OpenAI()

MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a helpful AI agent. "
    "You have access to tools. Use them whenever they help answer the user. "
    "Think step by step, but keep your final answers concise."
)


def run_agent(messages: list) -> str:
    """
    Given a message history (list of dicts), run the agent loop until
    the model produces a final text answer (no more tool calls).
    Returns the assistant's final text reply.
    """
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # Case 1: The model wants to call one or more tools.
        if msg.tool_calls:
            # Append the assistant message (with tool_calls) to history.
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

            # Run each requested tool and append the result as a tool message.
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

                print(f"  [tool] {fn_name}({args}) -> {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

            # Loop again: send the updated history back to the model.
            continue

        # Case 2: The model produced a final text answer.
        reply = msg.content or ""
        messages.append({"role": "assistant", "content": reply})
        return reply


def new_conversation() -> list:
    """Start a fresh conversation history with the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]