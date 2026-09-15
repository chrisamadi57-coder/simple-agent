"""
tools.py

Python functions that the AI agent can request to run.
Each function is paired with a JSON schema description in TOOL_SCHEMAS.
"""

import ast
import operator


# --- Safe calculator (no eval) -------------------------------------------

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Num):  # Python <3.8
        return node.n
    if isinstance(node, ast.Constant):  # Python >=3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unary operator not allowed: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Safely evaluate a basic math expression like '25 * 48'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# --- Text utilities -------------------------------------------------------

def word_count(text: str) -> str:
    """Return the number of words in the given text."""
    count = len(text.split())
    return f"{count} words"


def reverse_text(text: str) -> str:
    """Return the text reversed."""
    return text[::-1]


# --- Current time ---------------------------------------------------------

from datetime import datetime

def current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- Registry -------------------------------------------------------------
# Maps a function name (string) -> the actual Python function.
# The agent uses this to dispatch tool calls from the model.

TOOL_FUNCTIONS = {
    "calculator": calculator,
    "word_count": word_count,
    "reverse_text": reverse_text,
    "current_time": current_time,
}


# --- Schemas --------------------------------------------------------------
# Descriptions the model sees. These tell the model WHEN and HOW to use each tool.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression. Use for any math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '25 * 48' or '(3 + 5) / 2'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "word_count",
            "description": "Count how many words are in a piece of text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to count."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reverse_text",
            "description": "Reverse a string of text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to reverse."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "Get the current date and time. Use when the user asks what time or date it is.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]