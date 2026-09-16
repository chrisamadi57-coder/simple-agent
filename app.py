"""
app.py

Streamlit web UI for the Simple AI Agent.

Run with:
    streamlit run app.py
"""

import streamlit as st

from agent import run_agent, new_conversation

# --- Page config ----------------------------------------------------------

st.set_page_config(
    page_title="Simple AI Agent",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Simple AI Agent")
st.caption("A from-scratch agent with tool calling. Type a message below.")

# --- Session state --------------------------------------------------------
# Streamlit re-runs the whole script on every interaction, so state must
# live in st.session_state to persist across reruns.

if "messages" not in st.session_state:
    # The full model history (system + user + assistant + tool messages)
    st.session_state.messages = new_conversation()

if "display" not in st.session_state:
    # What we show in the chat UI (a friendlier view than raw messages)
    st.session_state.display = []  # list of dicts: {role, content, tools}

# --- Sidebar controls -----------------------------------------------------

with st.sidebar:
    st.header("Controls")

    if st.button("🗑️ Reset conversation", use_container_width=True):
        st.session_state.messages = new_conversation()
        st.session_state.display = []
        st.rerun()

    st.divider()
    st.markdown("**Available tools**")
    st.markdown(
        "- `calculator` — math\n"
        "- `word_count` — count words\n"
        "- `reverse_text` — reverse text\n"
        "- `current_time` — date/time"
    )

# --- Render chat history --------------------------------------------------

for item in st.session_state.display:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        # If this turn used tools, show them in an expander
        if item.get("tools"):
            with st.expander(f"🔧 {len(item['tools'])} tool call(s)"):
                for t in item["tools"]:
                    st.markdown(f"**`{t['name']}`**  \nArgs: `{t['args']}`  \nResult: `{t['result']}`")

# --- Chat input -----------------------------------------------------------

user_input = st.chat_input("Ask me anything...")

if user_input:
    # 1. Show the user's message immediately
    st.session_state.display.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Add to model history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 3. Run the agent, collecting tool events for display
    tool_events = []

    def record_tool(name, args, result):
        tool_events.append({"name": name, "args": args, "result": result})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = run_agent(st.session_state.messages, on_tool_call=record_tool)

        st.markdown(reply)

        if tool_events:
            with st.expander(f"🔧 {len(tool_events)} tool call(s)"):
                for t in tool_events:
                    st.markdown(
                        f"**10:45

Challenges

Roman Numeral Converter
Intermediate

Merge Overlapping Intervals`{t['name']}`**  \n"
                        f"Args: `{t['args']}`  \n"
                        f"Result: `{t['result']}`"
                    )

    # 4. Save assistant turn to display state
    st.session_state.display.append({
        "role": "assistant",
        "content": reply,
        "tools": tool_events,
    })