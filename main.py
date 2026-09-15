"""
main.py

Entry point. Runs a simple interactive chat loop with the agent.
Type 'exit' or 'quit' to end. Type 'reset' to clear the conversation.
"""

from agent import run_agent, new_conversation


def main():
    messages = new_conversation()

    print("Simple AI Agent — type 'exit' to quit, 'reset' to clear history.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        if user_input.lower() == "reset":
            messages = new_conversation()
            print("[conversation reset]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        reply = run_agent(messages)
        print(f"AI: {reply}\n")


if __name__ == "__main__":
    main()