import logging
import os
import warnings

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Keep app-level errors visible; silence only the noisy Google/absl output.
logging.basicConfig(level=logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
EXIT_COMMANDS = {"end", "bye", "exit", "leave"}
MAX_HISTORY = 20  # number of recent messages sent per request (keep even)

FORMAT_RULES = (
    "Formatting rules: do not use markdown bold (**) or headings (###). "
    "The system prompt is only guidance; answer the user's question yourself."
)


def load_system_prompt(path: str = "prompt.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        base = f.read().strip()
    return f"{base}\n\n{FORMAT_RULES}"


def chunk_text(chunk) -> str:
    """Return the text of a streamed chunk, whether content is a str or a list of blocks."""
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(
        block["text"] for block in content
        if isinstance(block, dict) and "text" in block
    )


def stream_answer(model, messages) -> str:
    """Print the answer token by token and return the complete text."""
    parts = []
    for chunk in model.stream(messages):
        text = chunk_text(chunk)
        if text:
            print(text, end="", flush=True)
            parts.append(text)
    print()
    return "".join(parts)


def main() -> None:
    if not API_KEY:
        raise SystemExit("No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.")

    model = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=API_KEY)
    system = SystemMessage(content=load_system_prompt())
    history = []

    while True:
        try:
            user_query = input("Enter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_query:
            continue
        if user_query.lower() in EXIT_COMMANDS:
            break

        history.append(HumanMessage(content=user_query))
        print("\nAns:\n")

        try:
            answer = stream_answer(model, [system, *history[-MAX_HISTORY:]])
        except Exception as exc:
            print(f"\nRequest failed: {exc}\n")
            history.pop()  # drop the unanswered question so the history stays consistent
            continue

        history.append(AIMessage(content=answer))
        print()


if __name__ == "__main__":
    main()