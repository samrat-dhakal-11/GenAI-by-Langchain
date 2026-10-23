import logging
import os
import sys
import time
import warnings
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

# Keep app-level errors visible; silence noisy logs.
logging.basicConfig(level=logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
EXIT_COMMANDS = {"end", "bye", "exit", "leave"}
MAX_HISTORY_TOKENS = 4000  # Dynamic token limit for sliding window

FORMAT_RULES = (
    "Formatting rules: do not use markdown bold (**) or headings (###). "
    "The system prompt is only guidance; answer the user's question yourself."
)

console = Console()


def load_system_prompt(path: str = "prompt.txt") -> str:
    """Load system prompt from file or return a smart default if missing."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                base = f.read().strip()
            return f"{base}\n\n{FORMAT_RULES}"
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read {path}: {e}. Using default prompt.[/yellow]")
    
    return f"You are a helpful, concise, and highly capable AI assistant.\n\n{FORMAT_RULES}"


def chunk_text(chunk) -> str:
    """Return the text of a streamed chunk securely across block formats."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict)
        )
    return str(content or "")


def trim_history_by_tokens(messages: List[BaseMessage], max_tokens: int) -> List[BaseMessage]:
    """Trim conversation history dynamically to stay under the token threshold."""
    total_tokens = 0
    trimmed = []
    # Iterate from newest to oldest
    for msg in reversed(messages):
        # Rough estimation: ~4 chars per token
        approx_tokens = len(str(msg.content)) // 4 + 1
        if total_tokens + approx_tokens > max_tokens and trimmed:
            break
        total_tokens += approx_tokens
        trimmed.insert(0, msg)
    return trimmed


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def invoke_stream_with_retry(model, payload):
    """Stream model response with built-in exponential backoff retries."""
    return model.stream(payload)


def stream_answer(model, messages) -> str:
    """Stream answer token by token with rich output handling."""
    parts = []
    try:
        stream_gen = invoke_stream_with_retry(model, messages)
        for chunk in stream_gen:
            text = chunk_text(chunk)
            if text:
                print(text, end="", flush=True)
                parts.append(text)
    except Exception as exc:
        console.print(f"\n[bold red]Stream interrupted after retries: {exc}[/bold red]")
    print()
    return "".join(parts)


def main() -> None:
    if not API_KEY:
        console.print("[bold red]Error: No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.[/bold red]")
        sys.exit(1)

    console.print(f"[bold cyan]Initializing Chat Model ({MODEL_NAME})...[/bold cyan]")
    model = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=API_KEY, streaming=True)
    system = SystemMessage(content=load_system_prompt())
    history: List[BaseMessage] = []

    console.print("[green]Session started. Type 'exit', 'quit', or 'bye' to end.[/green]\n")

    while True:
        try:
            user_query = console.input("[bold yellow]Enter your question:[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]Goodbye![/cyan]")
            break

        if not user_query:
            continue
        if user_query.lower() in EXIT_COMMANDS:
            console.print("[cyan]Goodbye![/cyan]")
            break

        history.append(HumanMessage(content=user_query))
        console.print("\n[bold green]Ans:[/bold green]\n")

        # Apply token-aware sliding window
        active_history = trim_history_by_tokens(history, MAX_HISTORY_TOKENS)
        payload = [system, *active_history]

        try:
            answer = stream_answer(model, payload)
            if not answer:
                console.print("[yellow]Notice: Received empty response from model.[/yellow]")
                history.pop()
                continue
            history.append(AIMessage(content=answer))
        except Exception as exc:
            console.print(f"\n[bold red]Request failed: {exc}[/bold red]\n")
            history.pop()  # Drop unanswered query to keep state consistent

        print()


if __name__ == "__main__":
    main()
