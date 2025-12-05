#!/usr/bin/env python3
"""
Run a "General Code Review Prompt" against a code file via the OpenAI API.

Usage:
    python code_review.py path/to/file.py

Optional flags:
    --language "Python"
    --purpose "Flask API for LED matrix control"
    --env "Flask, Python 3.11, macOS"

Example:
    python code_review.py app.py \
        --purpose "Web app to control a 16x64 LED matrix via UDP/DDP" \
        --env "Flask, Pillow, UDP sockets, macOS"
"""

import os
import sys
import argparse
from pathlib import Path

from openai import OpenAI

DEFAULT_API_KEY_FILE = Path.home() / "Documents" / "keys" / "openai.txt"


def read_api_key_from_file(filepath: Path) -> str | None:
    """Return an API key stored in a plaintext file, if present."""
    try:
        return filepath.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return None


def set_api_key_from_file(filepath: Path) -> str | None:
    """Read an API key file and set OPENAI_API_KEY when successful."""
    api_key = read_api_key_from_file(filepath)
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    return api_key
# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "gpt-5.1"  # you can change to "gpt-4.1" or "gpt-4.1-mini" etc.


def detect_language_from_extension(path: Path) -> str:
    """Best-effort language guess based on file extension."""
    ext = path.suffix.lower()
    return {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".jsx": "React (JavaScript JSX)",
        ".tsx": "React (TypeScript TSX)",
        ".rb": "Ruby",
        ".go": "Go",
        ".rs": "Rust",
        ".java": "Java",
        ".cs": "C#",
        ".cpp": "C++",
        ".cc": "C++",
        ".cxx": "C++",
        ".c": "C",
        ".php": "PHP",
        ".html": "HTML",
        ".css": "CSS",
        ".sh": "Shell script",
        ".bash": "Bash",
        ".zsh": "Zsh",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".sql": "SQL",
    }.get(ext, f"Unknown (file extension: {ext or 'none'})")


def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_bytes().decode("utf-8", errors="replace")


def build_system_prompt(language: str, purpose: str, env: str) -> str:
    """
    Build the System Prompt by filling in your General Code Review Prompt template
    with the provided context.
    """
    # Ensure we always have *something* in each slot
    language = language or "Unspecified"
    purpose = purpose or "Not explicitly specified"
    env = env or "Not explicitly specified"

    return f"""
You are an expert software engineer and code reviewer.

I’d like you to perform a detailed code review.

Context:

Language: {language}

Purpose of the code: {purpose}

Environment (if relevant): {env}

Task:

Correctness & Bugs

Look for logical errors, edge cases that may fail, off-by-one issues, race conditions, and incorrect assumptions.

Point out any parts that are likely to throw exceptions or behave unexpectedly.

Readability & Style

Comment on naming, structure, formatting, and clarity.

Suggest small refactors that would make the code easier to understand and maintain (e.g., breaking up long functions, removing duplication).

Design & Architecture

Evaluate the overall design: separation of concerns, abstraction level, single-responsibility, and modularity.

Suggest improvements if there’s a cleaner pattern or architecture (e.g., using dependency injection, strategy pattern, etc., if appropriate).

Performance & Complexity

Identify any obvious performance issues (unnecessary loops, expensive operations in hot paths, excessive allocations, N^2 patterns, etc.).

If useful, comment on time/space complexity of the main operations and how to improve them.

Security & Robustness (if applicable)

Call out potential security issues (injection, unsafe input handling, hard-coded secrets, insecure crypto, etc.).

Suggest better validation, error handling, and logging.

Testing

Suggest unit/integration tests that should exist, especially for edge cases.

Point out any parts that look hard to test and how to refactor them to be more testable.

Output format:

Start with a 1–2 paragraph summary of the overall quality.

Then provide a bullet-point list under these headings:

“Bugs / Correctness Issues”

“Readability & Style”

“Design & Architecture”

“Performance Considerations”

“Security / Robustness” (if relevant)

“Suggested Tests”

Where useful, include small code snippets to show how you would improve things.
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a General Code Review Prompt against a file.")
    parser.add_argument("file", type=str, help="Path to the code file to review")
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Programming language (if omitted, guessed from file extension)",
    )
    parser.add_argument(
        "--purpose",
        type=str,
        default=None,
        help="Brief description of what the code should do",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment: framework / library / runtime / OS",
    )
    parser.add_argument(
        "--api-key-file",
        type=str,
        default=None,
        help="Path to the plaintext file holding the OpenAI API key (default: ~/Documents/keys/openai.txt)",
    )

    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    code_text = read_file(file_path)

    # Resolve language if not explicitly given
    language = args.language or detect_language_from_extension(file_path)
    purpose = args.purpose
    env = args.env

    system_prompt = build_system_prompt(language, purpose, env)

    user_message = f"""Please review the following code using the instructions in the system prompt.

Filename: {file_path.name}

```text
{code_text}
```"""

    # Determine which file (if any) to read from
    key_file_arg = args.api_key_file or os.getenv("OPENAI_API_KEY_FILE")
    if key_file_arg:
        key_path = Path(key_file_arg).expanduser()
    else:
        key_path = DEFAULT_API_KEY_FILE

    api_key = set_api_key_from_file(key_path)
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Provide OPENAI_API_KEY or place the key in the configured key file.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Running code review on: {file_path}", file=sys.stderr)
    print(f"Language: {language}", file=sys.stderr)
    if purpose:
        print(f"Purpose: {purpose}", file=sys.stderr)
    if env:
        print(f"Environment: {env}", file=sys.stderr)
    print("", file=sys.stderr)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    review = response.choices[0].message.content
    print(review)


if __name__ == "__main__":
    main()
