#!/usr/bin/env python3
"""
RLM REPL - Recursive Language Model execution environment.

Provides a Python REPL with:
- `context`: The loaded input text
- `llm_query(prompt, model)`: Call LLM for semantic analysis
- Variables persist across executions within a session

Usage:
    # Initialize session with context file
    python rlm_repl.py init <context_file> [--session SESSION_ID]

    # Initialize session from directory (combines files matching glob)
    python rlm_repl.py init-dir <directory> [--glob "**/*.py"] [--session SESSION_ID]

    # Execute code in session (with LLM support)
    python rlm_repl.py exec <session_id> "<code>" [--llm]

    # Get session info
    python rlm_repl.py info <session_id>

    # Store a sub-agent result
    python rlm_repl.py store <session_id> <var_name> <value>

    # Clean up session
    python rlm_repl.py cleanup <session_id>

Environment:
    ANTHROPIC_API_KEY: Required for llm_query() calls
"""

import argparse
import fnmatch
import json
import os
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

# Session storage directory
SESSION_DIR = Path(tempfile.gettempdir()) / "rlm_sessions"
SESSION_DIR.mkdir(exist_ok=True)

# LLM query tracking for cost awareness
_llm_call_count = 0
_llm_total_tokens = 0


def create_llm_query_function(session_id: str) -> Callable:
    """
    Create an llm_query function bound to a session.

    Uses Anthropic API directly for sub-agent calls within RLM execution.
    Requires ANTHROPIC_API_KEY environment variable.
    """
    def llm_query(
        prompt: str,
        model: str = "haiku",
        max_tokens: int = 2048,
    ) -> str:
        """
        Query an LLM for semantic analysis of context chunks.

        Args:
            prompt: The prompt to send to the LLM
            model: Model shorthand - "haiku", "sonnet", or "opus"
            max_tokens: Maximum tokens in response

        Returns:
            The LLM's response text
        """
        global _llm_call_count, _llm_total_tokens

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Required for llm_query().\n"
                "Set it with: export ANTHROPIC_API_KEY=your-key"
            )

        # Map shorthand to full model names
        model_map = {
            "haiku": "claude-3-5-haiku-20241022",
            "sonnet": "claude-sonnet-4-20250514",
            "opus": "claude-opus-4-20250514",
        }
        full_model = model_map.get(model, model)

        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=api_key)

        _llm_call_count += 1
        print(f"[LLM Call #{_llm_call_count}] Model: {full_model}, Prompt: {len(prompt)} chars")

        try:
            response = client.messages.create(
                model=full_model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            result = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            _llm_total_tokens += tokens_used

            print(f"[LLM Response] {tokens_used} tokens, {len(result)} chars")
            return result

        except Exception as e:
            print(f"[LLM ERROR] {type(e).__name__}: {e}")
            raise

    return llm_query


def get_llm_stats() -> dict:
    """Return LLM usage statistics for the session."""
    return {
        "calls": _llm_call_count,
        "total_tokens": _llm_total_tokens,
    }


def reset_llm_stats() -> None:
    """Reset LLM usage statistics."""
    global _llm_call_count, _llm_total_tokens
    _llm_call_count = 0
    _llm_total_tokens = 0


def get_session_path(session_id: str) -> Path:
    """Get path to session state file."""
    return SESSION_DIR / f"{session_id}.json"


def load_session(session_id: str) -> dict:
    """Load session state from disk."""
    path = get_session_path(session_id)
    if not path.exists():
        raise ValueError(f"Session {session_id} not found")
    with open(path) as f:
        return json.load(f)


def save_session(session_id: str, state: dict) -> None:
    """Save session state to disk."""
    path = get_session_path(session_id)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def create_execution_namespace(
    context: str,
    variables: dict,
    session_id: str = "",
    enable_llm: bool = False,
) -> dict:
    """Create the execution namespace with context and optional LLM support."""
    namespace = {
        "context": context,
        "print": print,
        "__builtins__": __builtins__,
        "get_llm_stats": get_llm_stats,
    }

    if enable_llm:
        namespace["llm_query"] = create_llm_query_function(session_id)
    else:
        # Provide a stub that gives a helpful error
        def llm_query_stub(*args, **kwargs):
            raise RuntimeError(
                "llm_query() not enabled. Run exec with --llm flag:\n"
                "  python rlm_repl.py exec <session> '<code>' --llm"
            )
        namespace["llm_query"] = llm_query_stub

    # Add any persisted variables from previous executions
    namespace.update(variables)
    return namespace


def execute_code(code: str, namespace: dict) -> tuple[str, dict, Any]:
    """
    Execute code in the given namespace.

    Returns:
        tuple of (output, updated_variables, final_result_or_none)
    """
    import io
    from contextlib import redirect_stdout, redirect_stderr

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    final_result = None

    # Check for FINAL markers before execution
    if "FINAL(" in code or "FINAL_VAR(" in code:
        # Extract the final value
        if "FINAL_VAR(" in code:
            # Extract variable name
            import re
            match = re.search(r'FINAL_VAR\((\w+)\)', code)
            if match:
                var_name = match.group(1)
                if var_name in namespace:
                    final_result = ("VAR", var_name, namespace[var_name])
                else:
                    final_result = ("ERROR", f"Variable '{var_name}' not found")
        elif "FINAL(" in code:
            import re
            match = re.search(r'FINAL\(["\'](.+?)["\']\)', code, re.DOTALL)
            if match:
                final_result = ("DIRECT", match.group(1))

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)
    except Exception as e:
        stderr_capture.write(f"\n[EXECUTION ERROR] {type(e).__name__}: {e}\n")
        stderr_capture.write(traceback.format_exc())

    output = stdout_capture.getvalue()
    errors = stderr_capture.getvalue()

    if errors:
        output += f"\n[STDERR]\n{errors}"

    # Extract variables that should persist (exclude builtins and functions)
    persistable = {}
    for key, value in namespace.items():
        if key.startswith("_"):
            continue
        if key in ("context", "llm_query", "print"):
            continue
        if callable(value) and not isinstance(value, type):
            continue
        try:
            # Test if JSON serializable
            json.dumps(value, default=str)
            persistable[key] = value
        except (TypeError, ValueError):
            # Store string representation for non-serializable objects
            persistable[key] = str(value)

    return output, persistable, final_result


def cmd_init(args):
    """Initialize a new RLM session with context from file."""
    context_file = Path(args.context_file).expanduser()

    if not context_file.exists():
        print(f"[ERROR] Context file not found: {context_file}", file=sys.stderr)
        sys.exit(1)

    # Read context
    with open(context_file) as f:
        context = f.read()

    # Generate session ID
    session_id = args.session or str(uuid.uuid4())[:8]

    # Determine context type
    if context_file.suffix in (".json",):
        context_type = "JSON"
    elif context_file.suffix in (".md", ".txt"):
        context_type = "text"
    elif context_file.suffix in (".py", ".js", ".ts", ".go", ".rs"):
        context_type = "code"
    else:
        context_type = "text"

    # Create session state
    state = {
        "session_id": session_id,
        "context_file": str(context_file),
        "context_length": len(context),
        "context_type": context_type,
        "context": context,
        "variables": {},
        "iteration": 0,
        "history": []
    }

    save_session(session_id, state)

    # Output session info
    print(f"RLM Session Initialized")
    print(f"Session ID: {session_id}")
    print(f"Context: {context_type} with {len(context):,} characters")
    print(f"")
    print(f"Available in REPL:")
    print(f"  - context: Your input data ({len(context):,} chars)")
    print(f"  - All standard Python builtins")
    print(f"")
    print(f"Commands:")
    print(f"  exec {session_id} \"<code>\"  - Execute Python code")
    print(f"  store {session_id} <var> <value>  - Store sub-agent result")
    print(f"  info {session_id}  - Show session info")


def cmd_exec(args):
    """Execute code in an existing session."""
    session_id = args.session_id
    code = args.code
    enable_llm = getattr(args, "llm", False)

    # Load session
    try:
        state = load_session(session_id)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Reset LLM stats for this execution
    if enable_llm:
        reset_llm_stats()

    # Create execution namespace
    namespace = create_execution_namespace(
        state["context"],
        state.get("variables", {}),
        session_id=session_id,
        enable_llm=enable_llm,
    )

    # Execute code
    output, variables, final_result = execute_code(code, namespace)

    # Update session state
    state["variables"] = variables
    state["iteration"] += 1
    state["history"].append({
        "iteration": state["iteration"],
        "code": code[:500],  # Truncate for storage
        "output_preview": output[:500] if output else ""
    })

    save_session(session_id, state)

    # Print output
    if output:
        # Truncate very long outputs
        if len(output) > 10000:
            print(output[:10000])
            print(f"\n[OUTPUT TRUNCATED - {len(output):,} total characters]")
        else:
            print(output)

    # Handle final result
    if final_result:
        result_type, *result_data = final_result
        print(f"\n{'='*50}")
        print(f"[RLM FINAL RESULT]")
        if result_type == "DIRECT":
            print(f"Answer: {result_data[0]}")
        elif result_type == "VAR":
            var_name, var_value = result_data
            print(f"Variable: {var_name}")
            if isinstance(var_value, (list, dict)):
                print(json.dumps(var_value, indent=2, default=str)[:5000])
            else:
                print(str(var_value)[:5000])
        elif result_type == "ERROR":
            print(f"Error: {result_data[0]}")
        print(f"{'='*50}")

    # Print LLM usage stats if enabled
    if enable_llm:
        stats = get_llm_stats()
        if stats["calls"] > 0:
            print(f"\n[LLM Usage] {stats['calls']} calls, {stats['total_tokens']:,} tokens")


def cmd_info(args):
    """Show session information."""
    session_id = args.session_id

    try:
        state = load_session(session_id)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Session: {session_id}")
    print(f"Context file: {state['context_file']}")
    print(f"Context length: {state['context_length']:,} characters")
    print(f"Context type: {state['context_type']}")
    print(f"Iterations: {state['iteration']}")
    print(f"Variables: {list(state.get('variables', {}).keys())}")


def cmd_store(args):
    """Store a value (e.g., sub-agent result) in session variables."""
    session_id = args.session_id
    var_name = args.var_name
    value = args.value

    try:
        state = load_session(session_id)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Try to parse as JSON, otherwise store as string
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value

    state["variables"][var_name] = parsed_value
    save_session(session_id, state)

    print(f"Stored '{var_name}' = {str(parsed_value)[:200]}...")


def cmd_cleanup(args):
    """Clean up a session."""
    session_id = args.session_id
    path = get_session_path(session_id)

    if path.exists():
        path.unlink()
        print(f"Session {session_id} cleaned up")
    else:
        print(f"Session {session_id} not found")


def cmd_init_dir(args):
    """Initialize a new RLM session from a directory of files."""
    directory = Path(args.directory).expanduser().resolve()
    glob_pattern = args.glob or "**/*"
    exclude_patterns = (args.exclude or "").split(",") if args.exclude else []

    if not directory.exists():
        print(f"[ERROR] Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)

    if not directory.is_dir():
        print(f"[ERROR] Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    # Collect matching files
    files_found = []
    for path in directory.glob(glob_pattern):
        if path.is_file():
            # Check exclusions
            rel_path = str(path.relative_to(directory))
            if any(fnmatch.fnmatch(rel_path, excl.strip()) for excl in exclude_patterns if excl.strip()):
                continue
            # Skip common non-text files
            if path.suffix.lower() in (".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".png", ".jpg", ".gif", ".pdf"):
                continue
            # Skip hidden and cache directories
            if any(part.startswith(".") or part == "__pycache__" or part == "node_modules" for part in path.parts):
                continue
            files_found.append(path)

    if not files_found:
        print(f"[ERROR] No files matched pattern '{glob_pattern}' in {directory}", file=sys.stderr)
        sys.exit(1)

    # Sort for consistent ordering
    files_found.sort()

    # Combine files into context
    context_parts = []
    for file_path in files_found:
        try:
            rel_path = file_path.relative_to(directory)
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            context_parts.append(f"=== FILE: {rel_path} ===\n{content}\n")
        except Exception as e:
            print(f"[WARN] Skipping {file_path}: {e}", file=sys.stderr)

    context = "\n".join(context_parts)

    # Generate session ID
    session_id = args.session or str(uuid.uuid4())[:8]

    # Create session state
    state = {
        "session_id": session_id,
        "context_file": str(directory),
        "context_length": len(context),
        "context_type": "directory",
        "files_count": len(files_found),
        "glob_pattern": glob_pattern,
        "context": context,
        "variables": {},
        "iteration": 0,
        "history": [],
    }

    save_session(session_id, state)

    # Output session info
    print(f"RLM Session Initialized (Directory Mode)")
    print(f"Session ID: {session_id}")
    print(f"Directory: {directory}")
    print(f"Pattern: {glob_pattern}")
    print(f"Files: {len(files_found)}")
    print(f"Context: {len(context):,} characters")
    print(f"")
    print(f"Available in REPL:")
    print(f"  - context: Combined file contents ({len(context):,} chars)")
    print(f"  - llm_query(prompt, model): LLM calls (with --llm flag)")
    print(f"")
    print(f"Commands:")
    print(f"  exec {session_id} \"<code>\" --llm  - Execute with LLM support")
    print(f"  info {session_id}  - Show session info")


def cmd_list(args):
    """List all active sessions."""
    sessions = list(SESSION_DIR.glob("*.json"))

    if not sessions:
        print("No active RLM sessions")
        return

    print(f"Active RLM Sessions ({len(sessions)}):")
    print("-" * 60)

    for session_file in sessions:
        try:
            with open(session_file) as f:
                state = json.load(f)
            session_id = state.get("session_id", session_file.stem)
            context_len = state.get("context_length", 0)
            iterations = state.get("iteration", 0)
            print(f"  {session_id}: {context_len:,} chars, {iterations} iterations")
        except Exception:
            print(f"  {session_file.stem}: [corrupted]")


def main():
    parser = argparse.ArgumentParser(
        description="RLM REPL - Recursive Language Model execution environment"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize new session from file")
    init_parser.add_argument("context_file", help="Path to context file")
    init_parser.add_argument("--session", help="Custom session ID")
    init_parser.set_defaults(func=cmd_init)

    # init-dir command
    init_dir_parser = subparsers.add_parser("init-dir", help="Initialize session from directory")
    init_dir_parser.add_argument("directory", help="Path to directory")
    init_dir_parser.add_argument("--glob", default="**/*.py", help="Glob pattern (default: **/*.py)")
    init_dir_parser.add_argument("--exclude", help="Comma-separated exclusion patterns")
    init_dir_parser.add_argument("--session", help="Custom session ID")
    init_dir_parser.set_defaults(func=cmd_init_dir)

    # exec command
    exec_parser = subparsers.add_parser("exec", help="Execute code in session")
    exec_parser.add_argument("session_id", help="Session ID")
    exec_parser.add_argument("code", help="Python code to execute")
    exec_parser.add_argument("--llm", action="store_true", help="Enable llm_query() for API calls")
    exec_parser.set_defaults(func=cmd_exec)

    # info command
    info_parser = subparsers.add_parser("info", help="Show session info")
    info_parser.add_argument("session_id", help="Session ID")
    info_parser.set_defaults(func=cmd_info)

    # store command
    store_parser = subparsers.add_parser("store", help="Store value in session")
    store_parser.add_argument("session_id", help="Session ID")
    store_parser.add_argument("var_name", help="Variable name")
    store_parser.add_argument("value", help="Value to store (string or JSON)")
    store_parser.set_defaults(func=cmd_store)

    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up session")
    cleanup_parser.add_argument("session_id", help="Session ID")
    cleanup_parser.set_defaults(func=cmd_cleanup)

    # list command
    list_parser = subparsers.add_parser("list", help="List active sessions")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
