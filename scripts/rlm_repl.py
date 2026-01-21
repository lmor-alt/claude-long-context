#!/usr/bin/env python3
"""
RLM REPL - Recursive Language Model execution environment.

Provides a Python REPL with:
- `context`: The loaded input text
- Variables persist across executions within a session

Sub-agent calls are handled by Claude using the Task tool, not by this script.

Usage:
    # Initialize session with context file
    python rlm_repl.py init <context_file> [--session SESSION_ID]

    # Execute code in session
    python rlm_repl.py exec <session_id> "<code>"

    # Get session info
    python rlm_repl.py info <session_id>

    # Store a sub-agent result
    python rlm_repl.py store <session_id> <var_name> <value>

    # Clean up session
    python rlm_repl.py cleanup <session_id>
"""

import argparse
import json
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

# Session storage directory
SESSION_DIR = Path(tempfile.gettempdir()) / "rlm_sessions"
SESSION_DIR.mkdir(exist_ok=True)


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


def create_execution_namespace(context: str, variables: dict) -> dict:
    """Create the execution namespace with context."""
    namespace = {
        "context": context,
        "print": print,
        "__builtins__": __builtins__,
    }
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

    # Load session
    try:
        state = load_session(session_id)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Create execution namespace
    namespace = create_execution_namespace(
        state["context"],
        state.get("variables", {})
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
    init_parser = subparsers.add_parser("init", help="Initialize new session")
    init_parser.add_argument("context_file", help="Path to context file")
    init_parser.add_argument("--session", help="Custom session ID")
    init_parser.set_defaults(func=cmd_init)

    # exec command
    exec_parser = subparsers.add_parser("exec", help="Execute code in session")
    exec_parser.add_argument("session_id", help="Session ID")
    exec_parser.add_argument("code", help="Python code to execute")
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
