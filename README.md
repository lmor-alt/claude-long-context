# claude-long-context

A Claude Code skill for processing arbitrarily long contexts using the Recursive Language Model (RLM) approach.

Based on the [RLM paper (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601).

## The Problem

Claude Code has a ~200K token context window. When you need to analyze:
- Large codebases (500K+ tokens)
- Long documents or logs
- Multi-file aggregation tasks

...the context simply doesn't fit. Traditional approaches (truncation, summarization) lose information.

## The Solution

RLM treats long contexts as **external environment variables** rather than stuffing them into the prompt. Claude writes Python code to programmatically access, filter, and chunk the context—then uses `llm_query()` to analyze each chunk.

```
┌─────────────────────────────────────────────────────┐
│  User Query + Long Context                          │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  REPL Environment                                   │
│  ┌───────────────────────────────────────────────┐  │
│  │ context = "..." (loaded from file/directory)  │  │
│  │ llm_query(prompt, model) → API call           │  │
│  │ print() → observe results                     │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
    ┌──────────────┐          ┌──────────────┐
    │ llm_query 1  │          │ llm_query N  │
    │ (chunk 1)    │   ...    │ (chunk N)    │
    └──────────────┘          └──────────────┘
                         │
                         ▼
              ┌─────────────────┐
              │  Final Answer   │
              └─────────────────┘
```

## Installation

```bash
# Clone to your Claude Code skills directory
git clone https://github.com/lmor-alt/claude-long-context.git ~/.claude/skills/rlm

# Install the anthropic package (required for llm_query)
pip install anthropic

# Set your API key
export ANTHROPIC_API_KEY=your-key
```

## Quick Start

### From a Directory (Most Common)

```bash
# Initialize from a codebase
python ~/.claude/skills/rlm/scripts/rlm_repl.py init-dir ~/project --glob "**/*.py" --session myproj

# Execute with LLM support
python rlm_repl.py exec myproj "
import re
funcs = re.findall(r'def (\w+)', context)
print(f'Found {len(funcs)} functions')

# Use LLM to analyze
summary = llm_query(f'Summarize these functions: {funcs[:20]}', model='haiku')
print(summary)
" --llm
```

### From a Single File

```bash
python rlm_repl.py init ~/large_document.txt --session doc1
python rlm_repl.py exec doc1 "print(len(context))"
```

## Usage

### Slash Command (in Claude Code)

```
/rlm ~/path/to/large_file.txt "Find all API endpoints and their authentication methods"
```

### CLI Commands

```bash
# Initialize from file
python rlm_repl.py init <file_path> [--session SESSION_ID]

# Initialize from directory
python rlm_repl.py init-dir <directory> [--glob "**/*.py"] [--exclude "tests/*"] [--session SESSION_ID]

# Execute code (add --llm for API calls)
python rlm_repl.py exec <session_id> "<code>" [--llm]

# Other commands
python rlm_repl.py info <session_id>      # Show session info
python rlm_repl.py store <session_id> <var> <value>  # Store a value
python rlm_repl.py list                   # List active sessions
python rlm_repl.py cleanup <session_id>   # Clean up session
```

## Available in REPL

| Name | Description |
|------|-------------|
| `context` | The loaded text content |
| `llm_query(prompt, model)` | Call LLM API (requires `--llm` flag) |
| `get_llm_stats()` | Returns `{calls, total_tokens}` |
| `FINAL(answer)` | Mark final answer (string) |
| `FINAL_VAR(varname)` | Mark final answer (variable) |

### llm_query() Function

```python
llm_query(
    prompt: str,           # The prompt to send
    model: str = "haiku",  # "haiku", "sonnet", or "opus"
    max_tokens: int = 2048 # Max response tokens
) -> str
```

## Core Patterns

### Filtered Search
Use regex to narrow down, then LLM for semantic understanding:

```python
import re

# Find relevant sections (cheap)
matches = re.findall(r'(def |class )\w+', context)
print(f"Found {len(matches)} definitions")

# Analyze with LLM (expensive but targeted)
for match in matches[:5]:
    idx = context.find(match)
    snippet = context[idx:idx+500]
    analysis = llm_query(f"Analyze: {snippet}", model="haiku")
    print(f"{match}: {analysis}")
```

### Hierarchical Decomposition
Split by structure, summarize each, then synthesize:

```python
import re

# Split by file markers
files = re.split(r'=== FILE: (.+?) ===', context)

# Summarize each file
summaries = {}
for i in range(1, len(files), 2):
    filename, content = files[i], files[i+1]
    summaries[filename] = llm_query(f"Summarize: {content[:5000]}", model="haiku")

# Synthesize
final = llm_query(f"Create overview: {summaries}", model="sonnet")
```

## Cost Awareness

- Each `llm_query()` call costs tokens
- Filter with code first, then use LLM for semantic understanding
- Use haiku (~$0.25/M input) for simple tasks, sonnet for synthesis
- Stats are printed at the end of each `--llm` execution:
  ```
  [LLM Usage] 5 calls, 12,345 tokens
  ```

## Requirements

- Claude Code with skills enabled
- Python 3.8+
- `anthropic` package (`pip install anthropic`)
- `ANTHROPIC_API_KEY` environment variable

## License

MIT

## References

- [RLM Paper (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601)
- [Claude Code Skills Documentation](https://docs.anthropic.com/en/docs/claude-code/skills)
