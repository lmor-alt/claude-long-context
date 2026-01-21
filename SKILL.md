---
name: rlm
description: Recursive Language Model inference for processing arbitrarily long contexts. Treats prompts as external environment variables, enabling programmatic decomposition and recursive sub-agent calls. Use for documents/codebases exceeding 100K tokens or information-dense aggregation tasks.
---

# Recursive Language Model (RLM)

Based on the RLM paper (arXiv:2512.24601), this skill enables processing of arbitrarily long contexts by treating them as variables in a REPL environment rather than feeding them directly into the context window.

## When to Use This Skill

**Auto-detect and propose RLM when:**
- Processing files > 100K tokens (large codebases, long documents)
- User asks to analyze/aggregate information across many files
- Task requires examining nearly all parts of a large input
- Multi-document question answering
- Tasks where information density scales with input length

**Explicit triggers:**
- User invokes `/rlm`
- User mentions "process this large file" or similar
- Context would exceed practical limits

**Do NOT use for:**
- Simple single-file reads < 50K tokens
- Tasks where only a small needle needs finding (use grep/search instead)
- Real-time interactive queries

## How It Works

1. **Context as Environment**: Load the input into a Python REPL as a `context` variable
2. **Programmatic Access**: Write code to peek, filter, chunk, and transform the context
3. **LLM Sub-calls**: Use `llm_query(prompt, model)` for semantic analysis of chunks
4. **Iterative Refinement**: Continue until ready to provide `FINAL()` answer

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
            │                         │
            └────────────┬────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  FINAL(answer) or FINAL_VAR(variable_name)          │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### From a Directory (Most Common)

```bash
# Initialize from a codebase
python rlm_repl.py init-dir ~/project --glob "**/*.py" --session myproj

# Execute with LLM support
python rlm_repl.py exec myproj "
import re
funcs = re.findall(r'def (\w+)', context)
print(f'Found {len(funcs)} functions')

# Use LLM to analyze
summary = llm_query(f'Summarize these function names: {funcs[:20]}', model='haiku')
print(summary)
" --llm
```

### From a Single File

```bash
python rlm_repl.py init ~/large_document.txt --session doc1
python rlm_repl.py exec doc1 "print(len(context))"
```

## CLI Reference

### Initialize from File

```bash
python rlm_repl.py init <file_path> [--session SESSION_ID]
```

### Initialize from Directory

```bash
python rlm_repl.py init-dir <directory> [--glob PATTERN] [--exclude PATTERNS] [--session SESSION_ID]
```

Options:
- `--glob`: File pattern (default: `**/*.py`)
- `--exclude`: Comma-separated exclusion patterns (e.g., `tests/*,*_test.py`)
- `--session`: Custom session ID

### Execute Code

```bash
python rlm_repl.py exec <session_id> "<code>" [--llm]
```

Options:
- `--llm`: Enable `llm_query()` function for API calls (requires ANTHROPIC_API_KEY)

### Other Commands

```bash
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

**Requires:** `ANTHROPIC_API_KEY` environment variable

## Core Patterns

### Pattern 1: Chunking and Aggregating

```python
# Split context into manageable chunks
chunks = context.split('\n\n')
chunk_size = len(chunks) // 10

results = []
for i in range(10):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < 9 else len(chunks)
    chunk = '\n\n'.join(chunks[start:end])

    answer = llm_query(f"Extract key information from: {chunk}", model="haiku")
    results.append(answer)
    print(f"Processed chunk {i+1}/10")

final = llm_query(f"Synthesize these findings: {results}", model="sonnet")
print(f"Final synthesis: {final}")
```

### Pattern 2: Filtered Search

```python
import re

# Use regex to find relevant sections first (cheap)
pattern = r'(def |class |async def )\w+'
matches = re.findall(pattern, context)
print(f"Found {len(matches)} function/class definitions")

# Extract surrounding context for each match
for match in matches[:5]:
    idx = context.find(match)
    snippet = context[max(0, idx-100):idx+500]
    analysis = llm_query(f"Analyze this code: {snippet}", model="haiku")
    print(f"{match}: {analysis}")
```

### Pattern 3: Hierarchical Decomposition

```python
import re

# Split by file markers (for directory init)
file_pattern = r'=== FILE: (.+?) ==='
files = re.split(file_pattern, context)

# Process each file
summaries = {}
for i in range(1, len(files), 2):
    filename = files[i]
    content = files[i+1] if i+1 < len(files) else ''
    if len(content.strip()) > 100:
        summary = llm_query(f"Summarize this file:\n{content[:5000]}", model="haiku")
        summaries[filename] = summary
        print(f"{filename}: {summary[:100]}...")

# Synthesize
final = llm_query(f"Create overview from these file summaries: {summaries}", model="sonnet")
```

## Best Practices

1. **Probe first**: Print small samples of context to understand structure before chunking
2. **Use code for filtering**: Regex and string operations are much cheaper than LLM calls
3. **Batch sub-calls**: Aim for ~10-50 LLM calls, not thousands
4. **Use appropriate models**: haiku for simple extraction, sonnet for synthesis
5. **Track progress**: Print status updates for long operations
6. **Check stats**: Use `get_llm_stats()` to monitor token usage

## Cost Awareness

- Each `llm_query()` call costs tokens
- Filter with code first, then use LLM for semantic understanding
- Use haiku (~$0.25/M input) for simple tasks, sonnet for complex synthesis
- Stats are printed at the end of each `--llm` execution

## Environment Setup

```bash
# Required for llm_query()
export ANTHROPIC_API_KEY=your-key

# Install anthropic package
pip install anthropic
```

## Workflow Example

```
1. User: "Analyze this codebase for security issues"

2. Claude: Initialize RLM session from directory
   > python rlm_repl.py init-dir ./backend --glob "**/*.py" --session sec1

3. Claude: Probe context structure
   > python rlm_repl.py exec sec1 "
   import re
   files = re.findall(r'=== FILE: (.+?) ===', context)
   print(f'{len(files)} files, {len(context)} chars')
   "

4. Claude: Find security-relevant code
   > python rlm_repl.py exec sec1 "
   import re
   auth_code = re.findall(r'(password|token|secret|auth|session).{0,500}', context, re.I)
   print(f'Found {len(auth_code)} security-relevant snippets')
   "

5. Claude: Analyze with LLM
   > python rlm_repl.py exec sec1 "
   # Batch analyze security snippets
   for i, snippet in enumerate(auth_code[:10]):
       analysis = llm_query(f'Check for security issues: {snippet}', model='haiku')
       print(f'Snippet {i+1}: {analysis}')
   " --llm

6. Claude: Synthesize findings
   > python rlm_repl.py exec sec1 "
   findings = llm_query('Summarize all security findings from the analysis above', model='sonnet')
   FINAL(findings)
   " --llm
```

## References

- [scripts/rlm_repl.py](scripts/rlm_repl.py) - Main REPL implementation
- [examples/document_analysis.py](examples/document_analysis.py) - Example workflow
- [RLM Paper (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601)
