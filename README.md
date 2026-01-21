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

RLM treats long contexts as **external environment variables** rather than stuffing them into the prompt. Claude writes Python code to programmatically access, filter, and chunk the context—then spawns sub-agents to process each chunk.

```
┌─────────────────────────────────────────────────────┐
│  User Query + Long Context                          │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  REPL Environment                                   │
│  ┌───────────────────────────────────────────────┐  │
│  │ context = "..." (loaded from file)            │  │
│  │ Write Python to filter/chunk                  │  │
│  │ Spawn sub-agents for semantic analysis        │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
    ┌──────────────┐          ┌──────────────┐
    │ Sub-agent 1  │          │ Sub-agent N  │
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
```

Or copy manually:
```bash
mkdir -p ~/.claude/skills/rlm
cp -r . ~/.claude/skills/rlm/
```

## Usage

### Slash Command

```
/rlm ~/path/to/large_file.txt "Find all API endpoints and their authentication methods"
```

### Auto-Detection

Claude will automatically propose RLM when:
- Processing files > 100K tokens
- Analyzing/aggregating across many files
- Tasks requiring examination of nearly all parts of a large input

### When NOT to Use

- Simple single-file reads < 50K tokens
- Needle-in-haystack searches (use grep/search instead)
- Real-time interactive queries

## How It Works

### 1. Initialize Session

```bash
python ~/.claude/skills/rlm/scripts/rlm_repl.py init document.txt --session doc1
```

### 2. Probe Context Structure

```bash
python rlm_repl.py exec doc1 "print(len(context)); print(context[:1000])"
```

### 3. Process with Code

```python
# Split into chunks
chunks = context.split('\n\n')
print(f"Found {len(chunks)} chunks")

# Filter relevant sections with regex
import re
api_sections = [c for c in chunks if re.search(r'def |@app\.', c)]
print(f"Found {len(api_sections)} API-related sections")
```

### 4. Sub-Agent Calls

Claude uses the Task tool to spawn sub-agents for semantic analysis:

```python
# Claude spawns sub-agents for each chunk that needs understanding
for chunk in api_sections:
    summary = llm_query(f"Analyze this API endpoint: {chunk}")
    results.append(summary)
```

### 5. Final Answer

```python
FINAL("Based on analysis, there are 15 API endpoints...")
# or return a variable
FINAL_VAR(results_list)
```

## Core Patterns

### Chunking and Aggregating
Split → Process each → Synthesize results

### Filtered Search
Use regex/string ops to narrow down → LLM for semantic understanding

### Hierarchical Decomposition
Split by structure (headers, functions) → Summarize each → Overview

See [examples/document_analysis.py](examples/document_analysis.py) for a complete workflow.

## CLI Reference

```bash
# Initialize session with context file
python rlm_repl.py init <context_file> [--session SESSION_ID]

# Execute Python code in session
python rlm_repl.py exec <session_id> "<code>"

# Store sub-agent result as variable
python rlm_repl.py store <session_id> <var_name> "<value>"

# Show session info
python rlm_repl.py info <session_id>

# List all active sessions
python rlm_repl.py list

# Clean up session
python rlm_repl.py cleanup <session_id>
```

## Cost Awareness

- Each `llm_query()` spawns a sub-agent (costs tokens)
- Use code for filtering first, LLM for semantic understanding
- Batch items: 50-200 per sub-call when possible
- Aim for ~10-50 sub-calls, not thousands

## Requirements

- Claude Code with skills enabled
- Python 3.8+
- No external dependencies (uses stdlib only)

## License

MIT

## References

- [RLM Paper (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601)
- [Claude Code Skills Documentation](https://docs.anthropic.com/en/docs/claude-code/skills)
