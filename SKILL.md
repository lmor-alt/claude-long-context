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
3. **Recursive Sub-calls**: Use Task tool to spawn sub-agents on context chunks
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
│  │ context = "..." (loaded from file)            │  │
│  │ llm_query(prompt) → sub-agent call            │  │
│  │ print() → observe results                     │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
    ┌──────────────┐          ┌──────────────┐
    │ Sub-agent 1  │          │ Sub-agent N  │
    │ (chunk 1)    │   ...    │ (chunk N)    │
    └──────────────┘          └──────────────┘
            │                         │
            └────────────┬────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  FINAL(answer) or FINAL_VAR(variable_name)          │
└─────────────────────────────────────────────────────┘
```

## Usage

### Slash Command

```
/rlm <file_path> <query>
```

Example:
```
/rlm ~/documents/large_codebase.txt "Find all API endpoints and their authentication methods"
```

### Automatic Proposal

When Claude detects a task that would benefit from RLM, it will propose:

> This task involves processing a large context (~500K tokens). I recommend using the RLM approach to handle this effectively. Should I proceed with `/rlm`?

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

    answer = llm_query(f"Extract key information from: {chunk}")
    results.append(answer)
    print(f"Processed chunk {i+1}/10: {answer[:100]}...")

final = llm_query(f"Synthesize these findings: {results}")
print(f"Final synthesis: {final}")
```

### Pattern 2: Filtered Search

```python
import re

# Use regex to find relevant sections first
pattern = r'(def |class |async def )\w+'
matches = re.findall(pattern, context)
print(f"Found {len(matches)} function/class definitions")

# Extract surrounding context for each match
for match in matches[:5]:
    idx = context.find(match)
    snippet = context[max(0, idx-100):idx+500]
    analysis = llm_query(f"Analyze this code: {snippet}")
    print(f"{match}: {analysis}")
```

### Pattern 3: Semantic Transformation

```python
# When each line needs semantic understanding
lines = context.strip().split('\n')
transformed = []

for i, line in enumerate(lines):
    if i % 100 == 0:
        print(f"Processing line {i}/{len(lines)}")

    # Batch lines to reduce sub-calls
    if i % 50 == 0:
        batch = lines[i:i+50]
        batch_result = llm_query(f"Classify each line: {batch}")
        transformed.extend(batch_result.split('\n'))

print(f"Transformed {len(transformed)} lines")
```

### Pattern 4: Hierarchical Decomposition

```python
# For structured documents (markdown, code files)
import re

# Split by headers
sections = re.split(r'^#{1,3} ', context, flags=re.MULTILINE)
print(f"Found {len(sections)} sections")

summaries = {}
for i, section in enumerate(sections):
    if len(section.strip()) > 100:
        title = section.split('\n')[0][:50]
        summary = llm_query(f"Summarize this section: {section[:5000]}")
        summaries[title] = summary
        print(f"Section '{title}': {summary[:100]}...")

final = llm_query(f"Create overview from: {summaries}")
```

## Output Formats

When finished processing, use one of:

```python
# Direct answer
FINAL("The answer is X based on analysis of Y")

# Return a variable (for long outputs)
FINAL_VAR(results_list)
```

## Best Practices

1. **Probe first**: Print small samples of context to understand structure before chunking
2. **Batch sub-calls**: Group related queries to minimize API calls
3. **Track progress**: Print status updates for long operations
4. **Use code for filtering**: Regex and string operations are cheaper than LLM calls
5. **Verify answers**: Use sub-calls to cross-check critical findings

## Cost Awareness

- Each `llm_query()` call invokes a sub-agent (costs tokens)
- Filter with code first, then use LLM for semantic understanding
- Aim for ~10-50 sub-calls for most tasks, not thousands
- Batch 50-200 items per sub-call when possible

## Implementation

### Step 1: Initialize Session

```bash
python ~/.claude/skills/rlm/scripts/rlm_repl.py init <context_file> --session <id>
```

### Step 2: Execute Code

```bash
python ~/.claude/skills/rlm/scripts/rlm_repl.py exec <session_id> "<python_code>"
```

### Step 3: For Sub-agent Calls

When semantic analysis is needed on a chunk, Claude should:
1. Extract the chunk to a temp file or variable
2. Use the Task tool with `subagent_type=Explore` or `subagent_type=general-purpose`
3. Store the result: `python rlm_repl.py store <session_id> <var_name> "<result>"`

### Step 4: Cleanup

```bash
python ~/.claude/skills/rlm/scripts/rlm_repl.py cleanup <session_id>
```

## Workflow Example

```
1. User: "Analyze this 500K token document"
2. Claude: Initialize RLM session
   > python rlm_repl.py init document.txt --session doc1

3. Claude: Probe context structure
   > python rlm_repl.py exec doc1 "print(len(context)); print(context[:1000])"

4. Claude: Extract and process chunks
   > python rlm_repl.py exec doc1 "chunks = context.split('\\n\\n'); print(f'{len(chunks)} chunks')"

5. Claude: For each chunk needing semantic analysis, use Task tool
   > Task(subagent_type="general-purpose", prompt="Summarize: <chunk_content>")

6. Claude: Store sub-agent results
   > python rlm_repl.py store doc1 summary1 "The chunk discusses..."

7. Claude: Aggregate and return final answer
   > python rlm_repl.py exec doc1 "print(summary1, summary2, ...)"
```

## References

- [scripts/rlm_repl.py](scripts/rlm_repl.py) - Main REPL implementation
- [examples/document_analysis.py](examples/document_analysis.py) - Example workflow
