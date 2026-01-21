"""
Example RLM workflow for analyzing a large document.

This example shows how to use the RLM REPL to process a large document
by chunking it and using sub-agent calls for semantic analysis.
"""

# Example code that would be executed in the RLM REPL
# This is what Claude would generate when using the RLM skill

# ============================================================
# Step 1: Probe the context to understand its structure
# ============================================================

print(f"Context length: {len(context)} characters")
print(f"First 500 chars:\n{context[:500]}")
print(f"\nLast 500 chars:\n{context[-500:]}")

# Count lines
lines = context.split('\n')
print(f"\nTotal lines: {len(lines)}")

# ============================================================
# Step 2: Identify structure (e.g., markdown headers)
# ============================================================

import re

headers = re.findall(r'^#{1,3} .+$', context, re.MULTILINE)
print(f"\nFound {len(headers)} headers:")
for h in headers[:10]:
    print(f"  - {h}")
if len(headers) > 10:
    print(f"  ... and {len(headers) - 10} more")

# ============================================================
# Step 3: Chunk and process with sub-agents
# ============================================================

# Split by double newlines (paragraphs)
chunks = [c for c in context.split('\n\n') if len(c.strip()) > 50]
print(f"\nProcessing {len(chunks)} chunks...")

# Batch chunks to reduce API calls
batch_size = 20
summaries = []

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    batch_text = '\n---\n'.join(batch)

    # Use sub-agent to summarize batch
    summary = llm_query(
        f"Summarize the key points from these text sections:\n\n{batch_text}",
        model="haiku"
    )
    summaries.append(summary)
    print(f"Batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}: {summary[:100]}...")

# ============================================================
# Step 4: Synthesize final answer
# ============================================================

all_summaries = '\n\n'.join(summaries)
final_answer = llm_query(
    f"Based on these section summaries, provide a comprehensive overview:\n\n{all_summaries}",
    model="sonnet"
)

print(f"\n{'='*50}")
print("FINAL SYNTHESIS:")
print(final_answer)

# Return the final answer
FINAL(final_answer)
