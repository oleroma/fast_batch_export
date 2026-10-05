# Agent Instructions

## Important extension notes
- Headless blender instance and temp project copy is made to safely override geometry node inputs and keep UI unfrozen.

## Tool Discipline & Token Budget
- **Tool Whitelist**: Only use `read_file`, `edit_file`, `grep`, `list_directory`, and `web_search`.
- **Strictly Prohibited**: Never call `terminal`, `powershell`, or `spawn_subagent`. No syntax-check harnesses, mock tests, or throwaway scratch scripts.
- **Surgical Access**:
  - Read specific line windows (`start_line`/`end_line`) rather than full files.
  - Apply atomic, minimal-hunk edits via `edit_file` to keep diffs readable and token-efficient.

## Critical Judgment & Technical Pushback
- **Do Not Follow Blindly**: Proactively challenge requests that are redundant, degrade export/UI performance, introduce gratuitous complexity, or conflict with Blender 5.2 architecture.
- **Concisely State Trade-offs**: When pushing back, state the technical bottleneck (e.g., API constraints, undo stack corruption, event loop stalls) and recommend a simpler, superior alternative.
- **Block Impossible Implementations**: If a request violates Blender's C/Python boundary, data-block ownership model, or operator execution constraints, explain why it cannot work reliably and propose a viable alternative rather than shipping a brittle hack.
