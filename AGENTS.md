# Agent Instructions

## 1. Architecture & Mental Model
- **Extension Scope**: Blender 5.2+ add-on (`fast_batch_stl_export`) for batch-exporting collections with Geometry Nodes overrides.
- **Key Components**:
  - `__init__.py`: Contains UI lists, property groups (`BatchSTLPreset`, `BatchSTLCollection`, `BatchSTLNode`, `BatchSTLInput`), socket resolution, and export operators.
  - `blender_manifest.toml`: Extension packaging and version contract (`blender_version_min = "5.2.0"`).

## 2. Tool Discipline & Token Budget
- **Tool Whitelist**: Only use `read_file`, `edit_file`, `grep`, `list_directory`, and `web_search`.
- **Strictly Prohibited**: Never call `terminal`, `powershell`, or `spawn_subagent`. No syntax-check harnesses, mock tests, or throwaway scratch scripts.
- **Surgical Access**:
  - Read specific line windows (`start_line`/`end_line`) rather than full files.
  - Apply atomic, minimal-hunk edits via `edit_file` to keep diffs readable and token-efficient.

## 3. Blender 5.2 Standards & API Hygiene
- **Research First**: Consult official Blender Python API documentation via `web_search` for signature or deprecation checks before writing code.
- **Modern APIs**:
  - **Node Tree Sockets**: Use Blender 4.0+/5.x `node_tree.interface` items (never legacy `node_tree.inputs`/`outputs`).
  - **STL Exporter**: Use the modern C++ operator `bpy.ops.wm.stl_export()`.
  - **Context Handling**: Use `context.temp_override(...)` for context-sensitive operators.
- **Undo Management**: Wrap batch property updates, list re-indexing, and clipboard operations in `with suppress_undo():` to avoid polluting Blender's undo stack.

## 4. Interaction & Output Standards
- **Direct & Action-First**: Deliver final, production-ready code with concise explanations.
- **No Filler**: Skip conversational meta-chatter, repetitive apologies, or reciting raw tool output.

## 5. Gemini Optimizations
- **Zero Preambles**: Omit conversational openers, pleasantries, and transitional filler; start immediately with the action or concise technical summary.
- **Coherent Tool Execution**: Resolve tasks in a single deliberate sequence (targeted read -> doc check if needed -> surgical edit) without breaking into unnecessary turns or fragmented tool calls.
- **Exact String Anchoring**: In `edit_file`, capture sufficient surrounding context in `old_string` to guarantee an unambiguous, first-attempt match against Python indentation and avoid whitespace drift.
- **Grounded In-Context Attention**: Confine attention strictly to targeted components and active diffs; avoid speculative analysis, unverified API guesses, or unrequested refactoring of unrelated logic.

## 6. Critical Judgment & Technical Pushback
- **Do Not Follow Blindly**: Proactively challenge requests that are redundant, degrade export/UI performance, introduce gratuitous complexity, or conflict with Blender 5.2 architecture.
- **Concisely State Trade-offs**: When pushing back, state the technical bottleneck (e.g., API constraints, undo stack corruption, event loop stalls) and recommend a simpler, superior alternative.
- **Block Impossible Implementations**: If a request violates Blender's C/Python boundary, data-block ownership model, or operator execution constraints, explain why it cannot work reliably and propose a viable alternative rather than shipping a brittle hack.




