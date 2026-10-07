# Fast Batch STL Exporter

A high-performance batch export pipeline and parametric permutation engine for Blender 5.2+. Built around a custom vectorized NumPy binary STL generator and an adaptive dual-path execution engine, it enables one-click exporting of scene collections, multi-dimensional parameter sweeping via Geometry Nodes and modifiers, and automated variant generation.

---

## Architecture & Technical Overview

For in-depth architectural details, execution flowcharts, and engine design, see [ARCHITECTURE.md](ARCHITECTURE.md).

* **Blender Version Support:** Blender 5.2.0+ (Manifest schema v1.0.0, extension version 7.0.0).
* **Package Format:** Blender 5.2 Extension (`blender_manifest.toml`).
* **Design Philosophy:** Non-destructive execution, isolated subprocess execution for mutations, and zero undo-stack pollution.

---

## Core Features

### 1. Vectorized Binary STL Generator
* **NumPy Direct Memory Buffering:** Reads vertex coordinates, face indices, and loop normals directly into continuous memory buffers via `foreach_get`.
* **Vectorized Transformations:** Transforms vertex positions and face normals using SIMD matrix arithmetic in NumPy, accounting for world matrices and negative determinant winding flips.
* **Instances Included:** Unrealized Geometry Nodes instances are exported together with the object's own mesh.
* **Instant Disk Streaming:** Meshes with hundreds of thousands of triangles are triangulated, evaluated, and streamed to disk in milliseconds.

### 2. Adaptive Dual-Path Execution Model
* **Synchronous Bypass (Fast Path):** Exports without node overrides or parameter sweeps are evaluated natively in the current Blender process and written directly to disk.
* **Headless Background Worker (Safe Isolation Path):** When geometry node overrides, modifier changes, or parameter sweeps are present:
  - Automatically creates a temporary copy of the `.blend` file.
  - Spawns an isolated background Blender worker (`--factory-startup -b <temp.blend> -P ...`).
  - Performs intelligent dependency graph culling (`lc.exclude`) so unneeded collections are ignored during geometry updates.
  - Restores baseline socket connections and values after export.
  - Keeps the main Blender viewport and UI completely responsive and interactive.

### 3. Hierarchical Parameter Overrides (8 Tiers)
Define temporary parameter overrides and sweeps across an 8-tier hierarchy:
`Global` → `Preset` → `Collection` → `Object` → `NodeGroup` → `Node` → `Input Socket` → `Value / Sweep`

* **Override Semantics:** A more specific level replaces the inherited value of the same socket (an Object value replaces a Collection, Preset or Global value). Several values on the *same* level are variants and are all exported.
* **Target Flexibility:** Target either exposed modifier interface sockets (`<Modifier Interface>`) or specific internal nodes within a Geometry Node tree.
* **Type Auto-Detection:** Automatically inspects the node tree interface and infers socket data types (`FLOAT`, `INT`, `BOOLEAN`, `STRING`, `MENU`). Other socket types (vectors, colors, objects, ...) are flagged as unsupported and block the export until removed.
* **Menu/Enum Auto-Populate:** Searches and lists available items for Menu Switch nodes.

### 4. Parametric Sweeping & Combinatorial Engine
Generate variant permutations across any socket:
* **Float / Int Ranges:** Define numeric sweeps via explicit start, step, and step-count controls, or string ranges.
* **Boolean & Menu Combinations:** Automatically iterates through `True`/`False` states or all enum options.
* **Cartesian Product Generator:** Calculates multi-dimensional permutation matrices using `itertools.product`, ensuring all parameter combinations are generated systematically.

### 5. Predictive Directory Tree & Overwrite Protection
* **Interactive Nested UI Hierarchy:** Calculates permutations ahead of time and displays an interactive directory hierarchy using collapsible nested UI boxes and indentation (`batch_stl.toggle_dir_tree`).
* **Pre-Export Clash Detection:** Instantly flags duplicate output file paths with visual alerts before export starts, preventing accidental file overwrites.
* **Decoupled UI Cache:** Background timer cache (~10Hz) prevents UI stalls when evaluating large permutation matrices.

### 6. Scoped Live Console & Progress Tracking
* **Preset-Isolated Logging:** Each export preset tracks its own console log and export duration.
* **Real-Time Progress Streaming:** Non-blocking background worker output is piped directly into the Blender panel with operation counters and elapsed time display.
* **Auto-View Switching:** Displays the directory tree during setup, flips to the live console on export start, and allows instant cancellation.

### 7. Collection Mapping & Granular Exclusion Filters
* **Collection Bindings:** Map multiple collections per preset, configure custom sub-folder destinations, and append collection tags.
* **Object-Level Filtering:** Enable or disable specific mesh objects within collections without affecting viewport visibility.
* **Per-Object Overrides:** Assign distinct tags, sub-folders, and dedicated node override groups down to individual objects.

### 8. Dynamic Tagging & Directory Formatting
* **Sub-Directory Creation (`FILE_FOLDER`):** Route variant exports into dedicated sub-folders per value iteration.
* **Tagging Rules (`BOOKMARKS`):**
  - `tag`: Replaces the socket value label entirely (`tag`).
  - `tag_`: Prepends the tag to the value (`tag_15`).
  - `_tag`: Appends the tag to the value (`15_tag`).
  - Blank: Defaults to the formatted parameter value.

### 9. JSON Preset Portability & Clipboard Buffer
* **Import / Export Setup:** Save or restore presets, collections, object lists, exclusion states, and override matrices to external JSON files.
* **Internal Clipboard:** Copy and paste presets, collections, and node groups between tiers with one click.

---

## Installation & Requirements

* **Blender:** 5.2.0 or newer.
* **Dependencies:** Standard Blender Python environment (`numpy` is included with Blender).
* **Installation:** Install as an extension from the Blender Preferences extensions menu or place `fast_batch_stl_export` into your Blender extensions directory.

