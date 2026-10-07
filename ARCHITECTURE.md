# Architecture Overview: Fast Batch STL Exporter

`Fast Batch STL Exporter` (`fast_batch_stl_exporter_dev`, v7.0.0) is a high-performance, parametric batch export add-on for Blender 5.2+. It combines vectorized binary STL generation, multi-tier geometry node / modifier socket overrides, combinatorial parametric sweeping, and a dual-path execution engine (in-process synchronous export vs. isolated headless subprocess export).

---

## 1. Design Principles & Goals

1. **Non-destructive & Isolated**: Overriding geometry nodes or modifier inputs must never corrupt the active project file, viewport state, or undo history.
2. **Dual-Path Execution Model**:
   - **Direct Fast-Path**: Mesh objects without parameter permutations or overrides evaluate natively in the active scene using Blender's evaluated depsgraph and stream directly to disk.
   - **Headless Worker Process**: When overrides or sweeps are active, the add-on forks an independent background Blender process (`--factory-startup -b <temp.blend> -P <script> -- --batch-stl-headless <job.json>`) on a temporary copy of the project file. The main Blender UI stays completely responsive and interactive.
3. **Vectorized NumPy Disk I/O**: Direct memory extraction via `foreach_get` into contiguous NumPy structured arrays avoiding per-polygon Python iterations.
4. **Predictive UI & Safety**: Cached calculation (~10Hz) of directory structures and STL filenames with collapsible nested box visualization and automatic collision detection before executing exports.

---

## 2. System Architecture Diagram

```
+---------------------------------------------------------------------------------+
|                                Blender Main UI                                  |
|                                                                                 |
|  [ VIEW3D Panels ] ──> UI Cache Engine (10Hz Timer) ──> Directory Tree / Clashes |
|         │                                                                       |
|         ▼                                                                       |
|  [ Operator: EXPORT_OT_batch_stl_multi ]                                        |
+─────────┬───────────────────────────────────────┬───────────────────────────────+
          │                                       │
          │ (No Overrides / Sweeps)               │ (Overrides / Sweeps Present)
          ▼                                       ▼
+───────────────────────────+         +───────────────────────────────────────────+
|   Native Direct Export    |         |        Headless Worker Spawn Pipeline     |
|                           |         |                                           |
|  • Evaluated depsgraph    |         |  1. Save copy: wm.save_as_mainfile(copy)  |
|  • write_object_stl       |         |  2. Write job manifest (job.json)         |
|  • Main thread sync       |         |  3. subprocess.Popen(blender -b ...)      |
+───────────────────────────+         |  4. Modal handler / thread stdout queue   |
                                      +─────────────────────┬─────────────────────+
                                                            │
                                                            ▼
                                      +───────────────────────────────────────────+
                                      |         Headless Subprocess Engine        |
                                      |                                           |
                                      |  • Group objects by override signature    |
                                      |  • Dynamic depsgraph culling (lc.exclude) |
                                      |  • Itertools combinatorial sweep          |
                                      |  • Apply overrides -> depsgraph -> mesh   |
                                      |  • write_object_stl                       |
                                      |  • Revert baseline states                 |
                                      +───────────────────────────────────────────+
```

---

## 3. Data Model & Hierarchy

The add-on structures export configurations in a strictly scoped 8-tier hierarchy:

`Global` → `Preset` → `Collection` → `Object` → `NodeGroup` → `Node` → `Input Socket` → `Value / Sweep`

### Property Groups (`fast_batch_stl_export/__init__.py`)

1. **`BatchSTLExportPreset`**:
   - Holds preset name, root directory prefix (`preset_prefix`), progress indicators, execution status, and isolated execution logs (`console_logs`).
   - Owns a collection of `BatchSTLCollection` mappings and preset-level `BatchSTLNodeGroup` overrides.
2. **`BatchSTLCollection`**:
   - Maps a Blender `bpy.data.collections` entry.
   - Holds subfolder paths (`sub_path`), tagging flags (`use_tag`, `tag`), collection-level pinned overrides (`nodegroups`), and synchronized object entries (`objects`).
3. **`BatchSTLObject`**:
   - Mirrors individual meshes/curves within a collection.
   - Provides per-object export toggling (`export`), filename tag (`tag`), object subfolder (`sub_path`), and object-level overrides (`nodegroups`).
4. **`BatchSTLNodeGroup`**:
   - Targets a Geometry Node tree by name (`group_name`).
   - Contains a list of `BatchSTLNode` blocks.
5. **`BatchSTLNode`**:
   - Targets an internal node name within the node group, or `"<Modifier Interface>"` to target the modifier interface sockets directly.
6. **`BatchSTLInput`**:
   - Targets an input socket (`name`) and its inferred data type (`override_type`: `FLOAT`, `INT`, `BOOLEAN`, `STRING`, `MENU`).
7. **`BatchSTLValue`**:
   - Concrete parameter value or sweep definition (`use_sweep`, `sweep_start_*`, `sweep_step_*`, `sweep_count_*`, `sweep_range`).
   - Configures naming tags (`use_tag`, `tag`) and subfolder routing (`use_dir`).
8. **`BatchSTLLogLine`**:
   - Discrete line items stored per-preset for live output display.

### Scoping & Inheritance of Overrides

When generating variations for an object, overrides are gathered in hierarchical order:
1. `Global` (`scene.batch_stl_global_nodegroups`)
2. `Preset` (`preset.nodegroups`)
3. `Collection` (`collection.nodegroups`)
4. `Object` (`object.nodegroups`)

`resolve_overrides` then applies **most-specific-wins**: if a lower level defines the same socket (same target, node group, node and input name), the inherited values of higher levels for that socket are dropped. Several values on the same level remain variants. Objects are batched for the headless worker by `get_override_signature`, a fingerprint of the resolved overrides (values, sweeps, tags, folder flags and level).

---

## 4. Execution Pipelines

### A. Fast Direct Path (Native In-Process)
- **Condition**: Preset contains zero overrides across all hierarchy tiers.
- **Mechanism**:
  - Resolves active objects across included collections.
  - Computes evaluated geometry using `obj.evaluated_get(depsgraph).to_mesh()`, plus unrealized instances found through `depsgraph.object_instances` (`collect_instance_arrays`).
  - Streams geometry to disk using `write_object_stl`.
  - Cleans up evaluated mesh data via `to_mesh_clear()`.

### B. Headless Worker Pipeline (Process Isolation)
- **Condition**: Any override or sweep is defined in Global, Preset, Collection, or Object scope.
- **Workflow**:
  1. **Preparation**:
     - Saves a snapshot of current memory to a temporary file via `bpy.ops.wm.save_as_mainfile(filepath=..., copy=True)`.
     - Writes a job manifest `job.json` containing preset index, root directory path, and flags.
  2. **Process Spawn**:
     - Spawns background Blender: `[blender, "--factory-startup", "-b", temp_blend, "-P", script_file, "--", "--batch-stl-headless", job_json]`.
  3. **IPC & Streaming**:
     - A background reader thread drains worker `stdout` into a thread-safe `queue.Queue`.
     - A Blender modal timer operator (`0.05s`) polls the queue, updates preset progress bars (`BATCH_STL_PROGRESS:X`), and logs stdout into the preset's console UI.
  4. **Headless Execution Engine (`run_headless_export`)**:
     - **Phase 0 (Depsgraph Culling)**: Groups objects by override signature (`execution_batches`). Excludes unrelated layer collections (`lc.exclude = True`) to prevent Blender from evaluating unneeded geometry trees during node updates.
     - **Baseline Capture (`capture_baseline_states`)**: Caches baseline modifier socket values, internal node socket defaults, and node link connections.
     - **Combinatorial Cartesian Product (`generate_override_combinations`)**: Evaluates `itertools.product` across all parameter pools and sweeps.
     - **Evaluation Loop**:
       - Applies overrides via `apply_overrides`.
       - Calls `bpy.context.view_layer.update()` and fetches updated depsgraph.
       - Writes STL files via `write_object_stl`.
     - **Restoration (`revert_overrides`)**: Re-applies baseline values and links, restores layer collection exclusion states.
  5. **Teardown**:
     - Worker exits with code 0.
     - Main process modal handler catches `BATCH_STL_DONE`, stops modal timer, and removes temporary directory.

---

## 5. High-Performance Vectorized STL Writer

Instead of creating intermediate text or using standard single-threaded Python file writers, `mesh_to_stl_array` / `write_object_stl` implement direct binary packing:

1. **Triangulation**: Calls `mesh.calc_loop_triangles()` to ensure valid facet indices.
2. **Memory Extraction**:
   - `mesh.vertices.foreach_get("co", verts.ravel())` reads vertex coordinates directly into NumPy buffers.
   - `mesh.loop_triangles.foreach_get("vertices", tri_verts.ravel())` reads face indices.
   - `mesh.loop_triangles.foreach_get("normal", tri_normals.ravel())` reads precomputed loop normals.
3. **Matrix Transformations**:
   - Vertex coordinates are transformed by `matrix_world` using vectorized NumPy SIMD operations (`V' = V @ M.T + T`).
   - Normal vectors are transformed using the inverse matrix and re-normalized.
   - Automatically handles negative determinant matrices (flipped winding order) by swapping vertices $v_1$ and $v_2$.
4. **Structured Binary Array**:
   - Formatted using structured NumPy dtype:
     ```python
     STL_DTYPE = np.dtype([
         ('normals', np.float32, (3,)),
         ('v0', np.float32, (3,)),
         ('v1', np.float32, (3,)),
         ('v2', np.float32, (3,)),
         ('attr', np.uint16)
     ])
     ```
   - Written to disk in a single continuous binary block using `STL_HEADER + struct.pack('<I', num_tris) + data.tobytes()`.

---

## 6. UI Caching & Safety Subsystems

### UI Cache Engine (`rebuild_ui_cache_if_dirty`)
- Driven by a background timer (`bpy.app.timers`) running at ~10Hz with a dirty flag (`mark_dirty()`).
- Recomputes statistics (preset counts, collections, exported object count, total permutation iterations).
- Computes directory hierarchies and leaf files in advance.
- **Naming Collision Detection**: Analyzes all destination paths and flags collisions when two permutations or objects resolve to the identical output file path.

### Undo Stack Protection
- Property edits made in the UI get Blender's native undo step; property `update` callbacks only call `mark_dirty()`. The list/table operators declare `'UNDO'` in `bl_options`. Internal syncs (`sync_collection_objects`, type inference) happen outside the UI edit path and push nothing.
- Runtime export state (`is_exporting`, progress, status, cancel flag, console log) lives in `WindowManager.batch_stl_jobs` (`BatchSTLJob`, one entry per preset index), not on the Scene. It is therefore never saved in the `.blend` and never rolled back by undo; `load_post` clears it. Entries are created by operators (`get_job(..., create=True)`); draw code only reads them.
- Jobs are keyed by preset index, so removing/reordering presets is refused while any export runs. Pressing undo during an export cancels it for the same reason.
- The export modal never keeps an RNA pointer to a preset or job; it re-resolves them by index on every event because undo, file loads and collection growth invalidate pointers.

---

## 7. Configuration Portability

The add-on implements full JSON schema serialization and deserialization (`BATCH_STL_OT_export_presets_json` / `BATCH_STL_OT_import_presets_json`):
- Serializes presets, collections, object lists, exclusion states, node group overrides, input types, values, sweeps, and tagging configurations into clean, version-agnostic JSON files.
- Provides deep-copy and paste support across presets, collections, and node groups via internal clipboard buffers (`_clipboard`).
