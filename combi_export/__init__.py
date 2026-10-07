"""
Fast Batch STL Exporter
Architecture: Single-File Monolithic (Optimized for Agentic Environments)
Data Hierarchy: Global > Preset > Collection > Object > NodeGroup > Node > Input > Value
"""

# ==============================================================================
# === MODULE IMPORTS ===
# ==============================================================================
import os
import json
import time
import itertools
import functools
import math
import re
import threading
import queue
import subprocess
import tempfile
import sys
import struct
import shutil
import traceback
import numpy as np

import bpy
from bpy_extras.io_utils import ExportHelper, ImportHelper
from bpy.app.handlers import persistent

# ==============================================================================
# === [ 1. GLOBALS & STATE ] ===
# ==============================================================================

ICONS = {
    'PRESET': 'PRESET', 'COLLECTION': 'OUTLINER_COLLECTION', 'OBJECT': 'OBJECT_DATA',
    'SWEEP': 'CON_ROTLIMIT', 'GLOBAL': 'WORLD', 'DIR': 'FILE_FOLDER', 'TAG': 'BOOKMARKS',
    'ADD': 'ADD', 'DEL': 'TRASH', 'UP': 'TRIA_UP', 'DOWN': 'TRIA_DOWN',
    'COPY': 'COPYDOWN', 'PASTE': 'PASTEDOWN', 'CANCEL': 'CANCEL', 'EXPORT': 'EXPORT',
    'IMPORT': 'IMPORT', 'INFO': 'INFO', 'CONSOLE': 'CONSOLE', 'CHECK_ON': 'CHECKBOX_HLT',
    'CHECK_OFF': 'CHECKBOX_DEHLT', 'OVR': 'DECORATE_OVERRIDE', 'NODE': 'NODETREE',
    'TIME': 'TIME', 'MODIFIER': 'MODIFIER', 'TREE': 'OUTLINER_OB_EMPTY', 'ERROR': 'ERROR',
    'RIGHT': 'TRIA_RIGHT', 'BLANK': 'BLANK1', 'FILE': 'FILE_3D'
}

SUPPORTED_OBJECT_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT"}

STL_DTYPE = np.dtype([
    ('normals', np.float32, (3,)), ('v0', np.float32, (3,)),
    ('v1', np.float32, (3,)), ('v2', np.float32, (3,)), ('attr', np.uint16)
])

STL_HEADER = b'Batch STL Fast Export' + b'\x00' * 59

_clipboard = {"preset": None, "collection": None, "nodegroup": None}

_ui_cache = {
    "is_dirty": True,
    "visibility": {},
    "stats": {"global": {"presets": 0, "cols": 0, "objs": 0, "exp": 0}, "presets": {}, "cols": {}},
    "tree": ({}, set()),
    "preset_metrics": {},
}

def mark_dirty(self=None, context=None):
    _ui_cache["is_dirty"] = True

# Blender gives every UI-edited property an undo step except search fields (prop_search / StringProperty(search=...)):
# those buttons are created without UI_BUT_UNDO, so picking an item from the dropdown leaves no history entry.
# Their update callbacks push the step themselves. Operators record their own step ('UNDO' in bl_options), so
# property writes made while an operator runs must not push a second one.
_operator_depth = 0

def push_search_undo(label):
    if _operator_depth: return
    try: bpy.ops.ed.undo_push(message=label)
    except RuntimeError: pass

def search_field_update(label):
    """Update callback for a search field: refresh the UI cache and record an undo step."""
    def update(self, context):
        mark_dirty()
        push_search_undo(label)
    return update

def inside_operator(execute):
    """Decorator for operator execute(): property updates fired by the operator itself do not push undo steps."""
    @functools.wraps(execute)
    def wrapper(self, context):
        global _operator_depth
        _operator_depth += 1
        try: return execute(self, context)
        finally: _operator_depth -= 1
    return wrapper

_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

def sanitize_name(text):
    """Make text safe to use inside a single file or folder name on every OS."""
    return _INVALID_NAME_CHARS.sub("_", str(text))

def split_path_parts(path_text):
    """Split a user-typed sub-path into safe folder names; empty, '.' and '..' parts are dropped."""
    if not path_text: return []
    parts = (sanitize_name(p).strip(" .") for p in re.split(r"[\\/]", path_text))
    return [p for p in parts if p]

def clash_key(path):
    """Key for output-path collision checks; Windows and macOS file systems are case-insensitive by default."""
    norm = os.path.normpath(path)
    return norm.casefold() if sys.platform in ("win32", "darwin") else norm

def redraw_sidebars(context=None):
    """Redraw only the 3D viewport sidebars (where this add-on lives), not the whole viewports."""
    wm = getattr(context or bpy.context, "window_manager", None)
    if not wm: return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'UI': region.tag_redraw()

# ==============================================================================
# === [ 2. CORE LOGIC & ENGINE ] ===
# ==============================================================================

def clean_node_name(name):
    if not name: return ""
    return name.split(" [")[0].strip()

def is_collection_excluded(context, target_collection):
    if not target_collection: return True
    view_layer = getattr(context, "view_layer", None)
    if not view_layer: return False
    found_any_visible = False
    found_any = False

    def traverse(layer_collection, parent_excluded=False):
        nonlocal found_any_visible, found_any
        if found_any_visible: return
        current_excluded = parent_excluded or layer_collection.exclude
        if layer_collection.collection == target_collection:
            found_any = True
            if not current_excluded:
                found_any_visible = True
                return
        for child in layer_collection.children:
            traverse(child, current_excluded)
            if found_any_visible: return

    traverse(view_layer.layer_collection)
    if not found_any: return True
    return not found_any_visible

def get_modifier_socket_identifier(node_group, socket_name):
    if not node_group: return None
    if hasattr(node_group, "interface"):
        for item in node_group.interface.items_tree:
            if getattr(item, "item_type", "") == 'SOCKET' and getattr(item, "in_out", "INPUT") == 'INPUT' and item.name == socket_name:
                return item.identifier
    elif hasattr(node_group, "inputs"):
        for inp in node_group.inputs:
            if inp.name == socket_name: return inp.identifier
    return None

def get_modifier_socket_default(node_group, socket_name):
    if not node_group: return None
    if hasattr(node_group, "interface"):
        for item in node_group.interface.items_tree:
            if getattr(item, "item_type", "") == 'SOCKET' and getattr(item, "in_out", "INPUT") == 'INPUT' and item.name == socket_name:
                return getattr(item, "default_value", None)
    elif hasattr(node_group, "inputs"):
        for inp in node_group.inputs:
            if inp.name == socket_name: return getattr(inp, "default_value", None)
    return None

def get_menu_switch_items(node_group, node_name, input_name):
    if not node_group: return []
    is_mod = not node_name or node_name == "<Modifier Interface>"
    if is_mod and hasattr(node_group, "interface"):
        for item in node_group.interface.items_tree:
            if getattr(item, "item_type", "") == 'SOCKET' and item.name == input_name and hasattr(item, "enum_items"):
                return [getattr(ei, 'identifier', getattr(ei, 'name', '')) for ei in item.enum_items]

    nodes_to_search = node_group.nodes if is_mod else []
    if not nodes_to_search and node_name:
        target_n = node_group.nodes.get(clean_node_name(node_name))
        if target_n:
            if target_n.type == 'MENU_SWITCH' and hasattr(target_n, 'enum_items'):
                return [getattr(item, 'identifier', getattr(item, 'name', '')) for item in target_n.enum_items]
            elif target_n.type == 'GROUP' and hasattr(target_n, 'node_tree') and target_n.node_tree:
                nodes_to_search = target_n.node_tree.nodes

    for node in nodes_to_search:
        if node.type == 'MENU_SWITCH' and hasattr(node, 'enum_items'):
            for sock in node.inputs:
                for link in sock.links:
                    if link.from_node.type == 'GROUP_INPUT' and link.from_socket.name == input_name:
                        return [getattr(item, 'identifier', getattr(item, 'name', '')) for item in node.enum_items]
    return []

def get_modifier_input(mod, ident):
    try:
        if mod.is_property_set(ident): return mod[ident], True
    except Exception: pass

    if hasattr(mod, "properties") and hasattr(mod.properties, "inputs"):
        prop_input = getattr(mod.properties.inputs, ident, None)
        if prop_input is not None and hasattr(prop_input, "value"):
            return prop_input.value, True
    return None, False

def set_modifier_input(mod, ident, value):
    try:
        mod[ident] = value
        return
    except (TypeError, Exception): pass
    if hasattr(mod, "properties") and hasattr(mod.properties, "inputs"):
        prop_input = getattr(mod.properties.inputs, ident, None)
        if prop_input is not None and hasattr(prop_input, "value"):
            prop_input.value = value

def unset_modifier_input(mod, ident, default_val):
    try:
        mod.property_unset(ident)
        return
    except Exception: pass
    try:
        del mod[ident]
        return
    except Exception: pass
    if default_val is not None:
        try:
            mod[ident] = default_val
            return
        except Exception: pass
    if hasattr(mod, "properties") and hasattr(mod.properties, "inputs"):
        prop_input = getattr(mod.properties.inputs, ident, None)
        if prop_input is not None and hasattr(prop_input, "value") and default_val is not None:
            prop_input.value = default_val

def get_input_value(inp):
    if inp.override_type == 'BOOLEAN': return inp.value_bool
    elif inp.override_type == 'INT': return inp.value_int
    elif inp.override_type == 'FLOAT': return inp.value_float
    elif inp.override_type == 'STRING': return inp.value_string
    elif inp.override_type == 'MENU': return inp.value_menu
    return None

# More specific levels win: an Object override replaces the Collection/Preset/Global value of the same socket.
LEVEL_RANK = {"NONE": -1, "GLOBAL": 0, "PRESET": 1, "COLLECTION": 2, "OBJECT": 3}

def override_param_key(ovr, input_name):
    """Identity of one overridden socket, independent of the hierarchy level that defines it."""
    pg_name = ovr.parent_group_ptr.name if ovr.parent_group_ptr else ""
    node = clean_node_name(ovr.node_name) if ovr.override_target == 'NODE' else ""
    return (ovr.override_target, pg_name, node, input_name)

def resolve_overrides(overrides):
    """Drop inherited values of a socket whenever a more specific level also defines that socket."""
    best_rank = {}
    for ovr in overrides:
        rank = LEVEL_RANK.get(ovr.level, -1)
        for inp in ovr.inputs:
            key = override_param_key(ovr, inp.input_name)
            best_rank[key] = max(best_rank.get(key, rank), rank)

    resolved = []
    for ovr in overrides:
        rank = LEVEL_RANK.get(ovr.level, -1)
        kept = [inp for inp in ovr.inputs if best_rank[override_param_key(ovr, inp.input_name)] == rank]
        if kept: resolved.append(MockOverride(ovr.override_target, ovr.parent_group_ptr, ovr.node_name, kept, ovr.level))
    return resolved

def get_override_signature(overrides):
    """Hashable fingerprint of everything that decides which variants an object gets and how they are named."""
    sig = []
    for ovr in overrides:
        inputs_sig = tuple(
            (inp.input_name, inp.override_type, get_input_value(inp), inp.use_sweep, inp.sweep_range,
             inp.sweep_start_float, inp.sweep_step_float, inp.sweep_count_float,
             inp.sweep_start_int, inp.sweep_step_int, inp.sweep_count_int,
             inp.use_tag, inp.tag, inp.use_dir)
            for inp in ovr.inputs
        )
        sig.append((ovr.level, override_param_key(ovr, ""), inputs_sig))
    return tuple(sig)

class MockInput:
    def __init__(self, base_inp, override_val, is_temp=False):
        self.input_name = getattr(base_inp, 'input_name', getattr(base_inp, 'name', ''))
        self.name = self.input_name
        self.override_type = base_inp.override_type
        source = override_val if is_temp else base_inp
        self.use_tag = getattr(source, "use_tag", False)
        self.tag = getattr(source, "tag", "")
        self.use_dir = getattr(source, "use_dir", False)
        self.use_sweep = getattr(source, "use_sweep", False)
        self.sweep_range = getattr(source, "sweep_range", "")
        self.sweep_start_float = getattr(source, "sweep_start_float", 0.0)
        self.sweep_step_float = getattr(source, "sweep_step_float", 1.0)
        self.sweep_count_float = getattr(source, "sweep_count_float", 2)
        self.sweep_start_int = getattr(source, "sweep_start_int", 0)
        self.sweep_step_int = getattr(source, "sweep_step_int", 1)
        self.sweep_count_int = getattr(source, "sweep_count_int", 2)
        self._val = override_val
        self._is_temp = is_temp

    @property
    def value_bool(self): return self._val.value_bool if self._is_temp else bool(self._val)
    @property
    def value_int(self): return self._val.value_int if self._is_temp else (int(self._val) if self._val is not None else 0)
    @property
    def value_float(self): return self._val.value_float if self._is_temp else (float(self._val) if self._val is not None else 0.0)
    @property
    def value_string(self): return self._val.value_string if self._is_temp else str(self._val)
    @property
    def value_menu(self): return self._val.value_menu if self._is_temp else str(self._val)

class MockOverride:
    def __init__(self, target, ptr, node_name, inputs, level="NONE"):
        self.override_target = target
        self.parent_group_ptr = ptr
        self.node_name = node_name
        self.inputs = inputs
        self.level = level

def get_flat_overrides(nodegroups, level="NONE"):
    overrides = []
    if not nodegroups: return overrides
    for ng in nodegroups:
        ng_ptr = bpy.data.node_groups.get(ng.group_name)
        if not ng_ptr: continue
        for node in ng.nodes:
            target = 'MODIFIER' if not node.name or node.name == "<Modifier Interface>" else 'NODE'
            temp_inputs = [MockInput(inp, val, is_temp=True) for inp in node.inputs for val in inp.values]
            if temp_inputs:
                overrides.append(MockOverride(target, ng_ptr, node.name, temp_inputs, level))
    return overrides

def parse_sweep_values(ovr, inp):
    if inp.override_type == 'BOOLEAN':
        return [True, False]
    elif inp.override_type == 'STRING':
        if not inp.sweep_range:
            return [""]
        res = [s.strip() for s in inp.sweep_range.split(',') if s.strip()]
        return res if res else [""]
    elif inp.override_type in ['INT', 'FLOAT']:
        vals = []
        is_float = (inp.override_type == 'FLOAT')
        start = getattr(inp, "sweep_start_float" if is_float else "sweep_start_int", 0.0 if is_float else 0)
        step = getattr(inp, "sweep_step_float" if is_float else "sweep_step_int", 1.0 if is_float else 1)
        count = getattr(inp, "sweep_count_float" if is_float else "sweep_count_int", 2)

        if count <= 0:
            vals.append(round(start, 8) if is_float else int(start))
        else:
            for i in range(count):
                v = round(start + i * step, 8) if is_float else int(start + i * step)
                vals.append(v)
        return vals
    elif inp.override_type == 'MENU':
        items = get_menu_switch_items(ovr.parent_group_ptr, ovr.node_name, inp.input_name)
        return items if items else [""]
    return []

def _build_override_pools(overrides):
    grouped_inputs = {}
    for ovr in overrides:
        for inp in ovr.inputs:
            grouped_inputs.setdefault(override_param_key(ovr, inp.input_name), []).append((ovr, inp))

    pools = []
    for pairs in grouped_inputs.values():
        value_groups = {}
        for ovr, inp in pairs:
            vals_to_process = parse_sweep_values(ovr, inp) if getattr(inp, "use_sweep", False) else [get_input_value(inp)]
            for val in vals_to_process:
                value_groups.setdefault(val, []).append((ovr, MockInput(inp, val)))
        if value_groups: pools.append(list(value_groups.values()))
    return pools

def count_override_combinations(overrides):
    return math.prod(len(pool) for pool in _build_override_pools(overrides))

def generate_override_combinations(overrides):
    pools = _build_override_pools(overrides)
    if not pools: return [[]]

    combinations = list(itertools.product(*pools))
    return [[var for variation in combo for var in variation] for combo in combinations]

def reconstruct_overrides_for_combo(combo):
    grouped = {}
    for ovr, inp in combo:
        target_key = (ovr.override_target, ovr.parent_group_ptr, ovr.node_name, getattr(ovr, "level", "NONE"))
        grouped.setdefault(target_key, []).append(inp)
    return [MockOverride(tgt, ptr, name, inputs, lvl) for (tgt, ptr, name, lvl), inputs in grouped.items()]

def capture_baseline_states(overrides, target_objects):
    global_states, mod_states = [], []
    processed_node_sockets, processed_mod_sockets = set(), set()

    for ovr in overrides:
        if ovr.override_target == 'NODE' and ovr.parent_group_ptr and ovr.node_name:
            parent_tree = ovr.parent_group_ptr
            target_node = parent_tree.nodes.get(clean_node_name(ovr.node_name))
            if not target_node: continue
            for inp in ovr.inputs:
                socket = target_node.inputs.get(inp.input_name)
                if not socket: continue
                key = (parent_tree.name, target_node.name, inp.input_name)
                if key not in processed_node_sockets:
                    link_from = socket.links[0].from_socket if socket.is_linked else None
                    default_v = socket.default_value.copy() if hasattr(socket.default_value, "copy") else socket.default_value
                    global_states.append(('SOCKET', socket, default_v, link_from, parent_tree))
                    processed_node_sockets.add(key)

        elif ovr.override_target == 'MODIFIER' and ovr.parent_group_ptr:
            for inp in ovr.inputs:
                ident = get_modifier_socket_identifier(ovr.parent_group_ptr, inp.input_name)
                if not ident: continue
                default_val = get_modifier_socket_default(ovr.parent_group_ptr, inp.input_name)
                for obj in target_objects:
                    for mod in obj.modifiers:
                        if mod.type == 'NODES' and mod.node_group == ovr.parent_group_ptr:
                            key = (obj.name, mod.name, ident)
                            if key not in processed_mod_sockets:
                                orig_val, is_set = get_modifier_input(mod, ident)
                                if hasattr(orig_val, "copy"):
                                    orig_val = orig_val.copy()
                                mod_states.append((mod, ident, is_set, orig_val, default_val))
                                processed_mod_sockets.add(key)
    return global_states, mod_states

def menu_value_for_modifier(mod, ident, group, node_name, input_name, value):
    """Menu inputs can be stored on the modifier as an item number instead of the item name."""
    current, is_set = get_modifier_input(mod, ident)
    if not is_set: current = get_modifier_socket_default(group, input_name)
    if isinstance(current, int) and not isinstance(current, bool):
        items = get_menu_switch_items(group, node_name, input_name)
        if value in items: return items.index(value)
    return value

def apply_overrides(overrides, target_objects):
    trees_to_update, objects_to_update = set(), set()
    for override in overrides:
        if override.override_target == 'NODE' and override.parent_group_ptr and override.node_name:
            parent_tree = override.parent_group_ptr
            target_node = parent_tree.nodes.get(clean_node_name(override.node_name))
            if not target_node: continue
            for inp in override.inputs:
                socket = target_node.inputs.get(inp.input_name)
                if not socket: continue
                if socket.is_linked: parent_tree.links.remove(socket.links[0])
                val = get_input_value(inp)
                if val is not None:
                    try:
                        if socket.default_value != val:
                            socket.default_value = val
                            trees_to_update.add(parent_tree)
                    except (TypeError, ValueError):
                        try:
                            socket.default_value = (val, val, val)
                            trees_to_update.add(parent_tree)
                        except Exception: pass

        elif override.override_target == 'MODIFIER' and override.parent_group_ptr:
            for inp in override.inputs:
                ident = get_modifier_socket_identifier(override.parent_group_ptr, inp.input_name)
                val = get_input_value(inp)
                if not ident or val is None: continue
                for obj in target_objects:
                    for mod in obj.modifiers:
                        if mod.type == 'NODES' and mod.node_group == override.parent_group_ptr:
                            mod_val = val
                            if inp.override_type == 'MENU':
                                mod_val = menu_value_for_modifier(mod, ident, override.parent_group_ptr, override.node_name, inp.input_name, val)
                            orig_val, is_set = get_modifier_input(mod, ident)
                            if not is_set or orig_val != mod_val:
                                set_modifier_input(mod, ident, mod_val)
                                objects_to_update.add(obj)
    for tree in trees_to_update: tree.update_tag()
    # Setting modifier inputs from Python does not reliably re-evaluate the object, so tag it explicitly.
    for obj in objects_to_update: obj.update_tag()

def revert_overrides(global_states, mod_states, target_objects):
    objects_to_update = set()
    for mod, ident, is_set, orig_val, default_val in mod_states:
        try:
            curr_val, curr_is_set = get_modifier_input(mod, ident)
            if is_set and orig_val is not None:
                is_diff = not curr_is_set or (curr_val != orig_val)
                if hasattr(is_diff, "__iter__"): is_diff = any(is_diff)
                if is_diff:
                    set_modifier_input(mod, ident, orig_val)
                    objects_to_update.add(mod.id_data)
            else:
                if curr_is_set:
                    unset_modifier_input(mod, ident, default_val)
                    objects_to_update.add(mod.id_data)
        except (ReferenceError, Exception): pass
    for obj in objects_to_update:
        try: obj.update_tag()
        except (ReferenceError, Exception): pass

    trees_to_update = set()
    for state in global_states:
        if state[0] == 'SOCKET':
            _, socket, original_val, link_from, parent_tree = state
            try:
                is_diff = (socket.default_value != original_val)
                if hasattr(is_diff, "__iter__"): is_diff = any(is_diff)
                if is_diff:
                    socket.default_value = original_val
                    trees_to_update.add(parent_tree)
                if link_from and not any(l.from_socket == link_from for l in socket.links):
                    parent_tree.links.new(link_from, socket)
                    trees_to_update.add(parent_tree)
            except Exception: pass

    for tree in trees_to_update:
        try: tree.update_tag()
        except (ReferenceError, Exception): pass

def get_active_preset(scene):
    presets = scene.batch_stl_presets
    idx = scene.batch_stl_preset_index
    return presets[idx] if presets and 0 <= idx < len(presets) else None

def get_active_collection(preset):
    return preset.collections[preset.collection_index] if preset and preset.collections and 0 <= preset.collection_index < len(preset.collections) else None

def get_active_object(collection):
    return collection.objects[collection.object_index] if collection and collection.objects and 0 <= collection.object_index < len(collection.objects) else None

def get_job(preset_index, create=False):
    """Runtime export state of a preset (keyed by its index). Creating requires an operator context, not a draw call."""
    jobs = bpy.context.window_manager.batch_stl_jobs
    for job in jobs:
        if job.preset_index == preset_index: return job
    if not create: return None
    job = jobs.add()
    job.preset_index = preset_index
    return job

def is_any_exporting():
    return any(job.is_exporting for job in bpy.context.window_manager.batch_stl_jobs)

def log_to_console(job, text):
    if job:
        job.console_logs.add().text = text
        if len(job.console_logs) > 300: job.console_logs.remove(0)
        job.console_index = len(job.console_logs) - 1

def format_export_filename(bl_obj_name, obj_tag, col_use_tag, col_tag, combo_suffix=""):
    safe_name = bpy.path.clean_name(bl_obj_name)
    obj_tag = sanitize_name(obj_tag)
    if obj_tag:
        # _tag appends, tag_ prepends; a bare tag fully replaces the object name as the filename
        safe_name = f"{safe_name}{obj_tag}" if obj_tag.startswith("_") else (f"{obj_tag}{safe_name}" if obj_tag.endswith("_") else obj_tag)
    tag_suffix = sanitize_name(col_tag) if col_use_tag and col_tag else ""
    return f"{safe_name}{tag_suffix}{combo_suffix}.stl"

def build_export_dir_parts(preset_prefix, col_sub_path, obj_sub_path, paths_by_level=None):
    if paths_by_level is None: paths_by_level = {}
    c_parts = split_path_parts(col_sub_path)
    o_parts = split_path_parts(obj_sub_path)
    p_prefix = split_path_parts(preset_prefix)
    return (
        paths_by_level.get("GLOBAL", []) +
        p_prefix +
        paths_by_level.get("PRESET", []) +
        c_parts +
        paths_by_level.get("COLLECTION", []) +
        o_parts +
        paths_by_level.get("OBJECT", []) +
        paths_by_level.get("NONE", [])
    )

def compute_override_freq_dict(overrides):
    freq_dict = {}
    for o in overrides:
        for i in o.inputs:
            key = override_param_key(o, i.input_name)
            freq_dict[key] = freq_dict.get(key, 0) + (2 if getattr(i, "use_sweep", False) else 1)
    return freq_dict

def evaluate_combo_naming(combo, freq_dict):
    combo_suffix = ""
    paths_by_level = {"GLOBAL": [], "PRESET": [], "COLLECTION": [], "OBJECT": [], "NONE": []}
    processed_params = set()

    for ovr, inp in combo:
        param_key = override_param_key(ovr, inp.input_name)
        if param_key not in processed_params:
            val = get_input_value(inp)
            val_str = f"{val:g}" if isinstance(val, float) else str(val)
            naming_str = f"{val_str}{inp.tag}" if inp.tag.startswith("_") else (f"{inp.tag}{val_str}" if inp.tag.endswith("_") else inp.tag) if inp.tag else val_str
            naming_str = sanitize_name(naming_str)

            if freq_dict.get(param_key, 0) > 1:
                if getattr(inp, "use_tag", False): combo_suffix += f"_{naming_str}"
                dir_part = naming_str.strip(" .")
                if getattr(inp, "use_dir", False) and dir_part: paths_by_level[getattr(ovr, "level", "NONE")].append(dir_part)
            processed_params.add(param_key)
    return combo_suffix, paths_by_level

def sync_collection_objects(col_prop, col_ptr=None):
    if not col_ptr: col_ptr = bpy.data.collections.get(col_prop.collection_name)
    if not col_ptr: return

    actual_names = {obj.name for obj in col_ptr.all_objects if obj.type in SUPPORTED_OBJECT_TYPES}
    existing_names = {obj.name: obj for obj in col_prop.objects}

    for i in reversed(range(len(col_prop.objects))):
        if col_prop.objects[i].name not in actual_names: col_prop.objects.remove(i)

    for name in actual_names:
        if name not in existing_names:
            new_obj = col_prop.objects.add()
            new_obj.name = name
            new_obj.export = True

# --- BINARY STL WRITER ---
def mesh_to_stl_array(mesh, matrix_world):
    """Return the mesh as a world-space STL record array, or None when it has no triangles."""
    mesh.calc_loop_triangles()
    num_tris = len(mesh.loop_triangles)
    if num_tris == 0 or len(mesh.vertices) == 0: return None

    verts = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", verts.ravel())

    mat_3x3 = np.array(matrix_world.to_3x3(), dtype=np.float32)
    trans = np.array(matrix_world.translation, dtype=np.float32)
    verts = np.dot(verts, mat_3x3.T) + trans

    tri_verts = np.empty((num_tris, 3), dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri_verts.ravel())

    tri_normals = np.empty((num_tris, 3), dtype=np.float32)
    mesh.loop_triangles.foreach_get("normal", tri_normals.ravel())

    mat_inv = np.array(matrix_world.to_3x3().inverted_safe(), dtype=np.float32)
    tri_normals = np.dot(tri_normals, mat_inv)

    norms = np.sqrt(np.sum(tri_normals**2, axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    tri_normals /= norms

    data = np.zeros(num_tris, dtype=STL_DTYPE)
    data['normals'] = tri_normals
    data['v0'] = verts[tri_verts[:, 0]]

    if matrix_world.determinant() < 0.0:
        data['v1'] = verts[tri_verts[:, 2]]
        data['v2'] = verts[tri_verts[:, 1]]
    else:
        data['v1'] = verts[tri_verts[:, 1]]
        data['v2'] = verts[tri_verts[:, 2]]
    return data

def collect_instance_arrays(depsgraph, target_objects):
    """One pass over the depsgraph instances (e.g. unrealized Geometry Nodes instances) generated by target_objects.
    Returns {object name_full: [STL arrays]}; instance data is only valid while iterating, so it is converted immediately."""
    wanted = {obj.name_full for obj in target_objects}
    result = {}
    if not wanted: return result
    for inst in depsgraph.object_instances:
        if not inst.is_instance or not inst.parent: continue
        parent_name = inst.parent.original.name_full
        if parent_name not in wanted: continue
        inst_obj = inst.object
        try: mesh = inst_obj.to_mesh()
        except RuntimeError: continue
        if not mesh: continue
        try:
            arr = mesh_to_stl_array(mesh, inst.matrix_world)
        finally:
            inst_obj.to_mesh_clear()
        if arr is not None: result.setdefault(parent_name, []).append(arr)
    return result

def write_object_stl(filepath, bl_obj, depsgraph, instance_arrays=()):
    """Write the evaluated object (plus its pre-collected instances) as one binary STL. Returns the triangle count."""
    arrays = []
    obj_eval = bl_obj.evaluated_get(depsgraph)
    try: mesh = obj_eval.to_mesh()
    except RuntimeError: mesh = None
    if mesh:
        try:
            arr = mesh_to_stl_array(mesh, obj_eval.matrix_world)
        finally:
            obj_eval.to_mesh_clear()
        if arr is not None: arrays.append(arr)
    arrays.extend(instance_arrays)

    num_tris = sum(len(a) for a in arrays)
    if num_tris == 0: return 0
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(STL_HEADER)
        f.write(struct.pack('<I', num_tris))
        for a in arrays: a.tofile(f)
    return num_tris

# --- JSON UTILS ---
def copy_val_to_dict(v):
    return {
        "value_bool": v.value_bool, "value_int": v.value_int, "value_float": v.value_float,
        "value_string": v.value_string, "value_menu": v.value_menu, "use_tag": v.use_tag,
        "tag": v.tag, "use_dir": v.use_dir, "use_sweep": getattr(v, "use_sweep", False),
        "sweep_range": getattr(v, "sweep_range", ""),
        "sweep_start_float": getattr(v, "sweep_start_float", 0.0),
        "sweep_step_float": getattr(v, "sweep_step_float", 1.0),
        "sweep_count_float": getattr(v, "sweep_count_float", 2),
        "sweep_start_int": getattr(v, "sweep_start_int", 0),
        "sweep_step_int": getattr(v, "sweep_step_int", 1),
        "sweep_count_int": getattr(v, "sweep_count_int", 2)
    }

def copy_input_to_dict(i):
    return {"name": i.name, "override_type": i.override_type, "values": [copy_val_to_dict(v) for v in i.values]}

def copy_node_to_dict(n):
    return {"name": n.name, "inputs": [copy_input_to_dict(i) for i in n.inputs]}

def copy_ng_to_dict(ng):
    return {"group": ng.group_name, "nodes": [copy_node_to_dict(n) for n in ng.nodes]}

def copy_obj_to_dict(o):
    return {"name": o.name, "export": o.export, "tag": getattr(o, "tag", ""), "sub_path": getattr(o, "sub_path", ""), "nodegroups": [copy_ng_to_dict(ng) for ng in o.nodegroups]}

def copy_collection_to_dict(c):
    return {"collection_name": c.collection_name, "use_tag": c.use_tag, "tag": c.tag, "sub_path": c.sub_path, "objects": [copy_obj_to_dict(o) for o in c.objects], "nodegroups": [copy_ng_to_dict(ng) for ng in c.nodegroups]}

def copy_preset_to_dict(src):
    return {"name": src.name, "preset_prefix": src.preset_prefix, "collections": [copy_collection_to_dict(c) for c in src.collections], "nodegroups": [copy_ng_to_dict(ng) for ng in src.nodegroups]}

def paste_val_from_dict(new_v, data):
    for k, v in data.items():
        if hasattr(new_v, k): setattr(new_v, k, v)

def paste_input_from_dict(new_i, data):
    new_i.name = data["name"]
    new_i.override_type = data.get("override_type", 'FLOAT')
    for v_data in data.get("values", []): paste_val_from_dict(new_i.values.add(), v_data)

def paste_node_from_dict(new_n, data):
    new_n.name = data["name"]
    for i_data in data.get("inputs", []): paste_input_from_dict(new_n.inputs.add(), i_data)

def paste_ng_from_dict(new_ng, data):
    new_ng.group_name = data.get("group", "")
    for n_data in data.get("nodes", []): paste_node_from_dict(new_ng.nodes.add(), n_data)

def paste_obj_from_dict(new_o, data):
    new_o.name = data.get("name", "")
    new_o.export = data.get("export", True)
    new_o.tag = data.get("tag", "")
    new_o.sub_path = data.get("sub_path", "")
    for ng_data in data.get("nodegroups", []): paste_ng_from_dict(new_o.nodegroups.add(), ng_data)

def paste_collection_from_dict(new_c, data):
    new_c.collection_name = data.get("collection_name", "")
    new_c.use_tag = data.get("use_tag", True)
    new_c.tag = data.get("tag", "")
    new_c.sub_path = data.get("sub_path", "")
    for o_data in data.get("objects", []): paste_obj_from_dict(new_c.objects.add(), o_data)
    for ng_data in data.get("nodegroups", []): paste_ng_from_dict(new_c.nodegroups.add(), ng_data)

def paste_preset_from_dict(new_p, data):
    new_p.name = data.get("name", "Imported Preset")
    new_p.preset_prefix = data.get("preset_prefix", "")
    for c_data in data.get("collections", []): paste_collection_from_dict(new_p.collections.add(), c_data)
    for ng_data in data.get("nodegroups", []): paste_ng_from_dict(new_p.nodegroups.add(), ng_data)

# --- UI CACHE ENGINE ---
def is_override_group_valid(ng):
    if not ng.group_name or not bpy.data.node_groups.get(ng.group_name):
        return False
    return True

def is_override_node_valid(ng_ptr, node):
    if not node.name or node.name == "<Modifier Interface>":
        return True
    if not ng_ptr:
        return False
    return clean_node_name(node.name) in ng_ptr.nodes

def is_override_input_valid(ng_ptr, node, inp):
    if not inp.name or not ng_ptr:
        return False
    is_mod = not node.name or node.name == "<Modifier Interface>"
    if is_mod:
        if hasattr(ng_ptr, "interface"):
            for item in ng_ptr.interface.items_tree:
                if getattr(item, "item_type", "") == 'SOCKET' and getattr(item, "in_out", "INPUT") == 'INPUT' and item.name == inp.name:
                    return True
            return False
        elif hasattr(ng_ptr, "inputs"):
            return inp.name in ng_ptr.inputs
        return False
    target_n = ng_ptr.nodes.get(clean_node_name(node.name))
    if not target_n:
        return False
    return inp.name in target_n.inputs

def is_override_val_valid(inp, val, ng_ptr=None, node=None):
    if val is None or inp.override_type not in SUPPORTED_OVERRIDE_TYPES:
        return False
    if getattr(val, "use_sweep", False):
        if inp.override_type == 'FLOAT':
            return getattr(val, "sweep_count_float", 0) >= 1
        elif inp.override_type == 'INT':
            return getattr(val, "sweep_count_int", 0) >= 1
        elif inp.override_type == 'STRING':
            return bool(val.sweep_range and val.sweep_range.strip())
        return True
    if inp.override_type == 'STRING':
        return bool(val.value_string and val.value_string.strip())
    elif inp.override_type == 'MENU':
        if not (val.value_menu and val.value_menu.strip()):
            return False
        if ng_ptr and node:
            valid_items = get_menu_switch_items(ng_ptr, node.name, inp.name)
            if valid_items and val.value_menu not in valid_items:
                return False
        return True
    return True

def validate_overrides(nodegroups):
    for ng in nodegroups:
        if not is_override_group_valid(ng):
            return False
        ng_ptr = bpy.data.node_groups.get(ng.group_name)
        for node in ng.nodes:
            if not is_override_node_valid(ng_ptr, node):
                return False
            for inp in node.inputs:
                if not is_override_input_valid(ng_ptr, node, inp):
                    return False
                if not inp.values:
                    return False
                for val in inp.values:
                    if not is_override_val_valid(inp, val, ng_ptr, node):
                        return False
    return True

def is_preset_setup_valid(scene, preset):
    if not validate_overrides(scene.batch_stl_global_nodegroups):
        return False
    if not validate_overrides(preset.nodegroups):
        return False
    for c in preset.collections:
        if not validate_overrides(c.nodegroups):
            return False
        for obj in c.objects:
            if obj.export:
                if not validate_overrides(obj.nodegroups):
                    return False
    return True

def check_ng_for_overrides(nodegroups):
    has_ovr = len(nodegroups) > 0
    has_perm = any(len(i.values) > 1 or any(getattr(v, "use_sweep", False) for v in i.values) for ng in nodegroups for n in ng.nodes for i in n.inputs)
    return has_ovr, has_perm

def _get_preset_status(scene, preset, g_has_ovr=False, g_has_perm=False):
    pho, php = check_ng_for_overrides(preset.nodegroups)
    has_ovr = g_has_ovr or pho
    has_perm = g_has_perm or php

    for c in preset.collections:
        cho, chp = check_ng_for_overrides(c.nodegroups)
        has_ovr = has_ovr or cho
        has_perm = has_perm or chp
        for obj in c.objects:
            if obj.export:
                oho, ohp = check_ng_for_overrides(obj.nodegroups)
                has_ovr = has_ovr or oho
                has_perm = has_perm or ohp
        if has_ovr and has_perm:
            return True, True
    return has_ovr, has_perm

def rebuild_ui_cache_if_dirty():
    if not _ui_cache.get("is_dirty", False): return 0.1
    _ui_cache["is_dirty"] = False

    try:
        context = bpy.context
        if not hasattr(context, "scene") or not context.scene:
            _ui_cache["is_dirty"] = True
            return 0.1

        scene = context.scene

        gho, ghp = check_ng_for_overrides(scene.batch_stl_global_nodegroups)
        preset_metrics = {}
        for p_idx, p in enumerate(scene.batch_stl_presets):
            ho, hp = _get_preset_status(scene, p, gho, ghp)
            preset_metrics[p_idx] = {"has_ovr": ho, "has_perm": hp}
        _ui_cache["preset_metrics"] = preset_metrics

        visibility = {}
        if hasattr(context, "view_layer") and context.view_layer:
            def traverse(layer_collection, parent_excluded=False):
                current_excluded = parent_excluded or layer_collection.exclude
                if layer_collection.collection:
                    cname = layer_collection.collection.name
                    visibility[cname] = visibility.get(cname, True) and current_excluded
                for child in layer_collection.children: traverse(child, current_excluded)
            traverse(context.view_layer.layer_collection)
        _ui_cache["visibility"] = visibility

        preset = get_active_preset(scene)
        total_presets = len(scene.batch_stl_presets)
        g_cols, g_objs, g_exp = 0, 0, 0
        preset_stats, col_stats = {}, {}

        global_ovrs = get_flat_overrides(scene.batch_stl_global_nodegroups, "GLOBAL")

        for p_idx, p in enumerate(scene.batch_stl_presets):
            p_cols, p_objs, p_exp = len(p.collections), 0, 0
            preset_ovrs = global_ovrs + get_flat_overrides(p.nodegroups, "PRESET")

            for c_idx, c in enumerate(p.collections):
                c_ptr = bpy.data.collections.get(c.collection_name)
                sync_collection_objects(c, c_ptr)
                c_objs, c_exp = 0, 0

                if c_ptr and not visibility.get(c.collection_name, True):
                    c_pinned_ovrs = preset_ovrs + get_flat_overrides(c.nodegroups, "COLLECTION")
                    for obj_prop in c.objects:
                        if not obj_prop.export: continue
                        bl_obj = c_ptr.all_objects.get(obj_prop.name)
                        if bl_obj and bl_obj.type in SUPPORTED_OBJECT_TYPES and not bl_obj.hide_viewport:
                            c_objs += 1
                            p_objs += 1
                            obj_ovrs = resolve_overrides(c_pinned_ovrs + get_flat_overrides(obj_prop.nodegroups, "OBJECT"))
                            c_exp += count_override_combinations(obj_ovrs)

                p_exp += c_exp
                col_stats[(p_idx, c_idx)] = {"objs": c_objs, "exp": c_exp}

            g_cols += p_cols; g_objs += p_objs; g_exp += p_exp
            preset_stats[p_idx] = {"cols": p_cols, "objs": p_objs, "exp": p_exp}

        _ui_cache["stats"] = {"global": {"presets": total_presets, "cols": g_cols, "objs": g_objs, "exp": g_exp}, "presets": preset_stats, "cols": col_stats}

        show_console = getattr(scene, "batch_stl_show_console", False)
        info_tab = getattr(scene, "batch_stl_info_tab", 'LOG')

        if not show_console or info_tab != 'TREE':
            redraw_sidebars(context)
            return 0.1

        is_global = getattr(scene, "batch_stl_info_global", False)

        if not preset and not is_global:
            _ui_cache["tree"] = ({}, set())
        else:
            _ui_cache["tree"] = build_tree_dict(context, visibility, is_global)

        redraw_sidebars(context)
        return 0.1
    except Exception:
        try:
            if bpy.context.scene.batch_stl_verbose_console:
                traceback.print_exc()
        except Exception:
            pass
        return 0.25

@persistent
def batch_stl_depsgraph_handler(*args):
    mark_dirty()

# --- TREE VISUALIZER LOGIC ---
def build_tree_dict(context, visibility_cache=None, is_global=False):
    scene = context.scene
    root_name = bpy.path.abspath(scene.batch_stl_root_dir) if scene.batch_stl_root_dir else "//"
    root_name = os.path.normpath(root_name)
    tree, all_filepaths, duplicates = {}, set(), set()
    global_ovrs = get_flat_overrides(scene.batch_stl_global_nodegroups, "GLOBAL")
    target_presets = scene.batch_stl_presets if is_global else ([get_active_preset(scene)] if get_active_preset(scene) else [])

    for preset in target_presets:
        if not preset: continue
        preset_ovrs = get_flat_overrides(preset.nodegroups, "PRESET")

        for c in preset.collections:
            c_ptr = bpy.data.collections.get(c.collection_name)
            if not c_ptr: continue
            if visibility_cache is not None:
                if visibility_cache.get(c.collection_name, True): continue
            elif is_collection_excluded(context, c_ptr): continue

            col_ovrs = get_flat_overrides(c.nodegroups, "COLLECTION")

            for obj_prop in c.objects:
                if not obj_prop.export: continue
                bl_obj = c_ptr.all_objects.get(obj_prop.name)
                if not bl_obj or bl_obj.hide_viewport or bl_obj.type not in SUPPORTED_OBJECT_TYPES: continue

                obj_ovrs = get_flat_overrides(obj_prop.nodegroups, "OBJECT")
                all_overrides = resolve_overrides(global_ovrs + preset_ovrs + col_ovrs + obj_ovrs)

                freq_dict = compute_override_freq_dict(all_overrides)
                combinations = generate_override_combinations(all_overrides) or [[]]

                for combo in combinations:
                    combo_suffix, paths_by_level = evaluate_combo_naming(combo, freq_dict)
                    full_dir_parts = build_export_dir_parts(preset.preset_prefix, c.sub_path, obj_prop.sub_path, paths_by_level)

                    combo_root = tree
                    for part in full_dir_parts:
                        combo_root = combo_root.setdefault(part, {})

                    filename = format_export_filename(bl_obj.name, obj_prop.tag, getattr(c, 'use_tag', False), c.tag, combo_suffix)
                    combo_root.setdefault('_files', []).append(filename)

                    full_path_key = clash_key(os.path.join(root_name, *full_dir_parts, filename))
                    if full_path_key in all_filepaths:
                        duplicates.add(full_path_key)
                    else:
                        all_filepaths.add(full_path_key)

    return {root_name: tree}, duplicates

def draw_tree_dict(layout, tree_node, current_path="", toggled_list=None, duplicates=None, actual_path=""):
    if toggled_list is None:
        try: toggled_list = json.loads(bpy.context.scene.batch_stl_collapsed_dirs)
        except Exception: toggled_list = []
    if duplicates is None: duplicates = set()

    dirs = [k for k in tree_node.keys() if k != '_files']
    for k in dirs:
        dir_path = f"{current_path}/{k}"
        next_actual = os.path.normpath(os.path.join(actual_path, k)) if actual_path else os.path.normpath(k)
        # The root folder starts expanded and everything below it collapsed; a click on the arrow flips that default.
        expanded_by_default = (current_path == "")
        is_collapsed = (dir_path in toggled_list) if expanded_by_default else (dir_path not in toggled_list)

        split = layout.split(factor=0.005)
        split.column()
        box = split.column().box()
        row = box.row()
        row.operator("batch_stl.toggle_dir_tree", text="", icon=ICONS['RIGHT'] if is_collapsed else ICONS['DOWN'], emboss=False).dir_path = dir_path
        row.scale_y = 0.4
        row.label(text=str(k))
        if not is_collapsed and isinstance(tree_node[k], dict):
            draw_tree_dict(box, tree_node[k], dir_path, toggled_list, duplicates, next_actual)

    for f in tree_node.get('_files', []):
        split = layout.split(factor=0.025)
        split.column()        # Consume the 2.5% width as an empty indent spacer
        col = split.column()  # Assign the remaining 97.5% width to your content

        row = col.row()
        row.scale_y = 0.4
        check_path = os.path.normpath(os.path.join(actual_path, f)) if actual_path else os.path.normpath(f)
        if clash_key(check_path) in duplicates: row.alert = True
        row.label(text=str(f))

# --- HEADLESS EXPORT EXECUTION ROUTINE ---
def run_headless_export(job_file_path):
    try:
        with open(job_file_path, 'r', encoding="utf-8") as f: job_data = json.load(f)
        preset_index, root_dir, start_time_unix, skip_direct = job_data["preset_index"], job_data["root_dir"], job_data.get("start_time", time.time()), job_data.get("skip_direct", False)
    except Exception as e:
        print(f"ERROR: Failed to load job manifest: {e}", flush=True)
        sys.exit(1)

    scene = bpy.context.scene
    if preset_index < 0 or preset_index >= len(scene.batch_stl_presets):
        print("ERROR: Invalid preset index", flush=True)
        sys.exit(1)

    preset = scene.batch_stl_presets[preset_index]
    print("\n  [Phase 0] Evaluating Targets and Building Depsgraph Culling Maps...", flush=True)
    t_phase0_start = time.perf_counter()

    # Objects whose effective overrides are identical share one batch (one set of variants, one depsgraph pass).
    execution_batches, batch_overrides = {}, {}
    preset_ovrs = get_flat_overrides(scene.batch_stl_global_nodegroups, "GLOBAL") + get_flat_overrides(preset.nodegroups, "PRESET")

    for c in preset.collections:
        c_ptr = bpy.data.collections.get(c.collection_name)
        if not c_ptr or is_collection_excluded(bpy.context, c_ptr): continue
        pinned_ovrs = preset_ovrs + get_flat_overrides(c.nodegroups, "COLLECTION")

        for obj_prop in c.objects:
            if not obj_prop.export: continue
            bl_obj = c_ptr.all_objects.get(obj_prop.name)
            if not bl_obj or bl_obj.hide_viewport or bl_obj.type not in SUPPORTED_OBJECT_TYPES: continue

            obj_overrides = resolve_overrides(pinned_ovrs + get_flat_overrides(obj_prop.nodegroups, "OBJECT"))
            full_sig = get_override_signature(obj_overrides)
            if not full_sig and skip_direct: continue
            execution_batches.setdefault(full_sig, []).append((c, obj_prop, bl_obj))
            batch_overrides.setdefault(full_sig, obj_overrides)

    layer_collection_map, layer_collection_parents = {}, {}
    def map_layer_collections(lc, parent=None):
        if lc.collection:
            layer_collection_map[lc.collection.name] = lc
            layer_collection_parents[lc.collection.name] = parent
        for child in lc.children: map_layer_collections(child, lc)
    map_layer_collections(bpy.context.view_layer.layer_collection)

    print(f"    ├─ Mapped {len(layer_collection_map)} collections for depsgraph culling in {time.perf_counter() - t_phase0_start:.4f}s", flush=True)

    if not execution_batches:
        print("  └─ No active objects to export.\nBATCH_STL_DONE", flush=True)
        sys.exit(0)

    batch_combinations = {sig: generate_override_combinations(ovrs) for sig, ovrs in batch_overrides.items()}
    total_ops = sum(len(items) * len(batch_combinations[sig]) for sig, items in execution_batches.items())
    print(f"BATCH_STL_TOTAL:{total_ops}", flush=True)

    current_op_step, batch_counter, first_export_started = 0, 1, False

    for signature, batch_items in execution_batches.items():
        all_overrides = batch_overrides[signature]
        freq_dict = compute_override_freq_dict(all_overrides)
        combinations = batch_combinations[signature]
        batch_objects = {item[2] for item in batch_items}

        isolated_collections = []
        if len(all_overrides) > 0:
            batch_col_names = {col.name for item in batch_items for col in item[2].users_collection}
            visible_hierarchy = set()
            for c_name in batch_col_names:
                curr = c_name
                while curr in layer_collection_map:
                    visible_hierarchy.add(curr)
                    parent_lc = layer_collection_parents.get(curr)
                    curr = parent_lc.collection.name if (parent_lc and parent_lc.collection) else None
            for name, lc in layer_collection_map.items():
                if name not in visible_hierarchy and not lc.exclude:
                    lc.exclude = True
                    isolated_collections.append(name)

        baseline_global_states, baseline_mod_states = capture_baseline_states(all_overrides, batch_objects)

        try:
            for combo_idx, combo in enumerate(combinations):
                t_perm_start = time.perf_counter()
                if not first_export_started:
                    print(f"=== Headless init took {time.time() - start_time_unix:.2f} s to start first export ===", flush=True)
                    first_export_started = True

                combo_suffix, paths_by_level = evaluate_combo_naming(combo, freq_dict)

                apply_overrides(reconstruct_overrides_for_combo(combo), batch_objects)
                bpy.context.view_layer.update()
                depsgraph = bpy.context.evaluated_depsgraph_get()
                instance_arrays = collect_instance_arrays(depsgraph, batch_objects)

                for c, obj_prop, bl_obj in batch_items:
                    full_dir_parts = build_export_dir_parts(preset.preset_prefix, c.sub_path, obj_prop.sub_path, paths_by_level)
                    out_dir = os.path.normpath(os.path.join(root_dir, *full_dir_parts)) if full_dir_parts else root_dir
                    os.makedirs(out_dir, exist_ok=True)

                    filename = format_export_filename(bl_obj.name, obj_prop.tag, getattr(c, 'use_tag', False), c.tag, combo_suffix)
                    filepath = os.path.join(out_dir, filename)

                    num_tris = write_object_stl(filepath, bl_obj, depsgraph, instance_arrays.get(bl_obj.name_full, ()))
                    result_text = filepath if num_tris else f"SKIPPED (no geometry): {filepath}"

                    print(f"{preset.name} | {c.collection_name} | {bl_obj.name} | Permutation {combo_idx + 1}/{len(combinations)} | Batch {batch_counter}/{len(execution_batches)}\n  └─ {result_text} | {time.perf_counter() - t_perm_start:.2f} s", flush=True)
                    current_op_step += 1
                    print(f"BATCH_STL_PROGRESS:{current_op_step}", flush=True)

        finally:
            revert_overrides(baseline_global_states, baseline_mod_states, batch_objects)
            for name in isolated_collections:
                if name in layer_collection_map: layer_collection_map[name].exclude = False
            bpy.context.view_layer.update()

        batch_counter += 1

    print("BATCH_STL_DONE", flush=True)
    sys.exit(0)

# ==============================================================================
# === [ 3. PROPERTY GROUPS ] ===
# ==============================================================================

# Undo: Blender records an undo step for every property edited in the UI, and the operators below declare
# 'UNDO' in bl_options, so property updates only refresh the UI cache. The exception is search fields,
# which push their own step (see search_field_update).

class HierarchyIterator:
    """Helper to yield all active node groups in the hierarchy context."""
    @staticmethod
    def iterate(scene):
        for ng in scene.batch_stl_global_nodegroups: yield ('GLOBAL', ng)
        preset = get_active_preset(scene)
        if not preset: return
        for ng in preset.nodegroups: yield ('PRESET', ng)
        col = get_active_collection(preset)
        if not col: return
        for ng in col.nodegroups: yield ('COLLECTION', ng)
        obj = get_active_object(col)
        if not obj: return
        for ng in obj.nodegroups: yield ('OBJECT', ng)

SUPPORTED_OVERRIDE_TYPES = ('FLOAT', 'INT', 'BOOLEAN', 'STRING', 'MENU')

def classify_socket_type(socket_type):
    """Map a node socket / interface socket type to an override type; 'UNSUPPORTED' for vectors, colors, objects, ..."""
    if socket_type in ('VALUE', 'FLOAT') or 'Float' in socket_type: return 'FLOAT'
    if socket_type == 'INT' or socket_type.startswith('NodeSocketInt') and 'Vector' not in socket_type: return 'INT'
    if socket_type == 'BOOLEAN' or 'Bool' in socket_type: return 'BOOLEAN'
    if socket_type == 'STRING' or 'String' in socket_type: return 'STRING'
    if socket_type == 'MENU' or 'Menu' in socket_type: return 'MENU'
    return 'UNSUPPORTED'

def infer_input_type(group_ptr, node_name, input_name):
    if not group_ptr or not input_name: return 'FLOAT'
    if not node_name or node_name == "<Modifier Interface>":
        if hasattr(group_ptr, "interface"):
            for it in group_ptr.interface.items_tree:
                if getattr(it, "item_type", "") == 'SOCKET' and getattr(it, "in_out", "INPUT") == 'INPUT' and it.name == input_name:
                    return classify_socket_type(getattr(it, "socket_type", ""))
        elif hasattr(group_ptr, "inputs"):
            inp = group_ptr.inputs.get(input_name)
            if inp: return classify_socket_type(inp.type)
    else:
        node = group_ptr.nodes.get(clean_node_name(node_name))
        if node and input_name in node.inputs:
            return classify_socket_type(node.inputs[input_name].type)
    return 'FLOAT'

def sync_input_type(inp, scene):
    for lvl, ng in HierarchyIterator.iterate(scene):
        for n in ng.nodes:
            if inp in n.inputs.values():
                ng_ptr = bpy.data.node_groups.get(ng.group_name)
                inp.override_type = infer_input_type(ng_ptr, n.name, inp.name)
                for v in inp.values: v.use_sweep = False
                return

def on_input_name_update(self, context):
    mark_dirty()
    try: sync_input_type(self, context.scene)
    except Exception: pass
    push_search_undo("Edit Override Input")

def search_target_node_cb(self, context, edit_text):
    if not context or not getattr(context, "scene", None): return ["<Modifier Interface>"]
    if edit_text == self.name: edit_text = ""
    res = ["<Modifier Interface>"]

    for lvl, ng in HierarchyIterator.iterate(context.scene):
        if self in ng.nodes.values():
            ng_ptr = bpy.data.node_groups.get(ng.group_name)
            if ng_ptr:
                for node in ng_ptr.nodes:
                    val = f"{node.name} [{node.node_tree.name if node.type == 'GROUP' and getattr(node, 'node_tree', None) else node.type}]"
                    if not edit_text or edit_text.lower() in val.lower(): res.append(val)
            return res
    return res

def search_menu_items_cb(self, context, edit_text):
    if not context or not getattr(context, "scene", None): return []
    for lvl, ng in HierarchyIterator.iterate(context.scene):
        for n in ng.nodes:
            for i in n.inputs:
                if self in i.values.values():
                    ng_ptr = bpy.data.node_groups.get(ng.group_name)
                    items = get_menu_switch_items(ng_ptr, n.name, i.name)
                    if edit_text == self.value_menu: edit_text = ""
                    return [item for item in items if edit_text.lower() in item.lower()] if edit_text else items
    return []

class BatchSTLLogLine(bpy.types.PropertyGroup): text: bpy.props.StringProperty()
class BatchSTLValue(bpy.types.PropertyGroup):
    value_bool: bpy.props.BoolProperty(name="Value", default=True, update=mark_dirty)
    value_int: bpy.props.IntProperty(name="Value", default=0, update=mark_dirty)
    value_float: bpy.props.FloatProperty(name="Value", default=0.0, update=mark_dirty)
    value_string: bpy.props.StringProperty(name="Value", default="", update=mark_dirty)
    value_menu: bpy.props.StringProperty(name="Value", default="", search=search_menu_items_cb, update=search_field_update("Edit Override Value"))
    use_tag: bpy.props.BoolProperty(name="Use Tag", default=False, update=mark_dirty)
    tag: bpy.props.StringProperty(name="Tag", default="", update=mark_dirty)
    use_dir: bpy.props.BoolProperty(name="Use Dir", default=True, update=mark_dirty)
    use_sweep: bpy.props.BoolProperty(name="Sweep", default=False, update=mark_dirty)
    sweep_range: bpy.props.StringProperty(name="Sweep Range", default="", update=mark_dirty)
    sweep_start_float: bpy.props.FloatProperty(name="Start", default=0.0, update=mark_dirty)
    sweep_step_float: bpy.props.FloatProperty(name="Step", default=1.0, update=mark_dirty)
    sweep_count_float: bpy.props.IntProperty(name="Steps", default=2, min=1, update=mark_dirty)
    sweep_start_int: bpy.props.IntProperty(name="Start", default=0, update=mark_dirty)
    sweep_step_int: bpy.props.IntProperty(name="Step", default=1, update=mark_dirty)
    sweep_count_int: bpy.props.IntProperty(name="Steps", default=2, min=1, update=mark_dirty)

class BatchSTLInput(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Input Socket", default="", update=on_input_name_update)
    override_type: bpy.props.StringProperty(default='FLOAT', update=mark_dirty)
    values: bpy.props.CollectionProperty(type=BatchSTLValue)

class BatchSTLNode(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Target Node", default="<Modifier Interface>", search=search_target_node_cb, update=search_field_update("Edit Override Node"), description="Select <Modifier Interface> to target the modifier directly")
    inputs: bpy.props.CollectionProperty(type=BatchSTLInput)

class BatchSTLNodeGroup(bpy.types.PropertyGroup):
    group_name: bpy.props.StringProperty(name="Node Group", default="", update=search_field_update("Edit Override Node Group"))
    nodes: bpy.props.CollectionProperty(type=BatchSTLNode)

class BatchSTLObject(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    export: bpy.props.BoolProperty(default=True, update=mark_dirty)
    tag: bpy.props.StringProperty(name="Tag", default="", update=mark_dirty)
    sub_path: bpy.props.StringProperty(name="Sub-folder", default="", update=mark_dirty)
    nodegroups: bpy.props.CollectionProperty(type=BatchSTLNodeGroup)

class BatchSTLCollection(bpy.types.PropertyGroup):
    collection_name: bpy.props.StringProperty(name="Collection", default="", update=search_field_update("Edit Collection"))
    use_tag: bpy.props.BoolProperty(name="Use Tag", default=True, update=mark_dirty)
    tag: bpy.props.StringProperty(name="Tag", default="", update=mark_dirty)
    sub_path: bpy.props.StringProperty(name="Sub-folder", default="", update=mark_dirty)
    objects: bpy.props.CollectionProperty(type=BatchSTLObject)
    object_index: bpy.props.IntProperty(default=0, update=mark_dirty)
    nodegroups: bpy.props.CollectionProperty(type=BatchSTLNodeGroup)

class BatchSTLJob(bpy.types.PropertyGroup):
    """Runtime state of one preset's export. Lives on the WindowManager so it is never saved or rolled back by undo."""
    preset_index: bpy.props.IntProperty(default=-1)
    is_exporting: bpy.props.BoolProperty(default=False)
    cancel_export: bpy.props.BoolProperty(default=False)
    export_progress: bpy.props.FloatProperty(name="Progress", default=0.0, min=0.0, max=1.0)
    export_status: bpy.props.StringProperty(default="")
    console_logs: bpy.props.CollectionProperty(type=BatchSTLLogLine)
    console_index: bpy.props.IntProperty(default=0)

class BatchSTLExportPreset(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Preset Name", default="New Preset", update=mark_dirty)
    preset_prefix: bpy.props.StringProperty(name="Preset Root Directory", default="", update=mark_dirty)
    collections: bpy.props.CollectionProperty(type=BatchSTLCollection)
    collection_index: bpy.props.IntProperty(name="Collection Index", default=0, update=mark_dirty)
    nodegroups: bpy.props.CollectionProperty(type=BatchSTLNodeGroup)
    last_export_time: bpy.props.FloatProperty(name="Last Export Time", default=0.0)


# ==============================================================================
# === [ 4. OPERATORS ] ===
# ==============================================================================

class BATCH_STL_OT_export_presets_json(bpy.types.Operator, ExportHelper):
    bl_idname = "batch_stl.export_presets_json"
    bl_label = "Export JSON"
    bl_description = "Export all presets to a JSON file"
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})
    def execute(self, context):
        try:
            with open(self.filepath, 'w', encoding="utf-8") as f:
                json.dump([copy_preset_to_dict(p) for p in context.scene.batch_stl_presets], f, indent=4)
            self.report({'INFO'}, f"Presets exported to {os.path.basename(self.filepath)}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to export presets: {e}")
            return {'CANCELLED'}

class BATCH_STL_OT_import_presets_json(bpy.types.Operator, ImportHelper):
    bl_idname = "batch_stl.import_presets_json"
    bl_label = "Import JSON"
    bl_description = "Import presets from a JSON file"
    bl_options = {'REGISTER'}
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})
    @inside_operator
    def execute(self, context):
        before = len(context.scene.batch_stl_presets)
        try:
            with open(self.filepath, 'r', encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                self.report({'ERROR'}, "Invalid JSON: expected a list of preset objects.")
                return {'CANCELLED'}
            for p_data in data:
                paste_preset_from_dict(context.scene.batch_stl_presets.add(), p_data)
            self.report({'INFO'}, f"Presets imported from {os.path.basename(self.filepath)}")
        except Exception as e:
            while len(context.scene.batch_stl_presets) > before:
                context.scene.batch_stl_presets.remove(len(context.scene.batch_stl_presets) - 1)
            self.report({'ERROR'}, f"Failed to import presets: {e}")
            return {'CANCELLED'}
        mark_dirty()
        return {'FINISHED'}

class BATCH_STL_OT_clear_console(bpy.types.Operator):
    bl_idname = "batch_stl.clear_console"
    bl_label = "Clear Console"
    bl_description = "Clear console logs for the current view"
    def execute(self, context):
        job = get_job(context.scene.batch_stl_preset_index)
        if job: job.console_logs.clear()
        return {'FINISHED'}

class ListActionHandler:
    """Utility class to handle standardized ADD/REMOVE/UP/DOWN/COPY/PASTE operations for UI lists."""
    @staticmethod
    def perform_action(action, lst, index, shift_pressed, clipboard_key, copy_func, paste_func):
        new_index = index
        if action == 'ADD':
            lst.add()
            new_index = len(lst) - 1
        elif action == 'REMOVE' and lst:
            if 0 <= index < len(lst):
                lst.remove(index)
                new_index = max(0, min(index, len(lst) - 1))
        elif action == 'UP' and 0 < index < len(lst):
            target = 0 if shift_pressed else index - 1
            lst.move(index, target)
            new_index = target
        elif action == 'DOWN' and 0 <= index < len(lst) - 1:
            target = len(lst) - 1 if shift_pressed else index + 1
            lst.move(index, target)
            new_index = target
        elif action == 'COPY' and lst and 0 <= index < len(lst):
            _clipboard[clipboard_key] = copy_func(lst[index])
        elif action == 'PASTE' and _clipboard.get(clipboard_key):
            paste_func(lst.add(), _clipboard[clipboard_key])
            new_index = len(lst) - 1
        return new_index

class BATCH_STL_OT_preset_actions(bpy.types.Operator):
    bl_idname = "batch_stl.preset_actions"
    bl_label = "Preset Actions"
    bl_options = {'UNDO', 'INTERNAL'}  # no REGISTER: hides the "Adjust Last Operation" redo panel
    action: bpy.props.EnumProperty(items=(('ADD', "", ""), ('REMOVE', "", ""), ('UP', "", ""), ('DOWN', "", ""), ('COPY', "", ""), ('PASTE', "", "")))
    shift_pressed: bpy.props.BoolProperty(options={'HIDDEN', 'SKIP_SAVE'}, default=False)

    @classmethod
    def description(cls, context, properties):
        act = properties.action
        if act == 'ADD': return "Add new preset"
        if act == 'REMOVE': return "Remove active preset"
        if act == 'UP': return "Move preset up (Shift: Move to top)"
        if act == 'DOWN': return "Move preset down (Shift: Move to bottom)"
        if act == 'COPY': return "Copy active preset"
        if act == 'PASTE': return "Paste preset from clipboard"
        return "Preset action"

    def invoke(self, context, event):
        self.shift_pressed = event.shift
        return self.execute(context)

    @inside_operator
    def execute(self, context):
        lst = context.scene.batch_stl_presets
        idx = context.scene.batch_stl_preset_index
        # Export state is keyed by preset index, so removing or reordering presets is only safe while nothing runs.
        reorders = self.action in {'REMOVE', 'UP', 'DOWN'}

        if reorders and is_any_exporting():
            self.report({'WARNING'}, "Cannot remove or reorder presets while an export is running.")
        else:
            context.scene.batch_stl_preset_index = ListActionHandler.perform_action(
                self.action, lst, idx, self.shift_pressed, "preset", copy_preset_to_dict, paste_preset_from_dict
            )
            if reorders: context.window_manager.batch_stl_jobs.clear()  # old logs would point at the wrong presets

        mark_dirty()
        return {'FINISHED'}

class BATCH_STL_OT_collection_actions(bpy.types.Operator):
    bl_idname = "batch_stl.collection_actions"
    bl_label = "Collection Actions"
    bl_options = {'UNDO', 'INTERNAL'}
    action: bpy.props.EnumProperty(items=(('ADD', "", ""), ('REMOVE', "", ""), ('UP', "", ""), ('DOWN', "", ""), ('COPY', "", ""), ('PASTE', "", "")))
    shift_pressed: bpy.props.BoolProperty(options={'HIDDEN', 'SKIP_SAVE'}, default=False)

    @classmethod
    def description(cls, context, properties):
        act = properties.action
        if act == 'ADD': return "Add new collection target"
        if act == 'REMOVE': return "Remove active collection"
        if act == 'UP': return "Move collection up (Shift: Move to top)"
        if act == 'DOWN': return "Move collection down (Shift: Move to bottom)"
        if act == 'COPY': return "Copy active collection target"
        if act == 'PASTE': return "Paste collection target from clipboard"
        return "Collection action"

    def invoke(self, context, event):
        self.shift_pressed = event.shift
        return self.execute(context)

    @inside_operator
    def execute(self, context):
        preset = get_active_preset(context.scene)
        if not preset: return {'CANCELLED'}

        preset.collection_index = ListActionHandler.perform_action(
            self.action, preset.collections, preset.collection_index, self.shift_pressed, "collection", copy_collection_to_dict, paste_collection_from_dict
        )

        mark_dirty()
        return {'FINISHED'}

class BATCH_STL_OT_table_action(bpy.types.Operator):
    bl_idname = "batch_stl.table_action"
    bl_label = "Table Action"
    bl_options = {'UNDO', 'INTERNAL'}

    action: bpy.props.StringProperty()
    is_global: bpy.props.BoolProperty(default=False)
    is_preset: bpy.props.BoolProperty(default=False)
    is_collection: bpy.props.BoolProperty()
    c_idx: bpy.props.IntProperty(default=-1)
    o_idx: bpy.props.IntProperty(default=-1)
    ng_idx: bpy.props.IntProperty(default=-1)
    n_idx: bpy.props.IntProperty(default=-1)
    i_idx: bpy.props.IntProperty(default=-1)
    v_idx: bpy.props.IntProperty(default=-1)
    shift_pressed: bpy.props.BoolProperty(options={'HIDDEN', 'SKIP_SAVE'}, default=False)

    @classmethod
    def description(cls, context, properties):
        act = properties.action
        if act == 'ADD_GROUP': return "Add an Override Group to this level"
        if act == 'DEL_GROUP': return "Delete this Override Group"
        if act == 'COPY_GROUP': return "Copy this Override Group to clipboard"
        if act == 'PASTE_GROUP': return "Paste an Override Group from clipboard"
        if act == 'MOVE_GROUP_UP': return "Move group up (Shift: Move directly to parent hierarchy level)"
        if act == 'MOVE_GROUP_DOWN': return "Move group down (Shift: Localize and copy group to every nested child)"
        if act == 'ADD_NODE': return "Add a Target Node filter"
        if act == 'DEL_NODE': return "Delete this Target Node filter"
        if act == 'MOVE_NODE_UP': return "Move Target Node up"
        if act == 'MOVE_NODE_DOWN': return "Move Target Node down"
        if act == 'ADD_INPUT': return "Add an Input Parameter override (Shift: Auto-populate all available socket inputs)"
        if act == 'DEL_INPUT': return "Delete this Input Parameter"
        if act == 'MOVE_INPUT_UP': return "Move Input Parameter up"
        if act == 'MOVE_INPUT_DOWN': return "Move Input Parameter down"
        if act == 'ADD_VALUE': return "Add a Value Permutation iteration"
        if act == 'DEL_VALUE': return "Delete this Value Permutation"
        if act in ('DEL_VALUE_OR_INPUT', 'DEL_VALUE_MIXED'): return "Delete Value (Deletes entire Input if it's the last iteration)"
        if act == 'MOVE_VALUE_UP': return "Move Value up"
        if act == 'MOVE_VALUE_DOWN': return "Move Value down"
        if act == 'VALUE_ACTION': return "Add Permutation (Shift: Toggle Sweep range mode)"
        if act == 'TOGGLE_VALUE_USE_DIR': return "Toggle sub-directory folder structuring for this iteration"
        if act == 'TOGGLE_VALUE_USE_TAG': return "Toggle dynamic filename tagging for this iteration"
        if act == 'TOGGLE_COLLECTION_USE_TAG': return "Toggle collection-level filename prefix/suffix tag"
        if act == 'TOGGLE_OBJECT_EXPORT': return "Toggle object active export state"
        return "Perform structural table action"

    def invoke(self, context, event):
        self.shift_pressed = event.shift
        return self.execute(context)

    def _resolve_context_list(self, context, preset):
        if self.is_global: return context.scene.batch_stl_global_nodegroups
        elif self.is_preset: return preset.nodegroups

        active_col = get_active_collection(preset)
        if not active_col: return None
        if self.is_collection: return active_col.nodegroups

        active_obj = get_active_object(active_col)
        return active_obj.nodegroups if active_obj else None

    def _handle_group_action(self, ng_list, context, preset):
        if self.action == 'ADD_GROUP':
            ng = ng_list.add()
            node = ng.nodes.add(); node.name = "<Modifier Interface>"
            node.inputs.add().values.add()
        elif self.action == 'DEL_GROUP' and 0 <= self.ng_idx < len(ng_list):
            ng_list.remove(self.ng_idx)
        elif self.action == 'COPY_GROUP' and 0 <= self.ng_idx < len(ng_list):
            global _clipboard
            _clipboard["nodegroup"] = copy_ng_to_dict(ng_list[self.ng_idx])
        elif self.action == 'PASTE_GROUP' and _clipboard.get("nodegroup"):
            paste_ng_from_dict(ng_list.add(), _clipboard["nodegroup"])
        elif self.action == 'MOVE_GROUP_UP' and 0 <= self.ng_idx < len(ng_list):
            if self.shift_pressed:
                src_ng = ng_list[self.ng_idx]
                dst = None
                if self.is_preset: dst = context.scene.batch_stl_global_nodegroups
                elif self.is_collection: dst = preset.nodegroups
                elif not self.is_global:
                    col = get_active_collection(preset)
                    dst = col.nodegroups if col else None
                if dst is not None:
                    paste_ng_from_dict(dst.add(), copy_ng_to_dict(src_ng))
                    ng_list.remove(self.ng_idx)
            elif self.ng_idx > 0: ng_list.move(self.ng_idx, self.ng_idx - 1)
        elif self.action == 'MOVE_GROUP_DOWN' and 0 <= self.ng_idx < len(ng_list):
            if self.shift_pressed:
                src_ng = ng_list[self.ng_idx]
                copied_data = copy_ng_to_dict(src_ng)
                pushed = False
                if self.is_global:
                    for p in context.scene.batch_stl_presets:
                        paste_ng_from_dict(p.nodegroups.add(), copied_data)
                        pushed = True
                elif self.is_preset:
                    for c in preset.collections:
                        paste_ng_from_dict(c.nodegroups.add(), copied_data)
                        pushed = True
                elif self.is_collection:
                    col = get_active_collection(preset)
                    if col:
                        for o in col.objects:
                            paste_ng_from_dict(o.nodegroups.add(), copied_data)
                            pushed = True
                if pushed:
                    ng_list.remove(self.ng_idx)
            elif self.ng_idx < len(ng_list) - 1:
                ng_list.move(self.ng_idx, self.ng_idx + 1)

    def _handle_node_action(self, ng_list):
        if not (0 <= self.ng_idx < len(ng_list)): return
        nodes = ng_list[self.ng_idx].nodes
        if self.action == 'ADD_NODE':
            node = nodes.add(); node.name = "<Modifier Interface>"
            node.inputs.add().values.add()
        elif self.action == 'DEL_NODE' and 0 <= self.n_idx < len(nodes):
            nodes.remove(self.n_idx)
        elif self.action == 'MOVE_NODE_UP' and 0 < self.n_idx < len(nodes):
            nodes.move(self.n_idx, self.n_idx - 1)
        elif self.action == 'MOVE_NODE_DOWN' and 0 <= self.n_idx < len(nodes) - 1:
            nodes.move(self.n_idx, self.n_idx + 1)

    def _handle_input_action(self, ng_list):
        if not (0 <= self.ng_idx < len(ng_list)): return
        ng = ng_list[self.ng_idx]
        if not (0 <= self.n_idx < len(ng.nodes)): return
        node = ng.nodes[self.n_idx]
        inputs = node.inputs
        if self.action == 'ADD_INPUT':
            ng_ptr = bpy.data.node_groups.get(ng.group_name)
            if self.shift_pressed and ng_ptr:
                source_inputs = []
                if (not node.name or node.name == "<Modifier Interface>") and hasattr(ng_ptr, "interface"):
                    source_inputs = [item.name for item in ng_ptr.interface.items_tree if getattr(item, "item_type", "SOCKET") == 'SOCKET' and getattr(item, "in_out", "INPUT") == 'INPUT']
                elif node.name:
                    target_n = ng_ptr.nodes.get(clean_node_name(node.name))
                    if target_n: source_inputs = [i.name for i in target_n.inputs if not getattr(i, "is_unavailable", False) and not getattr(i, "hide", False)]

                if source_inputs:
                    existing = {i.name for i in node.inputs}
                    for s_name in source_inputs:
                        if s_name and s_name not in existing:
                            inp = node.inputs.add()
                            inp.name = s_name
                            inp.values.add()
                    return
            inputs.add().values.add()
        elif self.action == 'DEL_INPUT' and 0 <= self.i_idx < len(inputs):
            inputs.remove(self.i_idx)
        elif self.action == 'MOVE_INPUT_UP' and 0 < self.i_idx < len(inputs):
            inputs.move(self.i_idx, self.i_idx - 1)
        elif self.action == 'MOVE_INPUT_DOWN' and 0 <= self.i_idx < len(inputs) - 1:
            inputs.move(self.i_idx, self.i_idx + 1)

    def _handle_value_action(self, ng_list):
        if not (0 <= self.ng_idx < len(ng_list)): return
        ng = ng_list[self.ng_idx]
        if not (0 <= self.n_idx < len(ng.nodes)): return
        node = ng.nodes[self.n_idx]
        if not (0 <= self.i_idx < len(node.inputs)): return
        inp_obj = node.inputs[self.i_idx]
        vals = inp_obj.values
        if self.action == 'DEL_VALUE' and 0 <= self.v_idx < len(vals):
            vals.remove(self.v_idx)
        elif self.action == 'DEL_VALUE_MIXED':
            if len(vals) > 1 and 0 <= self.v_idx < len(vals):
                vals.remove(self.v_idx)
            else:
                node.inputs.remove(self.i_idx)
        elif self.action == 'MOVE_VALUE_UP' and 0 < self.v_idx < len(vals):
            vals.move(self.v_idx, self.v_idx - 1)
        elif self.action == 'MOVE_VALUE_DOWN' and 0 <= self.v_idx < len(vals) - 1:
            vals.move(self.v_idx, self.v_idx + 1)
        elif self.action in ['ADD_VALUE', 'TOGGLE_SWEEP', 'VALUE_ACTION']:
            if self.v_idx < 0:
                vals.add()
            elif 0 <= self.v_idx < len(vals):
                val = vals[self.v_idx]
                if not val.use_sweep:
                    if self.shift_pressed:
                        val.use_sweep = True
                        for j in reversed(range(len(vals))):
                            if j != self.v_idx: vals.remove(j)
                    else: vals.add()
                else:
                    val.use_sweep = False
                    if self.shift_pressed and inp_obj.override_type in ['FLOAT', 'INT', 'MENU', 'BOOLEAN', 'STRING']:
                        ng_obj = ng_list[self.ng_idx]
                        ng_ptr = bpy.data.node_groups.get(ng_obj.group_name)
                        node_obj = ng_obj.nodes[self.n_idx]
                        target = 'MODIFIER' if not node_obj.name or node_obj.name == "<Modifier Interface>" else 'NODE'
                        temp_inp = MockInput(inp_obj, val, is_temp=True)
                        parsed_vals = parse_sweep_values(MockOverride(target, ng_ptr, node_obj.name, [temp_inp]), temp_inp)
                        if parsed_vals:
                            for p_idx, p_val in enumerate(parsed_vals):
                                v = val if p_idx == 0 else vals.add()
                                v.use_sweep = False
                                if inp_obj.override_type == 'FLOAT': v.value_float = p_val
                                elif inp_obj.override_type == 'INT': v.value_int = p_val
                                elif inp_obj.override_type == 'MENU': v.value_menu = str(p_val)
                                elif inp_obj.override_type == 'BOOLEAN': v.value_bool = bool(p_val)
                                elif inp_obj.override_type == 'STRING': v.value_string = str(p_val)

    @inside_operator
    def execute(self, context):
        preset = get_active_preset(context.scene)
        if not preset: return {'CANCELLED'}

        try:
            if self.action == 'TOGGLE_COLLECTION_USE_TAG':
                if 0 <= self.c_idx < len(preset.collections): preset.collections[self.c_idx].use_tag = not preset.collections[self.c_idx].use_tag
            elif self.action == 'TOGGLE_OBJECT_EXPORT':
                active_col = get_active_collection(preset)
                if active_col and 0 <= self.o_idx < len(active_col.objects): active_col.objects[self.o_idx].export = not active_col.objects[self.o_idx].export
            else:
                ng_list = self._resolve_context_list(context, preset)
                if ng_list is None: return {'CANCELLED'}

                if self.action in ('TOGGLE_VALUE_USE_DIR', 'TOGGLE_VALUE_USE_TAG'):
                    if 0 <= self.ng_idx < len(ng_list) and 0 <= self.n_idx < len(ng_list[self.ng_idx].nodes) and 0 <= self.i_idx < len(ng_list[self.ng_idx].nodes[self.n_idx].inputs) and 0 <= self.v_idx < len(ng_list[self.ng_idx].nodes[self.n_idx].inputs[self.i_idx].values):
                        val = ng_list[self.ng_idx].nodes[self.n_idx].inputs[self.i_idx].values[self.v_idx]
                        if self.action == 'TOGGLE_VALUE_USE_DIR': val.use_dir = not val.use_dir
                        else: val.use_tag = not val.use_tag
                elif 'GROUP' in self.action: self._handle_group_action(ng_list, context, preset)
                elif 'NODE' in self.action: self._handle_node_action(ng_list)
                elif 'INPUT' in self.action: self._handle_input_action(ng_list)
                elif 'VALUE' in self.action: self._handle_value_action(ng_list)
        except IndexError:
            self.report({'WARNING'}, "UI Sync Error: List mutated unexpectedly. Please try again.")
            return {'CANCELLED'}

        mark_dirty()
        return {'FINISHED'}

class BATCH_STL_OT_toggle_dir_tree(bpy.types.Operator):
    bl_idname = "batch_stl.toggle_dir_tree"
    bl_label = "Toggle Directory Tree"
    bl_options = {'INTERNAL'}
    bl_description = "Toggle directory tree expansion"
    dir_path: bpy.props.StringProperty()
    def execute(self, context):
        scene = context.scene
        try: collapsed = json.loads(scene.batch_stl_collapsed_dirs)
        except Exception: collapsed = []
        if self.dir_path in collapsed: collapsed.remove(self.dir_path)
        else: collapsed.append(self.dir_path)
        scene.batch_stl_collapsed_dirs = json.dumps(collapsed)
        return {'FINISHED'}

class BATCH_STL_OT_cancel_export(bpy.types.Operator):
    bl_idname = "batch_stl.cancel_export"
    bl_label = "Cancel Export"
    bl_description = "Cancel the active batch export"
    preset_index: bpy.props.IntProperty(default=-1)
    def execute(self, context):
        job = get_job(self.preset_index)
        if job and job.is_exporting:
            job.cancel_export = True
            log_to_console(job, "[!] Export cancelled manually.")
        return {'FINISHED'}

class EXPORT_OT_batch_stl_multi(bpy.types.Operator):
    bl_idname = "export_scene.batch_stl_multi"
    bl_label = "Export"
    bl_description = "Start batch STL export"
    bl_options = {"REGISTER"}
    preset_index: bpy.props.IntProperty(default=-1)

    @classmethod
    def poll(cls, context): return len(context.scene.batch_stl_presets) > 0

    def invoke(self, context, event):
        self._timer = self.process = None
        self.total_operations, self.current_op = 1, 0
        self.export_start_time = time.perf_counter()
        scene = context.scene

        self.preset_idx = self.preset_index if self.preset_index >= 0 else scene.batch_stl_preset_index
        if self.preset_idx < 0 or self.preset_idx >= len(scene.batch_stl_presets): return {"CANCELLED"}
        context.scene.batch_stl_preset_index = self.preset_idx
        # Pointers are only valid during invoke: undo and file loads invalidate them, so modal()/cleanup() re-resolve by index.
        self.preset = scene.batch_stl_presets[self.preset_idx]
        job = get_job(self.preset_idx, create=True)

        if job.is_exporting: return {'CANCELLED'}
        if not scene.batch_stl_root_dir:
            self.report({'ERROR'}, "Missing Root Directory")
            return {"CANCELLED"}
        if scene.batch_stl_root_dir.startswith("//") and not bpy.data.is_saved:
            self.report({'ERROR'}, "Please save the .blend file before exporting to a relative path (//)")
            return {"CANCELLED"}
        if not is_preset_setup_valid(scene, self.preset):
            self.report({'ERROR'}, "Improper setup: One or more override fields are missing or invalid.")
            return {"CANCELLED"}

        if context.scene.batch_stl_show_console: context.scene.batch_stl_info_tab = 'LOG'
        job.console_logs.clear()

        objects_to_export_directly, objects_needing_headless = [], []
        preset_root = os.path.normpath(bpy.path.abspath(scene.batch_stl_root_dir))
        global_ovrs = get_flat_overrides(scene.batch_stl_global_nodegroups, "GLOBAL")
        preset_ovrs = global_ovrs + get_flat_overrides(self.preset.nodegroups, "PRESET")

        for c in self.preset.collections:
            c_ptr = bpy.data.collections.get(c.collection_name)
            if not c_ptr or is_collection_excluded(bpy.context, c_ptr): continue
            sync_collection_objects(c, c_ptr)
            c_pinned_ovrs = get_flat_overrides(c.nodegroups, "COLLECTION")
            for obj_prop in c.objects:
                if not obj_prop.export: continue
                bl_obj = c_ptr.all_objects.get(obj_prop.name)
                if not bl_obj or bl_obj.hide_viewport or bl_obj.type not in SUPPORTED_OBJECT_TYPES: continue

                obj_ovrs = get_flat_overrides(obj_prop.nodegroups, "OBJECT")
                if not preset_ovrs and not c_pinned_ovrs and not obj_ovrs: objects_to_export_directly.append((c, obj_prop, bl_obj))
                else: objects_needing_headless.append((c, obj_prop, bl_obj))

        if objects_to_export_directly:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            instance_arrays = collect_instance_arrays(depsgraph, [item[2] for item in objects_to_export_directly])
            log_to_console(job, f"=== STARTING NATIVE DIRECT EXPORT ({len(objects_to_export_directly)} Objects) ===")
            for c, obj_prop, bl_obj in objects_to_export_directly:
                t_dir_start = time.perf_counter()
                full_dir_parts = build_export_dir_parts(self.preset.preset_prefix, c.sub_path, obj_prop.sub_path)
                out_dir = os.path.normpath(os.path.join(preset_root, *full_dir_parts)) if full_dir_parts else preset_root
                os.makedirs(out_dir, exist_ok=True)

                filename = format_export_filename(bl_obj.name, obj_prop.tag, getattr(c, 'use_tag', False), c.tag)
                filepath = os.path.join(out_dir, filename)

                num_tris = write_object_stl(filepath, bl_obj, depsgraph, instance_arrays.get(bl_obj.name_full, ()))
                result_text = filepath if num_tris else f"SKIPPED (no geometry): {filepath}"

                log_to_console(job, f"{self.preset.name} | {c.collection_name} | {bl_obj.name} | Permutation 1/1 | Batch 1/1\n  └─ {result_text} | {time.perf_counter() - t_dir_start:.2f} s")

        if not objects_needing_headless:
            self.preset.last_export_time = time.perf_counter() - self.export_start_time
            log_to_console(job, f"=== BATCH EXPORT COMPLETE ({self.preset.last_export_time:.4f}s) ===")
            self.report({'INFO'}, f"Batch Export {self.preset.name} Complete in {self.preset.last_export_time:.2f}s.")
            redraw_sidebars(context)
            self.cleanup(context)
            return {'FINISHED'}

        job.export_status = "Spawning Worker... (0.0s)"
        t_spawn_start = time.perf_counter()
        self.temp_dir = tempfile.mkdtemp(prefix="fast_batch_stl_")
        self.temp_blend = os.path.join(self.temp_dir, "batch_stl_export_temp.blend")
        self.job_json = os.path.join(self.temp_dir, "job.json")

        bpy.ops.wm.save_as_mainfile(filepath=self.temp_blend, copy=True, compress=False)
        with open(self.job_json, 'w', encoding="utf-8") as f: json.dump({"preset_index": self.preset_idx, "root_dir": bpy.path.abspath(scene.batch_stl_root_dir), "start_time": time.time(), "skip_direct": True}, f)

        # The worker runs this very file as a script (see the __main__ block at the bottom).
        # --factory-startup disables Python auto-run, so mirror the user's setting to keep scripted drivers working.
        worker_args = [bpy.app.binary_path, "--factory-startup"]
        if context.preferences.filepaths.use_scripts_auto_execute: worker_args.append("--enable-autoexec")
        worker_args += ["-b", self.temp_blend, "-P", __file__, "--", "--batch-stl-headless", self.job_json]

        try:
            sub_env = dict(os.environ, PYTHONUNBUFFERED="1")
            self.process = subprocess.Popen(worker_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=sub_env)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to spawn headless Blender: {e}")
            self.cleanup(context)
            return {'CANCELLED'}

        log_to_console(job, f"=== INITIATING HEADLESS EXPORT '{self.preset.name}' [{time.perf_counter() - t_spawn_start:.4f}s Boot] ===")

        self.q = queue.Queue()
        def enqueue_output(out, q):
            for line in iter(out.readline, ''): q.put(line)
            out.close()
        self.t = threading.Thread(target=enqueue_output, args=(self.process.stdout, self.q)); self.t.daemon = True; self.t.start()

        job.is_exporting, job.cancel_export, job.export_progress = True, False, 0.0
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _preset(self):
        presets = bpy.context.scene.batch_stl_presets
        return presets[self.preset_idx] if 0 <= self.preset_idx < len(presets) else None

    def _job(self):
        return get_job(self.preset_idx)

    def _drain_output(self, job):
        """Process queued worker output. Returns True once the worker reported BATCH_STL_DONE."""
        while True:
            try: line = self.q.get_nowait().rstrip('\r\n')
            except queue.Empty: return False
            if line.startswith("BATCH_STL_TOTAL:"):
                try: self.total_operations = int(line.split(":")[1])
                except Exception: pass
            elif line.startswith("BATCH_STL_PROGRESS:"):
                try:
                    self.current_op = int(line.split(":")[1])
                    job.export_progress = self.current_op / max(1, self.total_operations)
                except Exception: pass
            elif line.startswith("BATCH_STL_DONE"):
                return True
            elif line:
                log_to_console(job, line)

    def modal(self, context, event):
        try:
            preset, job = self._preset(), self._job()
            if preset is None or job is None:  # presets were replaced underneath us (undo / file load)
                self.cleanup(context)
                return {'CANCELLED'}

            if event.type == 'Z' and event.value == 'PRESS' and (event.ctrl or event.oskey):
                # Undo could shift preset indices, which the running job is keyed by.
                job.cancel_export = True
                log_to_console(job, "[!] Undo pressed: cancelling the running export.")
                return {'RUNNING_MODAL'}

            if job.cancel_export:
                self.cleanup(context)
                self.report({'WARNING'}, f"Export cancelled for {preset.name}.")
                return {'CANCELLED'}

            if event.type == 'TIMER':
                elapsed = time.perf_counter() - self.export_start_time
                finished = self._drain_output(job)
                worker_exited = self.process.poll() is not None
                if worker_exited and not finished:
                    # The worker prints BATCH_STL_DONE right before exiting; let the reader thread flush before judging.
                    self.t.join(timeout=2.0)
                    finished = self._drain_output(job)

                if finished:
                    self.cleanup(context)
                    preset.last_export_time = time.perf_counter() - self.export_start_time
                    log_to_console(job, f"=== BATCH EXPORT COMPLETE ({preset.last_export_time:.4f}s) ===")
                    self.report({'INFO'}, f"Batch Export {preset.name} Complete in {preset.last_export_time:.2f}s.")
                    redraw_sidebars(context)
                    return {'FINISHED'}

                if worker_exited:
                    exit_code = self.process.returncode
                    self.cleanup(context)
                    log_to_console(job, f"[!] CRASH DETECTED: Worker exited with code {exit_code} before finishing.")
                    self.report({'ERROR'}, f"Background worker crashed for preset {preset.name}.")
                    redraw_sidebars(context)
                    return {'CANCELLED'}

                job.export_status = f"Obj {self.current_op}/{self.total_operations} | {elapsed:.1f}s" if (self.total_operations > 1 or self.current_op > 0) else f"Spawning Worker... ({elapsed:.1f}s)"
                redraw_sidebars(context)
        except Exception:
            traceback.print_exc()
            self.cleanup(context)
            self.report({'ERROR'}, "Unexpected error during batch export.")
            return {'CANCELLED'}
        return {'PASS_THROUGH'}

    def cancel(self, context):
        # Called by Blender when the modal is torn down from outside (e.g. loading another file).
        self.cleanup(context)

    def cleanup(self, context=None):
        if context and getattr(self, '_timer', None):
            try: context.window_manager.event_timer_remove(self._timer)
            except Exception: pass
            self._timer = None
        if hasattr(self, 'preset_idx'):
            try:
                job = self._job()
                if job: job.is_exporting, job.cancel_export, job.export_progress, job.export_status = False, False, 0.0, ""
            except Exception: pass
        if getattr(self, 'process', None):
            try:
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
            except Exception: pass
        if getattr(self, 't', None) and self.t.is_alive():
            self.t.join(timeout=0.5)
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

# ==============================================================================
# === [ 5. UI LISTS & PANELS ] ===
# ==============================================================================

class BATCH_STL_UL_presets(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        job = get_job(index)
        row = layout.row(align=True)
        prop_row = row.row(align=True)
        prop_row.enabled = not is_any_exporting()
        prop_row.prop(item, "name", text="", emboss=False)

        metrics = _ui_cache.get("preset_metrics", {}).get(index, {"has_ovr": False, "has_perm": False})
        icon_row = prop_row.row(align=True)
        icon_row.alignment = 'RIGHT'
        icon_row.label(text="", icon=ICONS['SWEEP'] if metrics["has_perm"] else ICONS['BLANK'])
        icon_row.label(text="", icon=ICONS['NODE'] if metrics["has_ovr"] else ICONS['BLANK'])
        prop_row.prop(item, "preset_prefix", text="", emboss=False, icon=ICONS['DIR'])

        if job and job.is_exporting:
            row.prop(job, "export_progress", text=job.export_status, slider=True)
            row.operator("batch_stl.cancel_export", text="", icon=ICONS['CANCEL']).preset_index = index
        else:
            row.operator("export_scene.batch_stl_multi", text="", icon=ICONS['EXPORT']).preset_index = index

class BATCH_STL_UL_collections(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop_search(item, "collection_name", bpy.data, "collections", text="", icon=ICONS['COLLECTION'])
        row.separator(factor=0.5)
        sub_row = row.row(align=True)
        op = sub_row.operator("batch_stl.table_action", text="", icon=ICONS['TAG'], depress=item.use_tag)
        op.action = 'TOGGLE_COLLECTION_USE_TAG'; op.c_idx = index
        sub_row.separator(factor=0.5)
        sub_row.row(align=True).prop(item, "tag", text="", emboss=False)
        row.prop(item, "sub_path", text="", emboss=False, icon=ICONS['DIR'])

class BATCH_STL_UL_objects(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        split = layout.split(factor=0.45)
        row = split.row(align=True)
        op = row.operator("batch_stl.table_action", text="", icon=ICONS['CHECK_ON'] if item.export else ICONS['CHECK_OFF'], emboss=False)
        op.action = 'TOGGLE_OBJECT_EXPORT'; op.o_idx = index
        row.label(text=item.name)

        tools = split.row(align=True)
        tools.label(text="", icon=ICONS['TAG'])
        tools.prop(item, "tag", text="", emboss=False)
        tools.separator(factor=0.5)
        tools.label(text="", icon=ICONS['DIR'])
        tools.prop(item, "sub_path", text="", emboss=False)

class BATCH_STL_UL_console_logs(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.text)

def draw_inline_controls(layout, operator_id, use_clipboard=False):
    row = layout.row(align=True)
    row.operator(operator_id, icon=ICONS['ADD'], text="").action = 'ADD'
    row.operator(operator_id, icon=ICONS['DEL'], text="").action = 'REMOVE'
    row.operator(operator_id, icon=ICONS['UP'], text="").action = 'UP'
    row.operator(operator_id, icon=ICONS['DOWN'], text="").action = 'DOWN'
    if use_clipboard:
        row.operator(operator_id, icon=ICONS['COPY'], text="").action = 'COPY'
        row.operator(operator_id, icon=ICONS['PASTE'], text="").action = 'PASTE'

def draw_stats_table(parent_layout, stats_list):
    box = parent_layout.box()
    row = box.row(align=True)
    row.alignment = 'CENTER'
    for i, (val, icon) in enumerate(stats_list):
        if i > 0: row.separator(factor=2.0)
        row.label(text=str(val), icon=icon)

def draw_overrides_table(layout, scene, nodegroups, is_collection, is_open_prop, title_text, is_preset=False, is_global=False, is_locked=False):
    # Setup inline helper to simplify conditional operator generation drastically
    def draw_op(parent, action, icon, depress=False, ng_idx=-1, n_idx=-1, i_idx=-1, v_idx=-1):
        op = parent.operator("batch_stl.table_action", text="", icon=icon, depress=depress)
        op.action, op.is_collection, op.is_preset, op.is_global = action, is_collection, is_preset, is_global
        op.ng_idx, op.n_idx, op.i_idx, op.v_idx = ng_idx, n_idx, i_idx, v_idx
        return op

    def draw_input_name(parent, inp, ng, node):
        ng_ptr = bpy.data.node_groups.get(ng.group_name)
        is_mod = not node.name or node.name == "<Modifier Interface>"
        is_valid = is_override_input_valid(ng_ptr, node, inp)
        row = parent.row(align=True)
        row.alert = not is_valid
        if is_mod and ng_ptr and hasattr(ng_ptr, "interface"):
            row.prop_search(inp, "name", ng_ptr.interface, "items_tree", text="")
        elif not is_mod and ng_ptr and node.name:
            target_n = ng_ptr.nodes.get(clean_node_name(node.name))
            if target_n: row.prop_search(inp, "name", target_n, "inputs", text="")
            else: row.prop(inp, "name", text="")
        else:
            row.prop(inp, "name", text="")

    box = layout.box()

    header_row = box.row()
    header_row.enabled = not is_locked
    is_open = getattr(scene, is_open_prop)
    icon_open = ICONS['DOWN'] if is_open else ICONS['RIGHT']
    header_row.prop(scene, is_open_prop, text="", icon=icon_open, emboss=False)

    icon_header = ICONS['GLOBAL'] if is_global else (ICONS['PRESET'] if is_preset else (ICONS['COLLECTION'] if is_collection else ICONS['OBJECT']))
    header_row.label(text="", icon=ICONS['OVR'])
    header_row.label(text=title_text, icon=icon_header)

    op_row = header_row.row(align=True)
    op_row.enabled = not is_locked
    draw_op(op_row, 'ADD_GROUP', ICONS['ADD'])
    draw_op(op_row, 'PASTE_GROUP', ICONS['PASTE'])

    if not is_open:
        return

    content_col = box.column()
    content_col.enabled = not is_locked

    if len(nodegroups) == 0:
        content_col.label(text="No overrides defined.")
        return

    for ng_idx, ng in enumerate(nodegroups):
        ng_box = content_col.box()
        ng_layout = ng_box.column()
        ng_row = ng_layout.row(align=True)

        draw_op(ng_row, 'ADD_NODE', ICONS['ADD'], ng_idx=ng_idx)
        ng_sub = ng_row.row(align=True)
        ng_sub.alert = not is_override_group_valid(ng)
        ng_sub.prop_search(ng, "group_name", bpy.data, "node_groups", text="")
        draw_op(ng_row, 'MOVE_GROUP_UP', ICONS['UP'], ng_idx=ng_idx)
        draw_op(ng_row, 'MOVE_GROUP_DOWN', ICONS['DOWN'], ng_idx=ng_idx)
        draw_op(ng_row, 'COPY_GROUP', ICONS['COPY'], ng_idx=ng_idx)
        draw_op(ng_row, 'DEL_GROUP', ICONS['DEL'], ng_idx=ng_idx)

        if not ng.nodes:
            continue

        ng_ptr = bpy.data.node_groups.get(ng.group_name)

        n_split = ng_layout.split(factor=0.03)
        n_split.column()
        nodes_col = n_split.column()
        nodes_box = nodes_col.box() if len(ng.nodes) > 1 else nodes_col
        nodes_layout = nodes_box.column()

        for n_idx, node in enumerate(ng.nodes):
            node_container = nodes_layout.box()
            node_layout = node_container.column()

            n_row = node_layout.row(align=True)
            draw_op(n_row, 'ADD_INPUT', ICONS['ADD'], ng_idx=ng_idx, n_idx=n_idx)
            n_sub = n_row.row(align=True)
            n_sub.alert = not is_override_node_valid(ng_ptr, node)
            n_sub.prop(node, "name", text="", icon=ICONS['NODE'])

            if len(ng.nodes) > 1:
                draw_op(n_row, 'MOVE_NODE_UP', ICONS['UP'], ng_idx=ng_idx, n_idx=n_idx)
                draw_op(n_row, 'MOVE_NODE_DOWN', ICONS['DOWN'], ng_idx=ng_idx, n_idx=n_idx)
                draw_op(n_row, 'DEL_NODE', ICONS['DEL'], ng_idx=ng_idx, n_idx=n_idx)

            if not node.inputs:
                continue

            i_split = node_layout.split(factor=0.03)
            i_split.column()
            inputs_col = i_split.column()
            inputs_box = inputs_col.box()
            inputs_layout = inputs_box.column()

            for i_idx, inp in enumerate(node.inputs):
                input_layout = inputs_layout.column()
                # Wrap in array to ensure rendering block triggers at least once even if 'values' logic is empty
                values = inp.values if inp.values else [None]

                for v_idx, val in enumerate(values):
                    i_first = (v_idx == 0)
                    i_row = input_layout.row(align=True)

                    s_main = i_row.split(factor=0.35, align=False)
                    c_inp = s_main.row(align=True)

                    if i_first:
                        action_icon = ICONS['SWEEP'] if val and getattr(val, "use_sweep", False) else ICONS['ADD']
                        draw_op(c_inp, 'VALUE_ACTION', action_icon, depress=(action_icon == ICONS['SWEEP']),
                                ng_idx=ng_idx, n_idx=n_idx, i_idx=i_idx, v_idx=v_idx if val else -1)
                        draw_input_name(c_inp, inp, ng, node)
                    else:
                        c_inp.alignment = 'RIGHT'

                    s_val = s_main.split(factor=0.5, align=False)
                    c_val = s_val.row(align=True)
                    c_dir = s_val.row(align=True)

                    # Handing unpopulated values safely
                    if val is None:
                        if len(node.inputs) > 1:
                            draw_op(c_dir, 'MOVE_INPUT_UP', ICONS['UP'], ng_idx=ng_idx, n_idx=n_idx, i_idx=i_idx)
                            draw_op(c_dir, 'MOVE_INPUT_DOWN', ICONS['DOWN'], ng_idx=ng_idx, n_idx=n_idx, i_idx=i_idx)
                        draw_op(c_dir, 'DEL_INPUT', ICONS['DEL'], ng_idx=ng_idx, n_idx=n_idx, i_idx=i_idx)
                        continue

                    # Render Value Properties
                    val_valid = is_override_val_valid(inp, val, ng_ptr, node)
                    c_val_prop = c_val.row(align=True)
                    c_val_prop.alert = not val_valid
                    if getattr(val, "use_sweep", False):
                        if inp.override_type == 'FLOAT':
                            c_val_prop.prop(val, "sweep_start_float", text="")
                            c_val_prop.prop(val, "sweep_step_float", text="")
                            c_val_prop.prop(val, "sweep_count_float", text="")
                        elif inp.override_type == 'INT':
                            c_val_prop.prop(val, "sweep_start_int", text="")
                            c_val_prop.prop(val, "sweep_step_int", text="")
                            c_val_prop.prop(val, "sweep_count_int", text="")
                        elif inp.override_type == 'STRING':
                            c_val_prop.prop(val, "sweep_range", text="")
                        elif inp.override_type in ['BOOLEAN', 'MENU']:
                            sub = c_val_prop.row(align=True); sub.active = False
                            if inp.override_type == 'BOOLEAN':
                                sub.label(text="True & False")
                            else:
                                n_items = len(get_menu_switch_items(ng_ptr, node.name, inp.name)) if ng_ptr else 0
                                sub.label(text=f"{n_items} values")
                    else:
                        prop_map = {'BOOLEAN': "value_bool", 'INT': "value_int", 'FLOAT': "value_float", 'STRING': "value_string", 'MENU': "value_menu"}
                        prop_name = prop_map.get(inp.override_type)
                        if prop_name:
                            kwargs = {"text": "True" if val.value_bool else "False", "toggle": True} if prop_name == "value_bool" else {"text": ""}
                            c_val_prop.prop(val, prop_name, **kwargs)
                        else:
                            c_val_prop.label(text="Unsupported socket type", icon=ICONS['ERROR'])

                    # Render Directory/Tag controls for permutations
                    is_permutation = len(inp.values) > 1 or any(getattr(v, "use_sweep", False) for v in inp.values)
                    if is_permutation:
                        draw_op(c_dir, 'TOGGLE_VALUE_USE_DIR', ICONS['DIR'], depress=val.use_dir, ng_idx=ng_idx, n_idx=n_idx, i_idx=i_idx, v_idx=v_idx)
                        draw_op(c_dir, 'TOGGLE_VALUE_USE_TAG', ICONS['TAG'], depress=val.use_tag, ng_idx=ng_idx, n_idx=n_idx, i_idx=i_idx, v_idx=v_idx)
                        c_dir.prop(val, "tag", text="")

                    # Render Action Buttons
                    if i_first:
                        if len(node.inputs) > 1:
                            draw_op(c_dir, 'MOVE_INPUT_UP', ICONS['UP'], ng_idx=ng_idx, n_idx=n_idx, i_idx=i_idx)
                            draw_op(c_dir, 'MOVE_INPUT_DOWN', ICONS['DOWN'], ng_idx=ng_idx, n_idx=n_idx, i_idx=i_idx)
                        draw_op(c_dir, 'DEL_VALUE_MIXED', ICONS['DEL'], ng_idx=ng_idx, n_idx=n_idx, i_idx=i_idx, v_idx=0)
                    else:
                        if len(inp.values) > 1:
                            draw_op(c_dir, 'MOVE_VALUE_UP', ICONS['UP'], ng_idx=ng_idx, n_idx=n_idx, i_idx=i_idx, v_idx=v_idx)
                            draw_op(c_dir, 'MOVE_VALUE_DOWN', ICONS['DOWN'], ng_idx=ng_idx, n_idx=n_idx, i_idx=i_idx, v_idx=v_idx)
                        draw_op(c_dir, 'DEL_VALUE', ICONS['DEL'], ng_idx=ng_idx, n_idx=n_idx, i_idx=i_idx, v_idx=v_idx)


class VIEW3D_PT_batch_export_stl_main(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Combi Export"
    bl_label = "Combi Export"

    def draw_header(self, context):
        self.layout.label(text="", icon=ICONS['EXPORT'])

    def draw_header_preset(self, context):
        layout = self.layout
        scene = context.scene
        any_exporting = is_any_exporting()

        row = layout.row(align=True)
        sub_row = row.row(align=True)
        sub_row.enabled = not any_exporting
        sub_row.operator("batch_stl.import_presets_json", text="", icon=ICONS['IMPORT'])
        sub_row.operator("batch_stl.export_presets_json", text="", icon=ICONS['EXPORT'])
        row.prop(scene, "batch_stl_show_console", text="", icon=ICONS['INFO'], toggle=True)
        row.separator()

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        any_exporting = is_any_exporting()

        dir_col = layout.column()
        dir_col.enabled = not any_exporting
        dir_col.prop(scene, "batch_stl_root_dir")

        active_preset = get_active_preset(scene)
        stats = _ui_cache.get("stats", {})

        if scene.batch_stl_show_console:
            info_box = layout.box()

            tab_row = info_box.row()
            tab_row.prop(scene, "batch_stl_info_tab", expand=True)

            if scene.batch_stl_info_tab == 'LOG':
                active_job = get_job(scene.batch_stl_preset_index)
                if active_job:
                    info_box.template_list("BATCH_STL_UL_console_logs", "", active_job, "console_logs", active_job, "console_index", rows=6)
                    clear_col = info_box.column()
                    clear_col.enabled = not any_exporting
                    clear_col.operator("batch_stl.clear_console", text="Clear Log", icon=ICONS['DEL'])
                else:
                    info_box.label(text="No export log for this preset yet.", icon=ICONS['INFO'])
                info_box.prop(scene, "batch_stl_verbose_console", toggle=True, icon=ICONS['CONSOLE'])

            elif scene.batch_stl_info_tab == 'TREE':
                tree_tools = info_box.row()
                tree_tools.prop(scene, "batch_stl_info_global", text="Global Tree View", toggle=True, icon=ICONS['GLOBAL'])

                tree_dict, duplicates = _ui_cache.get("tree", ({}, set()))
                if duplicates:
                    warn_box = info_box.box()
                    warn_row = warn_box.row()
                    warn_row.label(text=f"WARNING: {len(duplicates)} naming collisions detected! Files will be overwritten.", icon=ICONS['ERROR'])

                col = info_box.column(align=True)
                draw_tree_dict(col, tree_dict, duplicates=duplicates)

            info_box.separator()

            tip_box = info_box.box()
            tip_header = tip_box.row()
            icon_tip = ICONS['DOWN'] if scene.batch_stl_ui_tips else ICONS['RIGHT']
            tip_header.prop(scene, "batch_stl_ui_tips", text="", icon=icon_tip, emboss=False)
            tip_header.label(text="OVERRIDE INFO", icon=ICONS['INFO'])

            if scene.batch_stl_ui_tips:
                col = tip_box.column()
                col.label(text="Hierarchy: Global > Preset > Collection > Object > NodeGroup > Node.", icon=ICONS['BLANK'])
                col.label(text="For modifier targets, leave Node blank or set as <Modifier Interface>", icon=ICONS['BLANK'])
                col.separator()

                col.label(text="Sweep Mode (Shift-Click '+' button to toggle):", icon=ICONS['SWEEP'])
                col.label(text="  • Floats/Ints: Define start, step, and count", icon=ICONS['BLANK'])
                col.label(text="  • Menus/Bools: Auto-iterates all values", icon=ICONS['BLANK'])
                col.label(text="  • Shift-Click when active to populate all sweep values", icon=ICONS['BLANK'])
                col.separator()

                col.label(text="Export Tools (Per Value):", icon=ICONS['BLANK'])
                col.label(text="  • Folder Icon: Save this value's exports into a subfolder", icon=ICONS['DIR'])
                col.label(text="  • Bookmark Icon: Append/Prepend a tag to filename", icon=ICONS['TAG'])

                col.label(text="Tag Formatting:", icon=ICONS['BLANK'])
                col.label(text="  • [ tag ] replaces input value, [ _tag ] appends, [ tag_ ] prepends", icon=ICONS['BLANK'])

        layout.separator()

        # Global Overrides
        g_col = layout.column()
        g_col.enabled = not any_exporting
        draw_overrides_table(g_col, scene, scene.batch_stl_global_nodegroups, False, "batch_stl_ui_global_ovr_main", "Global Overrides", is_global=True, is_locked=any_exporting)

class VIEW3D_PT_batch_export_stl_presets(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Combi Export"
    bl_label = "Presets"

    @classmethod
    def poll(cls, context):
        return True

    def draw_header(self, context):
        self.layout.label(text="", icon=ICONS['PRESET'])

    def draw_header_preset(self, context):
        layout = self.layout
        scene = context.scene
        any_exporting = is_any_exporting()
        active_preset = get_active_preset(scene)

        row = layout.row(align=True)
        if active_preset and active_preset.last_export_time > 0:
            row.label(text=f"{active_preset.last_export_time:.2f}s", icon=ICONS['TIME'])
            row.separator()
        row.enabled = not any_exporting
        draw_inline_controls(row, "batch_stl.preset_actions", use_clipboard=True)
        row.separator()

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        any_exporting = is_any_exporting()
        active_preset = get_active_preset(scene)
        stats = _ui_cache.get("stats", {})

        content_col = layout.column(align=True)
        list_box = content_col.box()
        list_box.template_list("BATCH_STL_UL_presets", "", scene, "batch_stl_presets", scene, "batch_stl_preset_index", rows=3)

        locked_col = content_col.column(align=True)
        locked_col.enabled = not any_exporting

        g_stats = stats.get("global", {"presets": 0, "cols": 0, "objs": 0, "exp": 0})
        draw_stats_table(locked_col, [
            (g_stats['presets'], ICONS['PRESET']),
            (g_stats['cols'], ICONS['COLLECTION']),
            (g_stats['objs'], ICONS['OBJECT']),
            (g_stats['exp'], ICONS['SWEEP'])
        ])

        if active_preset:
            draw_overrides_table(locked_col, scene, active_preset.nodegroups, False, "batch_stl_ui_preset_ovr", f"Overrides for [ {active_preset.name} ]", is_preset=True, is_locked=any_exporting)


class VIEW3D_PT_batch_export_stl_collections(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Combi Export"
    bl_label = "Collections"

    @classmethod
    def poll(cls, context):
        return get_active_preset(context.scene) is not None

    def draw_header(self, context):
        self.layout.label(text="", icon=ICONS['COLLECTION'])

    def draw_header_preset(self, context):
        layout = self.layout
        scene = context.scene
        any_exporting = is_any_exporting()

        row = layout.row(align=True)
        row.enabled = not any_exporting
        draw_inline_controls(row, "batch_stl.collection_actions", use_clipboard=True)
        row.separator()

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        any_exporting = is_any_exporting()
        active_preset = get_active_preset(scene)
        stats = _ui_cache.get("stats", {})

        layout.enabled = not any_exporting
        active_col = get_active_collection(active_preset)

        content_col = layout.column(align=True)
        list_box = content_col.box()
        list_box.template_list("BATCH_STL_UL_collections", "", active_preset, "collections", active_preset, "collection_index", rows=5)

        p_stats = stats.get("presets", {}).get(scene.batch_stl_preset_index, {"cols": 0, "objs": 0, "exp": 0})
        draw_stats_table(content_col, [
            (p_stats['cols'], ICONS['COLLECTION']),
            (p_stats['objs'], ICONS['OBJECT']),
            (p_stats['exp'], ICONS['SWEEP'])
        ])

        if active_col:
            draw_overrides_table(content_col, scene, active_col.nodegroups, True, "batch_stl_ui_global_ovr", f"Overrides [ {active_col.collection_name or 'Shared'} ]", is_locked=any_exporting)


class VIEW3D_PT_batch_export_stl_objects(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Combi Export"
    bl_label = "Objects"

    @classmethod
    def poll(cls, context):
        active_preset = get_active_preset(context.scene)
        if not active_preset: return False
        return get_active_collection(active_preset) is not None

    def draw_header(self, context):
        self.layout.label(text="", icon=ICONS['OBJECT'])

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        any_exporting = is_any_exporting()
        active_preset = get_active_preset(scene)
        stats = _ui_cache.get("stats", {})

        layout.enabled = not any_exporting
        active_col = get_active_collection(active_preset)
        active_obj = get_active_object(active_col)

        content_col = layout.column(align=True)
        list_box = content_col.box()
        list_box.template_list("BATCH_STL_UL_objects", "", active_col, "objects", active_col, "object_index", rows=5)

        c_idx = active_preset.collection_index
        col_key = (scene.batch_stl_preset_index, c_idx)
        c_stats = stats.get("cols", {}).get(col_key, {"objs": 0, "exp": 0})
        draw_stats_table(content_col, [
            (c_stats['objs'], ICONS['OBJECT']),
            (c_stats['exp'], ICONS['SWEEP'])
        ])

        if active_obj:
            draw_overrides_table(content_col, scene, active_obj.nodegroups, False, "batch_stl_ui_local_ovr", f"Overrides [ {active_obj.name} ]", is_locked=any_exporting)

# ==============================================================================
# === [ 6. REGISTRATION & LIFECYCLE ] ===
# ==============================================================================

@persistent
def reset_batch_stl_state(*args):
    try:
        for wm in bpy.data.window_managers: wm.batch_stl_jobs.clear()  # no export survives a file load
    except Exception: pass
    mark_dirty()
    if "--batch-stl-headless" not in sys.argv and not bpy.app.timers.is_registered(rebuild_ui_cache_if_dirty):
        bpy.app.timers.register(rebuild_ui_cache_if_dirty)

classes = (
    BatchSTLLogLine, BatchSTLJob, BatchSTLValue, BatchSTLInput, BatchSTLNode, BatchSTLNodeGroup, BatchSTLObject, BatchSTLCollection, BatchSTLExportPreset,
    BATCH_STL_UL_presets, BATCH_STL_UL_collections, BATCH_STL_UL_objects, BATCH_STL_UL_console_logs,
    BATCH_STL_OT_clear_console, BATCH_STL_OT_preset_actions, BATCH_STL_OT_collection_actions, BATCH_STL_OT_table_action, BATCH_STL_OT_toggle_dir_tree, BATCH_STL_OT_cancel_export, BATCH_STL_OT_export_presets_json, BATCH_STL_OT_import_presets_json, EXPORT_OT_batch_stl_multi,
    VIEW3D_PT_batch_export_stl_main, VIEW3D_PT_batch_export_stl_presets, VIEW3D_PT_batch_export_stl_collections, VIEW3D_PT_batch_export_stl_objects
)

def update_show_console(self, context):
    mark_dirty()
    if not self.batch_stl_show_console: self.batch_stl_collapsed_dirs = "[]"

def register():
    for cls in classes: bpy.utils.register_class(cls)

    bpy.types.WindowManager.batch_stl_jobs = bpy.props.CollectionProperty(type=BatchSTLJob)

    Scene = bpy.types.Scene
    Scene.batch_stl_root_dir = bpy.props.StringProperty(name="Root", default="//", subtype="DIR_PATH", update=mark_dirty)
    Scene.batch_stl_presets = bpy.props.CollectionProperty(type=BatchSTLExportPreset)
    Scene.batch_stl_global_nodegroups = bpy.props.CollectionProperty(type=BatchSTLNodeGroup)
    Scene.batch_stl_preset_index = bpy.props.IntProperty(name="Active Preset", default=0, update=mark_dirty)
    Scene.batch_stl_verbose_console = bpy.props.BoolProperty(name="Verbose Console Output", default=False, options={'SKIP_SAVE'})

    for prop in ["batch_stl_ui_global_ovr_main", "batch_stl_ui_preset_ovr", "batch_stl_ui_global_ovr", "batch_stl_ui_local_ovr", "batch_stl_ui_global_ovr_nested", "batch_stl_ui_local_ovr_nested", "batch_stl_ui_tips"]:
        setattr(Scene, prop, bpy.props.BoolProperty(default=True if ("nested" not in prop and "tips" not in prop) else False, options={'SKIP_SAVE'}))

    Scene.batch_stl_show_console = bpy.props.BoolProperty(default=False, update=update_show_console, options={'SKIP_SAVE'})
    Scene.batch_stl_collapsed_dirs = bpy.props.StringProperty(default="[]", options={'SKIP_SAVE'})
    Scene.batch_stl_info_tab = bpy.props.EnumProperty(items=[('LOG', "Console Log", ""), ('TREE', "Tree View", "")], name="Info Tab", default='LOG', update=lambda s, c: mark_dirty(), options={'SKIP_SAVE'})
    Scene.batch_stl_info_global = bpy.props.BoolProperty(name="Global Mode", default=False, update=lambda s, c: mark_dirty(), options={'SKIP_SAVE'})

    reset_batch_stl_state(None)
    if reset_batch_stl_state not in bpy.app.handlers.load_post: bpy.app.handlers.load_post.append(reset_batch_stl_state)

    if "--batch-stl-headless" not in sys.argv:
        if batch_stl_depsgraph_handler not in bpy.app.handlers.depsgraph_update_post: bpy.app.handlers.depsgraph_update_post.append(batch_stl_depsgraph_handler)
        if not bpy.app.timers.is_registered(rebuild_ui_cache_if_dirty): bpy.app.timers.register(rebuild_ui_cache_if_dirty)

def unregister():
    if reset_batch_stl_state in bpy.app.handlers.load_post: bpy.app.handlers.load_post.remove(reset_batch_stl_state)
    if batch_stl_depsgraph_handler in bpy.app.handlers.depsgraph_update_post: bpy.app.handlers.depsgraph_update_post.remove(batch_stl_depsgraph_handler)
    if bpy.app.timers.is_registered(rebuild_ui_cache_if_dirty): bpy.app.timers.unregister(rebuild_ui_cache_if_dirty)

    for cls in reversed(classes):
        try: bpy.utils.unregister_class(cls)
        except RuntimeError: pass

    if hasattr(bpy.types.WindowManager, "batch_stl_jobs"): del bpy.types.WindowManager.batch_stl_jobs

    props = ["batch_stl_root_dir", "batch_stl_presets", "batch_stl_preset_index", "batch_stl_global_nodegroups", "batch_stl_ui_global_ovr_main", "batch_stl_verbose_console", "batch_stl_ui_preset_ovr", "batch_stl_ui_global_ovr", "batch_stl_ui_local_ovr", "batch_stl_show_console", "batch_stl_collapsed_dirs", "batch_stl_ui_tips", "batch_stl_ui_global_ovr_nested", "batch_stl_ui_local_ovr_nested", "batch_stl_info_tab", "batch_stl_info_global"]
    for p in props:
        if hasattr(bpy.types.Scene, p): delattr(bpy.types.Scene, p)

if __name__ == "__main__":
    if "--batch-stl-headless" in sys.argv:
        if not hasattr(bpy.types.Scene, "batch_stl_root_dir"): register()
        run_headless_export(sys.argv[sys.argv.index("--batch-stl-headless") + 1])
    else:
        if not hasattr(bpy.types.Scene, "batch_stl_root_dir"): register()
