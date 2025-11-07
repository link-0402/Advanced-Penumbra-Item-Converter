"""Advanced Penumbra Item Converter GUI.

Provides tools to preview and convert Penumbra mod assets from one item ID to another,
including planned renames, JSON updates, and binary path updates, with a checklist UI.
"""

import os
import difflib
import shutil
import sys
import json
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

SLOT_LABEL_MAP = {
    'met': 'Head',
    'top': 'Body',
    'glv': 'Hands',
    'dwn': 'Legs',
    'sho': 'Feet',
    'ear': 'Earring',
    'nek': 'Neck',
    'wrs': 'Wrists',
    'rir': 'Ring Right',
    'ril': 'Ring Left',
}


def _relative_path(base, target):
    base_norm = os.path.normcase(os.path.normpath(base))
    target_norm = os.path.normcase(os.path.normpath(target))
    try:
        common = os.path.commonpath([base_norm, target_norm])
    except ValueError:
        return target
    if common == base_norm:
        rel_path = os.path.relpath(os.path.normpath(target), os.path.normpath(base))
        return rel_path.replace("\\", "/")
    return target


# small helpers to keep path normalization consistent and concise
def _norm(path):
    """Normalize a filesystem path for comparisons (case-insensitive, normalized separators)."""
    return os.path.normcase(os.path.normpath(path))


def _norm_set(paths):
    """Return a set of normalized paths or None when input is None, for membership checks."""
    return { _norm(p) for p in paths } if paths is not None else None

# ============================================================

def _path_matches_slot(path: str, slot_key: str) -> bool:
    """Return True if the path appears to belong to the selected slot.

    Matches on filename suffix (e.g., "_top") or a directory segment (e.g., "/top/").
    """
    pl = path.replace("\\", "/").lower()
    base = os.path.basename(pl)
    slot_flag = f"_{slot_key}".lower()
    if slot_flag in base:
        return True
    return f"/{slot_key}/" in pl

def verify_target_path(base_dir, logger=print):
    """Validate the target directory and required Penumbra files exist before modification."""
    abs_path = os.path.abspath(base_dir)

    # Check that the path exists
    if not os.path.isdir(abs_path):
        raise ValueError(f"❌ Error: '{abs_path}' is not a valid directory.")

    # Check that required files exist in the root of the target directory
    required_files = ["default_mod.json", "meta.json"]
    missing = [f for f in required_files if not os.path.isfile(os.path.join(abs_path, f))]
    if missing:
        raise FileNotFoundError(
            f"❌ Missing required file(s) in target directory: {', '.join(missing)}"
        )

    logger(f"✅ Path verified: {abs_path}")
    return abs_path


def rename_files_and_dirs(
    base_dir,
    old_substring,
    new_substring,
    slot_key,
    logger=print,
    allowed_paths=None,
    ignore_slot=False,
):
    """Rename files and directories containing a token when their names match the selected slot."""
    slot_flag = f"_{slot_key}".lower()
    id_token = old_substring.lower()
    new_token = new_substring.lower()
    allowed_normalized = None
    if allowed_paths is not None:
        allowed_normalized = _norm_set(allowed_paths)
        if not allowed_normalized:
            logger("ℹ️ No files selected for renaming. Skipping.")
            return

    for root, dirs, files in os.walk(base_dir, topdown=False):
        # Rename files
        for filename in files:
            fname_low = filename.lower()
            if id_token in fname_low:
                # Only check slot flag if not ignoring slot restrictions
                if not ignore_slot and slot_flag not in fname_low:
                    continue
                old_path = os.path.join(root, filename)
                old_norm = _norm(old_path)
                if allowed_normalized is not None and old_norm not in allowed_normalized:
                    continue
                new_filename = filename.replace(old_substring, new_substring)
                new_path = os.path.join(root, new_filename)

                if os.path.exists(new_path):
                    logger(f"⚠️ Skipping file (already exists): {new_path}")
                    continue

                os.rename(old_path, new_path)
                logger(f"Renamed file: {old_path} -> {new_path}")

        # Rename directories
        for dirname in dirs:
            dname_low = dirname.lower()
            if id_token not in dname_low:
                continue

            old_dir = os.path.join(root, dirname)
            old_norm = _norm(old_dir)
            if allowed_normalized is not None and old_norm not in allowed_normalized:
                continue
            if not _directory_has_slot_content(old_dir, [id_token], slot_flag, ignore_slot):
                logger(f"⚠️ Skipping directory (no files with slot-specific content): {old_dir}")
                continue

            new_dir = os.path.join(root, dirname.replace(old_substring, new_substring))

            if os.path.exists(new_dir):
                logger(f"⚠️ Skipping directory (already exists): {new_dir}")
                continue

            os.rename(old_dir, new_dir)
            logger(f"Renamed directory: {old_dir} -> {new_dir}")


def _directory_has_slot_content(directory, id_tokens, slot_flag, ignore_slot=False):
    tokens = [t.lower() for t in id_tokens if t]
    for root, _, files in os.walk(directory):
        for fn in files:
            fn_low = fn.lower()
            # If ignoring slot, just check for any token match
            if ignore_slot:
                if any(token in fn_low for token in tokens):
                    return True
            else:
                if slot_flag in fn_low and any(token in fn_low for token in tokens):
                    return True
    return False

def _make_rel_norm(path, base_dir):
    try:
        rel = os.path.relpath(path, base_dir)
    except Exception:
        rel = path
    return rel.replace("\\", "/").lstrip("./").lower()


def _build_json_scope(base_dir, rename_pairs=None, rename_entries=None):
    """Build an allowlist of adjusted files/dirs (relative, lowercase) for constraining JSON edits."""
    files = set()
    dirs = set()
    if rename_entries is not None:
        for it in rename_entries:
            oldp = it.get("old")
            if not oldp:
                continue
            rel = _make_rel_norm(oldp, base_dir)
            if it.get("is_dir"):
                dirs.add(rel.rstrip("/"))
            else:
                files.add(rel)
    elif rename_pairs is not None:
        for oldp, _ in rename_pairs:
            rel = _make_rel_norm(oldp, base_dir)
            if os.path.isdir(oldp):
                dirs.add(rel.rstrip("/"))
            else:
                files.add(rel)
    return {"files": files, "dirs": dirs}


def _path_in_scope(path_value, scope):
    if not scope:
        return True
    s = path_value.replace("\\", "/").lstrip("./").lower()
    if s in scope.get("files", set()):
        return True
    for d in scope.get("dirs", set()):
        if s == d or s.startswith(d + "/"):
            return True
    return False


def _extract_race_code_from_name(name: str):
    """Extract a 4-digit race code following 'c' in a name (e.g., 'c0201'); return None if absent."""
    s = name.lower()
    for i in range(len(s) - 4):
        if s[i] == 'c':
            segment = s[i+1:i+5]
            if segment.isdigit():
                return segment
    return None


def _plan_race_deletions(base_dir, selected_race_code, exclude_paths=None):
    """Plan deletion of files for non-selected races and any directories left empty as a result.

    Returns a tuple: (files_to_delete, dirs_to_delete, delete_scope).
    """
    exclude_norm = {_norm(p) for p in (exclude_paths or set())}

    files_to_delete = []
    all_files = []
    all_dirs = set()

    # Gather all files and mark wrong-race files
    for r, ds, fs in os.walk(base_dir):
        for d in ds:
            all_dirs.add(_norm(os.path.join(r, d)))
        for fn in fs:
            p = os.path.join(r, fn)
            pn = _norm(p)
            all_files.append(pn)
            if exclude_norm and pn in exclude_norm:
                continue
            rc = _extract_race_code_from_name(fn)
            if rc and rc != selected_race_code:
                files_to_delete.append(p)

    wrong_norm = {_norm(p) for p in files_to_delete}

    # Determine which directories would have remaining files after deletion
    dirs_with_remaining = set()
    base_norm = _norm(base_dir)
    for f in all_files:
        if f not in wrong_norm:
            # mark this file's directory and all ancestors up to base as having remaining content
            cur = _norm(os.path.dirname(f))
            while cur and cur.startswith(base_norm):
                dirs_with_remaining.add(cur)
                if cur == base_norm:
                    break
                cur = _norm(os.path.dirname(cur))

    # Compute deletable directories: all directories under base that are not marked as having remaining
    # Exclude the base directory itself.
    deletable_dirs = []
    for d in sorted(all_dirs, key=lambda x: x.count(os.sep), reverse=True):
        if d == base_norm:
            continue
        if d not in dirs_with_remaining:
            deletable_dirs.append(d)

    # Convert normalized dir paths back to their canonical form (best-effort)
    def _denorm(path_norm: str) -> str:
        # Attempt to find actual-cased path on filesystem
        parts = os.path.relpath(path_norm, base_norm).split(os.sep)
        cur = base_dir
        for part in parts:
            if not part or part == "." or part == "..":
                continue
            try:
                # find matching child directory by case-insensitive compare
                entries = os.listdir(cur)
                match = None
                for e in entries:
                    if _norm(os.path.join(cur, e)) == _norm(os.path.join(cur, part)):
                        match = e
                        break
                cur = os.path.join(cur, match or part)
            except Exception:
                cur = os.path.join(cur, part)
        return cur

    deletable_dirs_real = [_denorm(d) for d in deletable_dirs]

    # Build delete scope from files and empty dirs
    entries = ([{"old": p, "new": p, "is_dir": False} for p in files_to_delete] +
               [{"old": d, "new": d, "is_dir": True} for d in deletable_dirs_real])
    delete_scope = _build_json_scope(base_dir, rename_entries=entries)
    return files_to_delete, deletable_dirs_real, delete_scope


def _build_json_context(old_padded, new_padded, old_stripped, new_stripped, slot_key, json_scope=None, delete_scope=None, race_code=None):
    slot_label = SLOT_LABEL_MAP.get(slot_key, slot_key)
    return {
        "old_token": f"e{old_padded}",
        "new_token": f"e{new_padded}",
        "id_token_lower": f"e{old_padded}".lower(),
        "slot_flag": f"_{slot_key}".lower(),
        "slot_label": slot_label,
        "slot_label_lower": slot_label.lower(),
        "old_int": int(old_stripped),
        "new_int": int(new_stripped),
        "new_stripped": new_stripped,
        "scope": json_scope,
        "delete_scope": delete_scope,
        "race_code": race_code,
    }


def _replace_numeric_field(value, context, apply_changes):
    if isinstance(value, int):
        if value == context["old_int"]:
            return (context["new_int"] if apply_changes else value), True, 1
        return value, False, 0

    if isinstance(value, str) and value.isdigit():
        if int(value) == context["old_int"]:
            return (context["new_stripped"] if apply_changes else value), True, 1
        return value, False, 0

    return value, False, 0


def _replace_path_string(value, context, apply_changes, ignore_scope=False):
    """Replace occurrences of the old item token in strings (typically file paths).

    When a scope is provided, only replace strings that fall within the allowed files/dirs.
    For Files dict keys (game paths), scope checks can be bypassed with ignore_scope=True.
    """
    if not isinstance(value, str):
        return value, False, 0

    lower_value = value.lower()
    if context["id_token_lower"] in lower_value:
        # Only replace if the string path is within the adjusted file/dir scope,
        # unless scope is explicitly ignored (used for Files dict keys which are game paths).
        scope = context.get("scope")
        # Heuristic: only treat as path-like if it contains a path separator
        looks_like_path = ("/" in value) or ("\\" in value)
        # Treat an empty scope (no files/dirs) as "no scope" so JSON edits are not suppressed
        has_effective_scope = bool(scope and (scope.get("files") or scope.get("dirs")))
        if has_effective_scope and looks_like_path and not ignore_scope and not _path_in_scope(value, scope):
            return value, False, 0
        occurrences = lower_value.count(context["id_token_lower"])
        if apply_changes:
            new_value = value.replace(context["old_token"], context["new_token"])
            if new_value != value:
                return new_value, True, occurrences
            return value, False, 0
        return value, True, occurrences

    return value, False, 0


def _process_files_dict(mapping, context, logger, path, apply_changes):
    if not isinstance(mapping, dict):
        return mapping, False, 0

    modified = False
    total_count = 0
    result = {} if apply_changes else None

    for key, value in mapping.items():
        # Deletion gating: if key/value path points to an unselected race, drop it
        delete_scope = context.get("delete_scope")
        # Treat an empty delete scope (no files/dirs) as disabled
        has_delete_scope = bool(delete_scope and (delete_scope.get("files") or delete_scope.get("dirs")))
        looks_like_path_key = isinstance(key, str) and ("/" in key or "\\" in key)
        looks_like_path_val = isinstance(value, str) and ("/" in value or "\\" in value)
        if has_delete_scope and ( (looks_like_path_key and _path_in_scope(key, delete_scope)) or (looks_like_path_val and _path_in_scope(value, delete_scope)) ):
            # Count one match; drop entry if applying changes
            if apply_changes:
                # drop by not adding to result
                pass
            modified = True
            total_count += 1
            if not apply_changes:
                # keep original mapping untouched in preview mode
                continue
            else:
                # skip further processing for this entry in apply mode
                continue

        # Race gating: if key or value contains an explicit c#### race tag that is
        # different from the selected race, do not modify this entry when deletion
        # is not enabled. This keeps other-race JSON entries untouched.
        selected_rc = context.get("race_code")
        wrong_race = False
        if selected_rc and (looks_like_path_key or looks_like_path_val):
            rc_k = _extract_race_code_from_name(key) if looks_like_path_key else None
            rc_v = _extract_race_code_from_name(value) if looks_like_path_val else None
            rc_any = rc_v or rc_k
            if rc_any and rc_any != selected_rc and not has_delete_scope:
                wrong_race = True
        if wrong_race:
            # Do not count or modify this entry; keep as-is
            if apply_changes:
                if result is not None:
                    result[key] = value
            continue
        # First, process the value so we know if it will change.
        new_value = value
        value_changed = False
        value_count = 0
        if isinstance(value, str):
            replaced_value, value_changed, value_count = _replace_path_string(value, context, apply_changes)
            if apply_changes and value_changed:
                new_value = replaced_value
        else:
            replaced_value, value_changed, value_count = _process_json_value(
                value,
                context,
                logger,
                f"{path}.{key}" if isinstance(key, str) else f"{path}.{key!r}",
                apply_changes,
            )
            if apply_changes and value_changed:
                new_value = replaced_value

        total_count += value_count
        modified = modified or value_changed

        # Rename the left-hand key independently of the right-hand value.
        new_key = key
        key_changed = False
        key_count = 0
        if isinstance(key, str):
            # For Files/FileSwaps dict keys (game paths), allow replacement regardless of scope
            replaced_key, key_changed, key_count = _replace_path_string(key, context, apply_changes, ignore_scope=True)
            if apply_changes and key_changed:
                new_key = replaced_key

        total_count += key_count
        modified = modified or key_changed

        if apply_changes:
            if key_changed and new_key in result and new_key != key:
                logger(
                    f"⚠️ Key collision while renaming '{key}' -> '{new_key}' at {path}. Keeping original key."
                )
                new_key = key
            result[new_key] = new_value

    if apply_changes:
        return result, modified, total_count
    return mapping, modified, total_count


def _process_json_value(value, context, logger, path, apply_changes):
    if isinstance(value, dict):
        slot_match = False
        for slot_field in ("EquipSlot", "Slot"):
            slot_value = value.get(slot_field)
            if isinstance(slot_value, str) and slot_value.lower() == context["slot_label_lower"]:
                slot_match = True
                break

        modified = False
        total_count = 0
        result = {} if apply_changes else None

        for key, sub_value in value.items():
            sub_path = f"{path}.{key}" if isinstance(key, str) else f"{path}.{key!r}"

            if key in ("Files", "FileSwaps") and isinstance(sub_value, dict):
                processed_value, sub_modified, sub_count = _process_files_dict(
                    sub_value, context, logger, sub_path, apply_changes
                )
                modified = modified or sub_modified
                total_count += sub_count
                if apply_changes:
                    result[key] = processed_value
                continue

            if key in ("PrimaryId", "SetId") and slot_match:
                processed_value, sub_modified, sub_count = _replace_numeric_field(
                    sub_value, context, apply_changes
                )
                modified = modified or sub_modified
                total_count += sub_count
                if apply_changes:
                    result[key] = processed_value if sub_modified else sub_value
                continue

            processed_value, sub_modified, sub_count = _process_json_value(
                sub_value, context, logger, sub_path, apply_changes
            )
            modified = modified or sub_modified
            total_count += sub_count
            if apply_changes:
                result[key] = processed_value if sub_modified else sub_value

        if apply_changes:
            return result, modified, total_count
        return value, modified, total_count

    if isinstance(value, list):
        modified = False
        total_count = 0
        result = [] if apply_changes else None

        for idx, item in enumerate(value):
            processed_item, item_modified, item_count = _process_json_value(
                item, context, logger, f"{path}[{idx}]", apply_changes
            )
            modified = modified or item_modified
            total_count += item_count
            if apply_changes:
                result.append(processed_item if item_modified else item)

        if apply_changes:
            return result, modified, total_count
        return value, modified, total_count

    if isinstance(value, str):
        new_value, changed, occurrences = _replace_path_string(value, context, apply_changes)
        if changed:
            if apply_changes:
                return new_value, True, occurrences
            return value, True, occurrences
        return value, False, 0

    return value, False, 0


def _transform_json_data(
    data,
    old_padded,
    new_padded,
    old_stripped,
    new_stripped,
    slot_key,
    race_code,
    apply_changes,
    logger,
    json_scope=None,
    delete_scope=None,
):
    context = _build_json_context(
        old_padded, new_padded, old_stripped, new_stripped, slot_key, json_scope=json_scope, delete_scope=delete_scope, race_code=race_code
    )
    return _process_json_value(data, context, logger, "<root>", apply_changes)


def _extract_paths_from_bytes_generic(data: bytes, suffix: str):
    """Extract ASCII path-like substrings from binary data that end with the given suffix."""
    results = set()
    suf_bytes = suffix.encode("latin-1", errors="ignore")
    idx = 0
    allowed = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-/\\")
    dl = len(data)
    sl = len(suf_bytes)
    while True:
        pos = data.find(suf_bytes, idx)
        if pos == -1:
            break
        start = pos - 1
        while start >= 0 and data[start] in allowed:
            start -= 1
        start += 1
        end = pos + sl
        while end < dl and data[end] in allowed:
            end += 1
        seg = data[start:end]
        try:
            s = seg.decode("latin-1")
        except Exception:
            s = None
        if s:
            s = s.replace("\\", "/")
            results.add(s)
        idx = pos + sl
    return list(results)


def _index_files_ci(base_dir):
    idx = {}
    for r, _d, fs in os.walk(base_dir):
        for fn in fs:
            p = os.path.join(r, fn)
            rel = os.path.relpath(p, base_dir).replace("\\", "/").lower()
            idx[rel] = p
    return idx


def _resolve_many_ci(index, rel_paths):
    resolved = []
    rel_norms = [rp.replace("\\", "/").lstrip("./").lstrip("/").lower() for rp in rel_paths]
    for rp in rel_norms:
        p = index.get(rp)
        if p:
            resolved.append(p)
        for key, ap in index.items():
            if key.endswith(rp):
                resolved.append(ap)
    # dedupe preserving order
    out = []
    seen = set()
    for p in resolved:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def plan_cleanup_wrong_race_assets(base_dir, old_padded, new_padded, selected_race_code, logger=print):
    """Plan deletion of assets belonging to other races and collect their referenced materials/textures.

    Returns a dict of absolute paths to delete: {models: [], materials: [], textures: []}.
    """
    token_old = f"e{old_padded}".lower()
    token_new = f"e{new_padded}".lower()

    idx = _index_files_ci(base_dir)

    def list_files(ext):
        out = []
        for r, _d, fs in os.walk(base_dir):
            for fn in fs:
                if fn.lower().endswith(ext):
                    out.append(os.path.join(r, fn))
        return out

    def has_selected_race(p):
        rc = _extract_race_code_from_name(os.path.basename(p))
        if rc:
            return rc == selected_race_code
        return True  # untagged treated as potentially shared; include as protected if referenced

    # Build protected sets from selected-race models with new token
    protected_models = set()
    protected_materials = set()
    protected_textures = set()
    for mdl in list_files(".mdl"):
        nm = os.path.basename(mdl).lower()
        if token_new in nm and has_selected_race(mdl):
            protected_models.add(mdl)
            try:
                with open(mdl, "rb") as f:
                    mb = f.read()
                mtrls = _extract_paths_from_bytes_generic(mb, ".mtrl")
                resolved_m = _resolve_many_ci(idx, mtrls)
                for m in resolved_m:
                    protected_materials.add(m)
                    try:
                        with open(m, "rb") as mf:
                            tb = mf.read()
                        texs = _extract_paths_from_bytes_generic(tb, ".tex")
                        resolved_t = _resolve_many_ci(idx, texs)
                        for t in resolved_t:
                            protected_textures.add(t)
                    except Exception:
                        continue
            except Exception:
                continue

    # Find wrong-race models for this item (old or new token) with explicit different race tag
    wrong_models = []
    for mdl in list_files(".mdl"):
        nm = os.path.basename(mdl).lower()
        if token_old in nm or token_new in nm:
            rc = _extract_race_code_from_name(nm)
            if rc and rc != selected_race_code:
                wrong_models.append(mdl)

    # Collect their referenced materials/textures
    cand_materials = set()
    cand_textures = set()
    for mdl in wrong_models:
        try:
            with open(mdl, "rb") as f:
                mb = f.read()
            mtrls = _extract_paths_from_bytes_generic(mb, ".mtrl")
            resolved_m = _resolve_many_ci(idx, mtrls)
            for m in resolved_m:
                cand_materials.add(m)
                try:
                    with open(m, "rb") as mf:
                        tb = mf.read()
                    texs = _extract_paths_from_bytes_generic(tb, ".tex")
                    resolved_t = _resolve_many_ci(idx, texs)
                    for t in resolved_t:
                        cand_textures.add(t)
                except Exception:
                    continue
        except Exception:
            continue

    # Exclude protected assets
    del_materials = [m for m in cand_materials if m not in protected_materials]
    del_textures = [t for t in cand_textures if t not in protected_textures]

    plan = {
        "models": wrong_models,
        "materials": del_materials,
        "textures": del_textures,
    }
    logger(f"Cleanup plan: {len(plan['models'])} wrong-race models, {len(plan['materials'])} materials, {len(plan['textures'])} textures")
    return plan


def apply_cleanup_wrong_race_assets(base_dir, plan, logger=print):
    """Delete files from the plan and remove references from JSON Files/FileSwaps entries."""
    files = list(plan.get("models", [])) + list(plan.get("materials", [])) + list(plan.get("textures", []))
    # Delete files
    for p in files:
        try:
            if os.path.isfile(p):
                os.remove(p)
                logger(f"🗑️ Deleted obsolete asset: {p}")
        except Exception as e:
            logger(f"❌ Failed to delete {p}: {e}")

    # Remove empty directories bottom-up
    for r, ds, fs in os.walk(base_dir, topdown=False):
        if not ds and not fs:
            try:
                if os.path.isdir(r):
                    os.rmdir(r)
                    logger(f"🧹 Removed empty folder: {r}")
            except Exception:
                pass

    # Build set of normalized relative paths for deleted files for JSON cleanup
    deleted_rel_norm = set()
    for p in files:
        try:
            rel = os.path.relpath(p, base_dir).replace("\\", "/").lower()
            deleted_rel_norm.add(rel)
        except Exception:
            continue

    # JSON cleanup: drop Files/FileSwaps entries whose key or value endswith any deleted rel path
    for r, _d, fs in os.walk(base_dir):
        for fn in fs:
            if not fn.lower().endswith('.json'):
                continue
            path = os.path.join(r, fn)
            try:
                with open(path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
            except Exception:
                continue
            changed = False
            try:
                opts = data.get('Options', [])
                if isinstance(opts, list):
                    for opt in opts:
                        if not isinstance(opt, dict):
                            continue
                        for dict_key in ("Files", "FileSwaps"):
                            m = opt.get(dict_key)
                            if not isinstance(m, dict):
                                continue
                            new_m = {}
                            for k, v in m.items():
                                kval = k if isinstance(k, str) else str(k)
                                vval = v if isinstance(v, str) else str(v)
                                kvaln = kval.replace('\\', '/').lstrip('./').lstrip('/').lower()
                                vvaln = vval.replace('\\', '/').lstrip('./').lstrip('/').lower()
                                drop = any(kvaln.endswith(d) or vvaln.endswith(d) for d in deleted_rel_norm)
                                if not drop:
                                    new_m[k] = v
                                else:
                                    changed = True
                            opt[dict_key] = new_m
            except Exception:
                continue
            if changed:
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    logger(f"Updated JSON after cleanup: {path}")
                except Exception as e:
                    logger(f"❌ Failed writing JSON {path}: {e}")


class PreviewDialog(tk.Toplevel):
    def __init__(self, parent, base_dir, renames, json_entries, binary_entries):
        super().__init__(parent)
        self.title("Preview Results")
        self.parent = parent
        self.base_dir = base_dir.rstrip("\\/")
        self.renames = renames
        self.json_entries = json_entries
        self.binary_entries = binary_entries
        self.result = None
        self.vars = {"renames": [], "jsons": [], "binaries": []}

        self.transient(parent)
        self.grab_set()

        self._build_ui()

    # Watch parent ID/slot variables to refresh preview; keep trace IDs to remove on close.
        self._parent_traces = []
        if hasattr(self.parent, "old_id_var"):
            tid = self.parent.old_id_var.trace_add("write", lambda *a: self._on_parent_var_change())
            self._parent_traces.append((self.parent.old_id_var, tid))
        if hasattr(self.parent, "new_id_var"):
            tid = self.parent.new_id_var.trace_add("write", lambda *a: self._on_parent_var_change())
            self._parent_traces.append((self.parent.new_id_var, tid))
        if hasattr(self.parent, "slot_var"):
            tid = self.parent.slot_var.trace_add("write", lambda *a: self._on_parent_var_change())
            self._parent_traces.append((self.parent.slot_var, tid))

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.resizable(True, True)
        self.minsize(600, 400)
    # Ensure traces are removed when the dialog is destroyed.
        self.bind("<Destroy>", lambda _e: self._remove_parent_traces())
        self.focus()

    def _build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        rename_frame = self._create_checklist_tab(
            notebook,
            "Renames",
            "renames",
            self.renames,
            lambda item: f"{'[DIR]' if item['is_dir'] else '[FILE]'} {_relative_path(self.base_dir, item['old'])} → {_relative_path(self.base_dir, item['new'])}",
        )
        notebook.add(rename_frame, text=f"Renames ({len(self.renames)})")

        json_frame = self._create_checklist_tab(
            notebook,
            "JSON Updates",
            "jsons",
            self.json_entries,
            lambda item: f"{_relative_path(self.base_dir, item['path'])} (matches: {item['count']})",
        )
        notebook.add(json_frame, text=f"JSON ({len(self.json_entries)})")

        binary_frame = self._create_checklist_tab(
            notebook,
            "Binary Updates",
            "binaries",
            self.binary_entries,
            lambda item: _relative_path(self.base_dir, item['path']),
        )
        notebook.add(binary_frame, text=f"Binary ({len(self.binary_entries)})")


        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        ttk.Button(btn_frame, text="Select All", command=self._select_all).pack(side="left")
        ttk.Button(btn_frame, text="Clear All", command=self._clear_all).pack(side="left", padx=(8, 0))
        ttk.Button(btn_frame, text="OK", command=self._on_ok).pack(side="right")
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(side="right", padx=(8, 0))

    def _create_checklist_tab(self, notebook, title, key, items, label_func):
        frame = ttk.Frame(notebook)

        if not items:
            ttk.Label(frame, text="No items in this category.").pack(padx=12, pady=12, anchor="w")
            return frame

        control_frame = ttk.Frame(frame)
        control_frame.pack(fill="x", padx=4, pady=(6, 0))
        ttk.Button(control_frame, text="Select", command=lambda: self._set_all_for(key, True)).pack(side="left")
        ttk.Button(control_frame, text="Clear", command=lambda: self._set_all_for(key, False)).pack(side="left", padx=(6, 0))

        canvas = tk.Canvas(frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=6)
        scrollbar.pack(side="right", fill="y", padx=(0, 4), pady=6)

        for item in items:
            var = tk.BooleanVar(value=True)
            if key == "renames":
                # Custom row: checkbox + two-line colored labels (old in red, new in green)
                row = ttk.Frame(inner)
                row.pack(anchor="w", fill="x", padx=4, pady=4)

                cb = tk.Checkbutton(row, variable=var)
                cb.pack(side="left", anchor="n", padx=(0, 6))

                text_frame = ttk.Frame(row)
                text_frame.pack(side="left", fill="x", expand=True)

                prefix = "[DIR]" if item.get("is_dir") else "[FILE]"
                old_rel = _relative_path(self.base_dir, item["old"]) if isinstance(item.get("old"), str) else str(item.get("old"))
                new_rel = _relative_path(self.base_dir, item["new"]) if isinstance(item.get("new"), str) else str(item.get("new"))
                self._render_diff_lines(text_frame, prefix, old_rel, new_rel)
            else:
                cb = tk.Checkbutton(inner, text=label_func(item), variable=var, justify="left", anchor="w", wraplength=520)
                cb.pack(anchor="w", fill="x", padx=4, pady=2)
            self.vars[key].append((var, item))

        return frame

    def _render_diff_lines(self, parent, prefix, old_text, new_text, wraplength=520):
        """Render old/new lines with highlighted differences (old: red, new: green) aligned vertically."""
        sm = difflib.SequenceMatcher(None, old_text, new_text)
        opcodes = sm.get_opcodes()

        # Container with two columns: left markers, right content
        grid = ttk.Frame(parent)
        grid.pack(anchor="w", fill="x", padx=0, pady=0)

        left = ttk.Frame(grid)
        left.pack(side="left", anchor="n", padx=0, pady=0)
        right = ttk.Frame(grid)
        right.pack(side="left", fill="x", expand=True, padx=0, pady=0)

        # Left markers (fixed width for alignment)
        marker_width = max(len(prefix), 1)
        tk.Label(left, text=prefix, anchor="e", width=marker_width, bd=0, highlightthickness=0).pack(anchor="w", padx=0, pady=0)
        tk.Label(left, text="→", anchor="e", width=marker_width, bd=0, highlightthickness=0).pack(anchor="w", padx=0, pady=0)

        # Old line content
        line_old = ttk.Frame(right)
        line_old.pack(anchor="w", fill="x", padx=0, pady=0)
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                seg = old_text[i1:i2]
                if seg:
                    tk.Label(line_old, text=seg, anchor="w", wraplength=wraplength, bd=0, highlightthickness=0).pack(side="left", anchor="w", padx=0, pady=0)
            elif tag in ('replace', 'delete'):
                seg = old_text[i1:i2]
                if seg:
                    tk.Label(line_old, text=seg, fg="#c62828", anchor="w", wraplength=wraplength, bd=0, highlightthickness=0).pack(side="left", anchor="w", padx=0, pady=0)
            # inserts do not appear on the old line

        # New line content
        line_new = ttk.Frame(right)
        line_new.pack(anchor="w", fill="x", padx=0, pady=0)
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                seg = new_text[j1:j2]
                if seg:
                    tk.Label(line_new, text=seg, anchor="w", wraplength=wraplength, bd=0, highlightthickness=0).pack(side="left", anchor="w", padx=0, pady=0)
            elif tag in ('replace', 'insert'):
                seg = new_text[j1:j2]
                if seg:
                    tk.Label(line_new, text=seg, fg="#2e7d32", anchor="w", wraplength=wraplength, bd=0, highlightthickness=0).pack(side="left", anchor="w", padx=0, pady=0)
            # deletes do not appear on the new line

    def _on_parent_var_change(self):
        """Schedule a preview refresh when a parent input variable changes."""
        # Use schedule_preview to debounce repeated changes
        try:
            self.schedule_preview()
        except Exception:
            # If the dialog is closing or schedule_preview is unavailable,
            # ignore the callback silently.
            pass

    def _remove_parent_traces(self):
        """Remove variable traces registered on the parent."""
        if not hasattr(self, "_parent_traces") or not self._parent_traces:
            return
        for var, tid in list(self._parent_traces):
            try:
                var.trace_remove("write", tid)
            except Exception:
                pass
        self._parent_traces = []

    def _set_all_for(self, key, state):
        for var, _ in self.vars.get(key, []):
            var.set(state)

    def _select_all(self):
        for key in self.vars:
            self._set_all_for(key, True)

    def _clear_all(self):
        for key in self.vars:
            self._set_all_for(key, False)

    def _on_ok(self):
        self.result = {
            "renames": [item for var, item in self.vars["renames"] if var.get()],
            "jsons": [item for var, item in self.vars["jsons"] if var.get()],
            "binaries": [item for var, item in self.vars["binaries"] if var.get()],
        }
        # clean up any traces we registered on the parent before closing
        try:
            self._remove_parent_traces()
        except Exception:
            pass
        self.destroy()

    def _on_cancel(self):
        self.result = None
        # clean up traces then close
        try:
            self._remove_parent_traces()
        except Exception:
            pass
        self.destroy()

    def show(self):
        self.wait_window(self)
        return self.result


def replace_in_files(
    base_dir,
    old_padded,
    new_padded,
    old_stripped,
    new_stripped,
    slot_key,
    race_code,
    extensions,
    logger=print,
    allowed_json_paths=None,
    allowed_binary_paths=None,
    json_scope=None,
    delete_scope=None,
):
    """Replace item identifiers inside files by extension (.json uses numeric IDs; .mtrl/.mdl use e####)."""
    allowed_json_norm = None
    allowed_bin_norm = None
    if allowed_json_paths is not None:
        allowed_json_norm = _norm_set(allowed_json_paths)
    if allowed_binary_paths is not None:
        allowed_bin_norm = _norm_set(allowed_binary_paths)

    if allowed_json_norm == set() and allowed_bin_norm == set():
        logger("ℹ️ No files selected for content updates. Skipping.")
        return

    for root, _, files in os.walk(base_dir):
        for filename in files:
            ext = os.path.splitext(filename.lower())[1]
            if ext not in extensions:
                continue

            file_path = os.path.join(root, filename)
            norm_path = os.path.normcase(os.path.normpath(file_path))

            try:
                if ext == ".json":
                    if allowed_json_norm is not None and norm_path not in allowed_json_norm:
                        continue
                    # Parse JSON and modify 'Files' and 'Manipulations' with slot-aware logic
                    try:
                        with open(file_path, "r", encoding="utf-8-sig") as f:
                            data = json.load(f)
                    except Exception as e:
                        logger(f"❌ Error reading JSON {file_path}: {e}")
                        continue

                    new_data, file_modified, _ = _transform_json_data(
                        data,
                        old_padded,
                        new_padded,
                        old_stripped,
                        new_stripped,
                        slot_key,
                        race_code,
                        apply_changes=True,
                        logger=logger,
                        json_scope=json_scope,
                        delete_scope=delete_scope,
                    )

                    if file_modified:
                        try:
                            with open(file_path, "w", encoding="utf-8") as f:
                                json.dump(new_data, f, indent=2, ensure_ascii=False)
                            logger(f"Updated (text .json): {file_path}")
                        except Exception as e:
                            logger(f"❌ Error writing JSON {file_path}: {e}")

                else:
                    # For .mtrl and .mdl use e-prefixed, zero-padded forms.
                    # If an explicit allowed list is provided, honor it and skip slot filtering.
                    if allowed_bin_norm is not None:
                        if norm_path not in allowed_bin_norm:
                            continue
                    else:
                        # Fallback behavior: restrict to files that include the slot flag in filename
                        slot_flag = f"_{slot_key}".lower()
                        fname_low = os.path.basename(file_path).lower()
                        if slot_flag not in fname_low:
                            continue

                    old_sub = f"e{old_padded}"
                    new_sub = f"e{new_padded}"

                    old_bytes = old_sub.encode("latin-1", errors="ignore")
                    new_bytes = new_sub.encode("latin-1", errors="ignore")

                    with open(file_path, "rb") as f:
                        data = f.read()

                    if old_bytes in data:
                        new_data = data.replace(old_bytes, new_bytes)
                        with open(file_path, "wb") as f:
                            f.write(new_data)
                        logger(f"Updated (binary .mtrl/.mdl): {file_path}")

            except Exception as e:
                logger(f"❌ Error processing {file_path}: {e}")


def preview_changes(base_dir, old_padded, new_padded, old_stripped, new_stripped, slot_key, race_code, remove_other_races=False, seed_models=None, ignore_slot=False):
    """Compute planned changes by following the asset chain (.mdl → .mtrl → .tex).

    Returns (rename_changes, json_changes, binary_changes, asset_graph).
    """

    token_old_name = f"e{old_padded}"
    token_new_name = f"e{new_padded}"
    id_token = token_old_name.lower()
    slot_flag = f"_{slot_key}".lower()

    # Build a case-insensitive index of files once
    idx = _index_files_ci(base_dir)

    def _resolve_with_token_fallback(index, rel_paths):
        """Resolve relative paths with a fallback that maps new-token to old-token.

        Uses existing _resolve_many_ci for case-insensitive endswith resolution and
        merges results while preserving order.
        """
        resolved = []
        seen = set()
        for rp in rel_paths:
            # normalize for token checks but let _resolve_many_ci handle its own normalization
            rp_norm = rp.replace("\\", "/").lstrip("./").lstrip("/")
            candidates = [rp_norm]
            if token_new_name.lower() in rp_norm.lower():
                candidates.append(rp_norm.lower().replace(token_new_name, token_old_name))
            for cand in candidates:
                for ap in _resolve_many_ci(index, [cand]):
                    if ap not in seen:
                        seen.add(ap)
                        resolved.append(ap)
        return resolved

    # 1) Determine candidate .mdl files (seeded by user selection, else discovered by slot+id+race)
    if seed_models:
        # Use provided models directly (validate existence)
        mdl_candidates = [p for p in seed_models if os.path.isfile(p) and p.lower().endswith('.mdl')]
    else:
        if ignore_slot:
            # Find all .mdl files containing the old item ID, regardless of slot
            mdl_candidates = _find_models_by_id(base_dir, id_token)
        else:
            mdl_candidates = _find_candidate_models(base_dir, id_token, slot_key, race_code)

    # 2) From each .mdl extract .mtrl references and resolve to files
    mtrl_rel_paths = set()
    mdl_to_mtrl = {}
    binary_changes = []
    for mdl in mdl_candidates:
        try:
            with open(mdl, "rb") as f:
                b = f.read()
            if token_old_name.encode("latin-1") in b:
                binary_changes.append(mdl)
            mdl_mtrls = _extract_paths_from_bytes_generic(b, ".mtrl")
            mtrl_rel_paths.update(mdl_mtrls)
            # resolve per-mdl now for accurate graph
            resolved_mtrls = _resolve_with_token_fallback(idx, mdl_mtrls)
            mdl_to_mtrl[mdl] = resolved_mtrls
        except Exception:
            continue
    mtrl_files = _resolve_with_token_fallback(idx, list(mtrl_rel_paths))

    # 3) From each .mtrl extract .tex references and resolve
    tex_rel_paths = set()
    mtrl_to_tex = {}
    for mtrl in mtrl_files:
        try:
            with open(mtrl, "rb") as f:
                b = f.read()
            if token_old_name.encode("latin-1") in b:
                binary_changes.append(mtrl)
            mtrl_tex = _extract_paths_from_bytes_generic(b, ".tex")
            tex_rel_paths.update(mtrl_tex)
            resolved_tex = _resolve_with_token_fallback(idx, mtrl_tex)
            mtrl_to_tex[mtrl] = resolved_tex
        except Exception:
            continue
    tex_files = _resolve_with_token_fallback(idx, list(tex_rel_paths))
    binary_changes = list(dict.fromkeys(binary_changes))

    # 4) Plan renames for files (mdl, mtrl, tex) whose names include the ID token
    def _maybe_file_rename(p):
        base = os.path.basename(p)
        if token_old_name in base:
            return p, os.path.join(os.path.dirname(p), base.replace(token_old_name, token_new_name))
        return None

    rename_changes = []
    for p in mdl_candidates + mtrl_files + tex_files:
        rn = _maybe_file_rename(p)
        if rn:
            rename_changes.append(rn)

    # 5) Plan directory renames only along the asset paths (ancestors of involved files)
    dir_renames = []
    asset_paths = mdl_candidates + mtrl_files + tex_files
    base_norm = os.path.normcase(os.path.normpath(base_dir))
    seen_dirs = set()
    for p in asset_paths:
        cur = os.path.dirname(p)
        while True:
            if not cur:
                break
            cur_norm = os.path.normcase(os.path.normpath(cur))
            # stop if we reached outside base or just above base
            try:
                common = os.path.commonpath([base_norm, cur_norm])
            except Exception:
                break
            if common != base_norm:
                break
            dname = os.path.basename(cur)
            if token_old_name in dname and cur_norm not in seen_dirs:
                newd = os.path.join(os.path.dirname(cur), dname.replace(token_old_name, token_new_name))
                dir_renames.append((cur, newd))
                seen_dirs.add(cur_norm)
            if cur_norm == base_norm:
                break
            cur = os.path.dirname(cur)

    # Expand directory renames using JSON Files values that include the old token.
    extra_dir_pairs = []
    json_resolved_files = set()
    json_file_cache = []  # (path, data) to avoid double reads
    for r, _d, fs in os.walk(base_dir):
        for fn in fs:
            if not fn.lower().endswith(".json"):
                continue
            path = os.path.join(r, fn)
            try:
                with open(path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
            except Exception:
                continue
            json_file_cache.append((path, data))
            # Look into Options -> Files and FileSwaps values
            try:
                opts = data.get("Options", [])
                if isinstance(opts, list):
                    for opt in opts:
                        if not isinstance(opt, dict):
                            continue
                        for dict_key in ("Files", "FileSwaps"):
                            m = opt.get(dict_key)
                            if not isinstance(m, dict):
                                continue
                            for _k, v in m.items():
                                if isinstance(v, str) and token_old_name in v.lower():
                                    # Build absolute path for the value and add directory renames for any component containing the token
                                    rel_v = v.replace("\\", "/").lstrip("./").lstrip("/")
                                    abs_v = os.path.normpath(os.path.join(base_dir, rel_v))
                                    # Walk ancestors from abs_v up to base_dir
                                    cur = abs_v
                                    while True:
                                        parent = os.path.dirname(cur)
                                        if not parent or os.path.normcase(os.path.normpath(parent)) == os.path.normcase(os.path.normpath(cur)):
                                            break
                                        # Stop once we go above base_dir
                                        try:
                                            common = os.path.commonpath([os.path.normpath(base_dir), os.path.normpath(parent)])
                                        except Exception:
                                            break
                                        if common != os.path.normpath(base_dir):
                                            break
                                        name = os.path.basename(parent)
                                        if token_old_name in name:
                                            newd = os.path.join(os.path.dirname(parent), name.replace(token_old_name, token_new_name))
                                            extra_dir_pairs.append((parent, newd))
                                        if os.path.normcase(os.path.normpath(parent)) == os.path.normcase(os.path.normpath(base_dir)):
                                            break
                                        cur = parent
                                    # Also try to resolve concrete file(s) referenced by JSON value to include in asset operations
                                    try:
                                        resolved = _resolve_with_token_fallback(idx, [v])
                                        for rp in resolved:
                                            json_resolved_files.add(rp)
                                    except Exception:
                                        pass
            except Exception:
                pass

    # Merge and deduplicate directory renames
    all_dir_pairs = dir_renames + extra_dir_pairs
    # Deduplicate while preserving order
    seen_dp = set()
    dedup_dir_pairs = []
    for a, b in all_dir_pairs:
        key = (os.path.normcase(os.path.normpath(a)), os.path.normcase(os.path.normpath(b)))
        if key in seen_dp:
            continue
        seen_dp.add(key)
        dedup_dir_pairs.append((a, b))

    # Count JSON changes constrained by planned renames and JSON-derived dir renames.
    json_changes = []
    json_scope = _build_json_scope(base_dir, rename_pairs=rename_changes + dedup_dir_pairs)
    # No deletion scope during preview; race-based deletion is only applied during convert
    delete_scope = None
    for path, data in json_file_cache:
        try:
            _, _, cnt = _transform_json_data(
                data,
                old_padded,
                new_padded,
                old_stripped,
                new_stripped,
                slot_key,
                race_code,
                apply_changes=False,
                logger=lambda *_: None,
                json_scope=json_scope,
                delete_scope=delete_scope,
            )
        except Exception:
            cnt = 0
        if cnt:
            json_changes.append((path, cnt))

    # Combine file and directory renames
    rename_all = []
    if remove_other_races:
        # Keep directory renames when deleting other races; safe to move the whole tree
        for a, b in rename_changes + dedup_dir_pairs:
            rename_all.append({"old": a, "new": b, "is_dir": True}) if os.path.isdir(a) else rename_all.append({"old": a, "new": b, "is_dir": False})
    else:
        # When not deleting other races, avoid directory-wide changes; prefer per-file operations.
        def extract_race_marker(p: str):
            s = p.replace("\\", "/")
            return _extract_race_code_from_name(s)

        asset_paths = list(dict.fromkeys(mdl_candidates + mtrl_files + tex_files + list(json_resolved_files)))
        seen_old = set()
        for p in asset_paths:
            # compute full-path replacement (dir + file) for the token
            if token_old_name not in os.path.basename(p) and token_old_name not in os.path.dirname(p):
                continue
            new_full = p.replace(token_old_name, token_new_name)
            if new_full == p:
                continue
            if p in seen_old:
                continue
            seen_old.add(p)
            rc = extract_race_marker(p)
            # If an explicit race marker exists and differs, leave the path unchanged.
            if rc and rc != race_code:
                continue
            # Otherwise (selected race or no explicit race), move into the new e[id] path
            action = "move"
            rename_all.append({"old": p, "new": new_full, "is_dir": False, "action": action})

    asset_graph = {"mdl_to_mtrl": mdl_to_mtrl, "mtrl_to_tex": mtrl_to_tex}
    return rename_all, json_changes, binary_changes, asset_graph


def _find_candidate_models(base_dir: str, id_token_lower: str, slot_key: str, race_code: str):
    """Return a list of .mdl files that match the given item id token, slot, and (if present) race."""
    out = []
    for r, _d, fs in os.walk(base_dir):
        for fn in fs:
            if not fn.lower().endswith('.mdl'):
                continue
            full = os.path.join(r, fn)
            fl = fn.lower()
            if id_token_lower not in fl:
                continue
            if not _path_matches_slot(full, slot_key):
                continue
            # Enforce race if race marker exists in filename
            enforce_race = False
            for i in range(len(fl) - 4):
                if fl[i] == 'c' and fl[i+1:i+5].isdigit():
                    enforce_race = True
                    break
            if enforce_race and f"c{race_code}".lower() not in fl:
                continue
            out.append(full)
    return out


def _find_models_by_id(base_dir: str, id_token_lower: str):
    """Return all .mdl files containing the given id token (case-insensitive)."""
    out = []
    for r, _d, fs in os.walk(base_dir):
        for fn in fs:
            if fn.lower().endswith('.mdl') and id_token_lower in fn.lower():
                out.append(os.path.join(r, fn))
    return out

class PenumbraConverterApp(tk.Tk):
    SLOT_OPTIONS = [
        ("Head (met)", "met"),
        ("Body (top)", "top"),
        ("Hands (glv)", "glv"),
        ("Legs (dwn)", "dwn"),
        ("Feet (sho)", "sho"),
        ("Earring (ear)", "ear"),
        ("Neck (nek)", "nek"),
        ("Wrists (wrs)", "wrs"),
        ("Ring Right (rir)", "rir"),
        ("Ring Left (ril)", "ril"),
    ]

    # Race options mapped to 4-digit race codes
    RACE_OPTIONS = [
        ("Midlander Male (0101)", "0101"),
        ("Midlander Female (0201)", "0201"),
        ("Highlander Male (0301)", "0301"),
        ("Highlander Female (0401)", "0401"),
        ("Elezen Male (0501)", "0501"),
        ("Elezen Female (0601)", "0601"),
        ("Miqo'te Male (0701)", "0701"),
        ("Miqo'te Female (0801)", "0801"),
        ("Roegadyn Male (0901)", "0901"),
        ("Roegadyn Female (1001)", "1001"),
        ("Lalafell Male (1101)", "1101"),
        ("Lalafell Female (1201)", "1201"),
        ("Au Ra Male (1301)", "1301"),
        ("Au Ra Female (1401)", "1401"),
        ("Hrothgar Male (1501)", "1501"),
        ("Hrothgar Female (1601)", "1601"),
        ("Viera Male (1701)", "1701"),
        ("Viera Female (1801)", "1801"),
    ]

    def __init__(self):
        super().__init__()
        self.title("Advanced Penumbra Item Converter")
        self.minsize(720, 520)

        self.dir_var = tk.StringVar()
        self.old_id_var = tk.StringVar()
        self.new_id_var = tk.StringVar()
        # Default to Body slot
        self.slot_var = tk.StringVar(value=self.SLOT_OPTIONS[1][0])
        self.race_var = tk.StringVar(value=self.RACE_OPTIONS[1][0])
        self.ignore_slot_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Select a Penumbra mod directory to begin.")

        self.last_preview = None
        self.preview_job = None
        self.suppress_preview = False

        self._build_ui()

    # Update the bottom hint once both item IDs are entered
        try:
            self.old_id_var.trace_add("write", lambda *a: self._on_id_vars_changed())
            self.new_id_var.trace_add("write", lambda *a: self._on_id_vars_changed())
        except Exception:
            pass

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self, padding=12)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)

        ttk.Label(main_frame, text="Target Directory:").grid(row=0, column=0, sticky="w", pady=(0, 8))
        dir_entry = ttk.Entry(main_frame, textvariable=self.dir_var)
        dir_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        ttk.Button(main_frame, text="Browse...", command=self.browse_directory).grid(row=0, column=2, pady=(0, 8))

        id_frame = ttk.Frame(main_frame)
        id_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(id_frame, text="Original Item ID:").grid(row=0, column=0, sticky="w")
        ttk.Entry(id_frame, textvariable=self.old_id_var, width=12).grid(row=0, column=1, sticky="w", padx=(4, 16))
        ttk.Label(id_frame, text="Target Item ID:").grid(row=0, column=2, sticky="w")
        ttk.Entry(id_frame, textvariable=self.new_id_var, width=12).grid(row=0, column=3, sticky="w", padx=(4, 0))

        slot_frame = ttk.Frame(main_frame)
        slot_frame.grid(row=2, column=0, columnspan=3, sticky="w")
        ttk.Label(slot_frame, text="Item Slot:").grid(row=0, column=0, sticky="w")
        slot_values = [label for label, _ in self.SLOT_OPTIONS]
        self.slot_combo = ttk.Combobox(slot_frame, textvariable=self.slot_var, values=slot_values, state="readonly", width=24)
        # Select Body by default
        self.slot_combo.current(1)
        self.slot_combo.grid(row=0, column=1, sticky="w", padx=(4, 0))

        race_frame = ttk.Frame(main_frame)
        race_frame.grid(row=2, column=2, sticky="w")
        ttk.Label(race_frame, text="Race:").grid(row=0, column=0, sticky="w")
        race_values = [label for label, _ in self.RACE_OPTIONS]
        self.race_combo = ttk.Combobox(race_frame, textvariable=self.race_var, values=race_values, state="readonly", width=28)
        self.race_combo.current(1)
        self.race_combo.grid(row=0, column=1, sticky="w", padx=(4, 0))

        # Ignore slot restrictions checkbox
        options_frame = ttk.Frame(main_frame)
        options_frame.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        tk.Checkbutton(
            options_frame, 
            text="Ignore slot restrictions (find all files with old item ID)", 
            variable=self.ignore_slot_var,
            command=self.schedule_preview
        ).grid(row=0, column=0, sticky="w")

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Button(button_frame, text="Convert Item", command=self.run_action).grid(row=0, column=0)

        self.log_text = ScrolledText(main_frame, wrap="word", state="disabled")
        self.log_text.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(12, 0))

        # Left: status text
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # Right: © Luci and clickable links
        footer = ttk.Frame(main_frame)
        footer.grid(row=6, column=2, sticky="e", pady=(8, 0))
        tk.Label(footer, text="© Luci").pack(side="left", padx=(0, 8))
        # Clickable link symbols
        self._create_link(footer, "XMA", "https://www.xivmodarchive.com/user/124593").pack(side="left", padx=(0, 6))
        self._create_link(footer, "𝕏", "https://x.com/Luci__xiv").pack(side="left", padx=(0, 6))
        self._create_link(footer, "☕", "https://ko-fi.com/Luci_xiv").pack(side="left")

    def _create_link(self, parent, text, url):
        lbl = tk.Label(parent, text=text, fg="#1a73e8", cursor="hand2")
        lbl.bind("<Button-1>", lambda _e, u=url: webbrowser.open(u))
        return lbl

    def browse_directory(self):
        initial_dir = self.dir_var.get() or os.getcwd()
        path = filedialog.askdirectory(initialdir=initial_dir)
        if path:
            self.dir_var.set(path)
            self.set_status(f"Selected directory: {path}")
            self.schedule_preview()

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.configure(state="disabled")
        self.log_text.see(tk.END)

    def set_status(self, message):
        self.status_var.set(message)

    def _on_id_vars_changed(self):
        """Guide the user after both item IDs are present without overriding specific statuses."""
        old_val = (self.old_id_var.get() or "").strip()
        new_val = (self.new_id_var.get() or "").strip()
        if not (old_val and new_val):
            return
        current = self.status_var.get() or ""
        if current.startswith("Select a Penumbra mod directory") or current.startswith("Waiting for both original and target item IDs."):
            self.set_status('Select correct slot and race, then hit "Convert Item".')

    def _set_var_safely(self, var, value):
        if var.get() == value:
            return
        self.suppress_preview = True
        try:
            var.set(value)
        finally:
            self.suppress_preview = False

    def _on_inputs_changed(self, *_):
        if self.suppress_preview:
            return
        self.schedule_preview()

    def schedule_preview(self, delay=600):
        if self.preview_job is not None:
            self.after_cancel(self.preview_job)
        self.preview_job = self.after(delay, self._auto_preview)

    def _auto_preview(self):
        self.preview_job = None
        if not self.winfo_exists():
            return
        self._run_preview(auto=True)

    def _parse_id(self, value, label):
        value = value.strip()
        if not value:
            raise ValueError(f"{label} item ID is required.")
        if not value.isdigit():
            raise ValueError(f"{label} item ID must be numeric and up to 4 digits.")
        if len(value) > 4:
            raise ValueError(f"{label} item ID must be at most 4 digits.")
        return value.zfill(4)

    def _get_slot_selection(self):
        selection = self.slot_var.get()
        for label, key in self.SLOT_OPTIONS:
            if label == selection:
                return key, label
        raise ValueError("Invalid slot selection.")

    def _get_race_selection(self):
        selection = self.race_var.get()
        for label, code in self.RACE_OPTIONS:
            if label == selection:
                return code, label
        raise ValueError("Invalid race selection.")

    def collect_inputs(self, show_errors=True):
        directory = self.dir_var.get().strip()
        if not directory:
            if show_errors:
                messagebox.showerror("Missing Directory", "Please select a target directory.")
            self.set_status("Missing directory.")
            return None

        raw_old = self.old_id_var.get().strip()
        raw_new = self.new_id_var.get().strip()

        if not raw_old or not raw_new:
            if show_errors:
                messagebox.showerror("Missing Item ID", "Enter both original and target item IDs.")
            self.set_status("Waiting for both original and target item IDs.")
            return None

        try:
            old_padded = self._parse_id(raw_old, "Original")
            new_padded = self._parse_id(raw_new, "Target")
        except ValueError as exc:
            if show_errors:
                messagebox.showerror("Invalid Item ID", str(exc))
            self.set_status("Invalid item ID provided.")
            return None

        slot_key, slot_label = self._get_slot_selection()
        race_code, race_label = self._get_race_selection()

        old_stripped = str(int(old_padded))
        new_stripped = str(int(new_padded))

        self._set_var_safely(self.old_id_var, old_padded)
        self._set_var_safely(self.new_id_var, new_padded)

        return {
            "directory": directory,
            "old_padded": old_padded,
            "new_padded": new_padded,
            "old_stripped": old_stripped,
            "new_stripped": new_stripped,
            "slot_key": slot_key,
            "slot_label": slot_label,
            "race_code": race_code,
            "race_label": race_label,
            "ignore_slot": self.ignore_slot_var.get(),
        }

    def preview_action(self):
        self._run_preview(auto=False)

    def _run_preview(self, auto):
        if self.preview_job is not None:
            self.after_cancel(self.preview_job)
            self.preview_job = None

        inputs = self.collect_inputs(show_errors=not auto)
        if not inputs:
            if auto:
                self.last_preview = None
            return

        self.set_status("Updating preview..." if auto else "Running preview...")

        self.clear_log()

        try:
            verified_dir = verify_target_path(inputs["directory"], logger=self.log)
        except Exception as exc:
            self.log(str(exc))
            if not auto:
                messagebox.showerror("Invalid Directory", str(exc))
            self.set_status("Preview unavailable.")
            self.last_preview = None
            return

        self.log("🔍 Previewing changes...")
        seed_models = getattr(self, 'selected_models_for_run', None)
        renames, jsons, binaries, asset_graph = preview_changes(
            verified_dir,
            inputs["old_padded"],
            inputs["new_padded"],
            inputs["old_stripped"],
            inputs["new_stripped"],
            inputs["slot_key"],
            inputs["race_code"],
            False,
            seed_models=seed_models,
            ignore_slot=inputs["ignore_slot"],
        )

        # If a model selection exists for this run, filter preview to those models and their chains
        selected_models = getattr(self, 'selected_models_for_run', None)
        if selected_models is not None:
            try:
                selected_set = { _norm(p) for p in selected_models }
                # Keep only rename entries directly on selected models or on assets discovered from them
                graph = asset_graph or {"mdl_to_mtrl": {}, "mtrl_to_tex": {}}
                mdl_to_mtrl = graph.get("mdl_to_mtrl", {})
                mtrl_to_tex = graph.get("mtrl_to_tex", {})

                # Build closure of assets reachable from selected models
                keep_files = set()
                for mdl, mtrls in mdl_to_mtrl.items():
                    if _norm(mdl) in selected_set:
                        keep_files.add(_norm(mdl))
                        for m in mtrls:
                            keep_files.add(_norm(m))
                            for t in mtrl_to_tex.get(m, []):
                                keep_files.add(_norm(t))

                def _entry_kept(entry):
                    oldp = entry.get('old') if isinstance(entry, dict) else entry[0]
                    return _norm(oldp) in keep_files

                renames = [e for e in renames if _entry_kept(e)]
                # 'binaries' is a list of file path strings at this stage
                binaries = [b for b in binaries if _norm(b) in keep_files]
            except Exception as e:
                self.log(f"⚠️ Model filtering failed: {e}")
            # JSON preview count stays, but we will scope JSON edits later using rename scope

        # preview_changes now returns annotated rename entries ready for UI
        rename_entries = []
        for it in renames:
            if isinstance(it, dict):
                # Ensure is_dir field if missing
                ent = dict(it)
                if "is_dir" not in ent:
                    ent["is_dir"] = os.path.isdir(ent.get("old", ""))
                rename_entries.append(ent)
            else:
                oldp, newp = it
                rename_entries.append({"old": oldp, "new": newp, "is_dir": os.path.isdir(oldp)})
        json_entries = [{"path": path, "count": cnt} for path, cnt in jsons]
        binary_entries = [{"path": path} for path in binaries]

        self._display_preview(inputs, rename_entries, json_entries, binary_entries, asset_graph)

        if not rename_entries and not json_entries and not binary_entries:
            # For auto previews, do not open the dialog; for manual, still show the checklist (empty tabs)
            self.last_preview = {
                **inputs,
                "directory": verified_dir,
                "renames": rename_entries,
                "jsons": json_entries,
                "binaries": binary_entries,
                "selected": {
                    "renames": [],
                    "jsons": [],
                    "binaries": [],
                },
            }
            if auto:
                self.set_status("Preview complete: no changes detected for the selected slot.")
                return

        if auto:
            # Save snapshot without opening dialog; dialog will be shown when running conversion
            self.last_preview = {
                **inputs,
                "directory": verified_dir,
                "renames": rename_entries,
                "jsons": json_entries,
                "binaries": binary_entries,
                "selected": {
                    "renames": [],
                    "jsons": [],
                    "binaries": [],
                },
                "json_scope": _build_json_scope(verified_dir, rename_entries=rename_entries),
            }
            summary = (
                f"Preview complete: {len(rename_entries)} rename(s), {len(json_entries)} JSON file(s), {len(binary_entries)} binary file(s)"
            )
            self.log(summary)
            self.set_status(summary)
            return

        self.log("Opening preview checklist...")
        # Always open the dialog during manual runs to let user confirm even empty selections
        selection = self._open_preview_dialog(
            verified_dir, rename_entries, json_entries, binary_entries
        )

        if selection is None:
            if auto:
                self.log("Preview cancelled. Adjust the values and try again.")
            else:
                self.log("Preview cancelled by user.")
            self.set_status("Preview cancelled.")
            self.last_preview = None
            return

        selected_dir_count = sum(1 for entry in selection["renames"] if entry["is_dir"])
        selected_file_count = len(selection["renames"]) - selected_dir_count

        deselected_renames = len(rename_entries) - len(selection["renames"])
        deselected_json = len(json_entries) - len(selection["jsons"])
        deselected_bin = len(binary_entries) - len(selection["binaries"])

        if deselected_renames or deselected_json or deselected_bin:
            self.log(
                f"Deselected: {deselected_renames} rename(s), {deselected_json} JSON file(s), {deselected_bin} binary file(s)."
            )

        # Ensure directory saved is the verified absolute path to avoid false mismatches.
        self.last_preview = {
            **inputs,
            "directory": verified_dir,
            "renames": rename_entries,
            "jsons": json_entries,
            "binaries": binary_entries,
            "selected": selection,
            "json_scope": _build_json_scope(verified_dir, rename_entries=rename_entries),
        }

        summary = (
            f"Preview complete: {len(selection['renames'])} selected rename(s) "
            f"({selected_dir_count} dir, {selected_file_count} file), "
            f"{len(selection['jsons'])} selected JSON file(s), {len(selection['binaries'])} selected binary file(s)."
        )
        self.log(summary)
        self.set_status(summary)

    def _display_preview(self, inputs, renames, jsons, binaries, asset_graph=None):
        base = inputs["directory"].rstrip("\\/")

        self.log(
            f"Input (padded): old={inputs['old_padded']}, new={inputs['new_padded']}"
        )
        self.log(
            f"Input (stripped for .json): old={inputs['old_stripped']}, new={inputs['new_stripped']}"
        )

        dir_count = sum(1 for entry in renames if entry["is_dir"])
        file_count = len(renames) - dir_count
        self.log(
            f"Rename targets: {len(renames)} (dirs: {dir_count}, files: {file_count})"
        )
        for entry in renames[:5]:
            prefix = "[DIR]" if entry["is_dir"] else "[FILE]"
            self.log(
                f"  • {prefix} {_relative_path(base, entry['old'])} → {_relative_path(base, entry['new'])}"
            )
        if len(renames) > 5:
            self.log(f"  • …and {len(renames) - 5} more (see checklist)")

        self.log(f"JSON edits: {len(jsons)} file(s)")
        for entry in jsons[:5]:
            self.log(
                f"  • {_relative_path(base, entry['path'])} (matches: {entry['count']})"
            )
        if len(jsons) > 5:
            self.log(f"  • …and {len(jsons) - 5} more (see checklist)")

        self.log(f"Binary edits: {len(binaries)} file(s)")
        for entry in binaries[:5]:
            self.log(f"  • {_relative_path(base, entry['path'])}")
        if len(binaries) > 5:
            self.log(f"  • …and {len(binaries) - 5} more (see checklist)")


        # Asset chain summary (mdl -> mtrl -> tex)
        if asset_graph:
            mdl_to_mtrl = asset_graph.get("mdl_to_mtrl", {})
            mtrl_to_tex = asset_graph.get("mtrl_to_tex", {})
            if mdl_to_mtrl:
                self.log("Asset chain summary:")
                mdl_keys = list(mdl_to_mtrl.keys())
                for mdl in mdl_keys[:3]:
                    mdl_rel = _relative_path(base, mdl)
                    mtrls = mdl_to_mtrl.get(mdl, [])
                    self.log(f"  • {mdl_rel} → {len(mtrls)} material(s)")
                    for m in mtrls[:3]:
                        m_rel = _relative_path(base, m)
                        texs = mtrl_to_tex.get(m, [])
                        self.log(f"      - {m_rel} → {len(texs)} texture(s)")
                if len(mdl_keys) > 3:
                    self.log(f"  • …and {len(mdl_keys) - 3} more models")

        self.log("Toggle entries in the checklist before running the conversion if needed.")

    def _open_preview_dialog(self, base_dir, renames, json_entries, binary_entries):
        dialog = PreviewDialog(self, base_dir, renames, json_entries, binary_entries)
        return dialog.show()

    def run_action(self):
        inputs = self.collect_inputs()
        if not inputs:
            return

        try:
            verified_dir = verify_target_path(inputs["directory"], logger=self.log)
        except Exception as exc:
            self.log(str(exc))
            messagebox.showerror("Invalid Directory", str(exc))
            self.set_status("Conversion aborted.")
            return

        # Build candidate model list and ask user which to include
        try:
            token_old_name = f"e{inputs['old_padded']}"
            id_token = token_old_name.lower()
            if inputs["ignore_slot"]:
                # When ignoring slot restrictions, find all models with the old item ID
                mdl_candidates = _find_models_by_id(verified_dir, id_token)
            else:
                mdl_candidates = _find_candidate_models(verified_dir, id_token, inputs['slot_key'], inputs['race_code'])
        except Exception as e:
            self.log(f"⚠️ Failed to enumerate models: {e}")
            mdl_candidates = []

        if not mdl_candidates:
            # Fallback: allow manual selection from any models that match the item ID (any slot)
            any_id_models = _find_models_by_id(verified_dir, id_token)
            sel = ModelSelectDialog(self, verified_dir, any_id_models).show()
        else:
            sel = ModelSelectDialog(self, verified_dir, mdl_candidates).show()

        if sel is None:
            self.set_status("Conversion cancelled.")
            return
        selected_models = sel

        # Save selection for preview/filtering downstream
        self.selected_models_for_run = selected_models

        # Open the preview checklist now to capture fine-grained selection
        self._run_preview(auto=False)
        if not self.last_preview or not self.last_preview.get("selected"):
            self.set_status("Conversion cancelled.")
            return

        # Determine selected items (if any). If nothing is selected, show an
        # informational prompt and abort instead of asking for confirmation.
        selection = self.last_preview.get("selected") if self.last_preview else None

        total_selected = (
            len(selection.get("renames", []))
            + len(selection.get("jsons", []))
            + len(selection.get("binaries", []))
        )

        if total_selected == 0:
            messagebox.showinfo(
                "No Changes Selected",
                "No files or directories are selected for conversion.\nPlease run Preview and select items first.",
            )
            self.set_status("Conversion cancelled: nothing selected.")
            return

        proceed = messagebox.askyesno(
            "Confirm Conversion",
            "This will rename files and update Penumbra metadata. Continue?",
        )
        if not proceed:
            self.set_status("Conversion cancelled.")
            return

        self.log("🚀 Running conversion...")

        try:
            selection = self.last_preview.get("selected") if self.last_preview else None
            if not selection:
                selection = {
                    "renames": self.last_preview.get("renames", []) if self.last_preview else [],
                    "jsons": self.last_preview.get("jsons", []) if self.last_preview else [],
                    "binaries": self.last_preview.get("binaries", []) if self.last_preview else [],
                }

            allowed_rename_paths = [item["old"] for item in selection["renames"]]
            allowed_json_paths = [item["path"] for item in selection["jsons"]]
            allowed_binary_paths = [item["path"] for item in selection["binaries"]]
            json_scope = None
            if self.last_preview:
                # Prefer a scope computed using the user's selection (more restrictive)
                json_scope = _build_json_scope(verified_dir, rename_entries=selection.get("renames", []))
                # Fallback to stored one if any
                if not json_scope.get("files") and not json_scope.get("dirs"):
                    json_scope = self.last_preview.get("json_scope")

            # Further constrain scope using explicitly selected models (if any)
            selected_models = getattr(self, 'selected_models_for_run', None)
            if selected_models:
                # Merge model paths into the JSON scope directories so path replacements stay limited
                for mp in selected_models:
                    rel = _make_rel_norm(mp, verified_dir)
                    if json_scope is None:
                        json_scope = {"files": set(), "dirs": set()}
                    json_scope.setdefault("dirs", set()).add(os.path.dirname(rel))

            # No pre-conversion deletion. Build delete_scope=None for JSON processing.
            delete_scope = None

            # Apply explicit renames/copied first
            apply_explicit_renames_entries(selection["renames"], self.log)

            # Transform allowed binary paths to destination paths when a rename/copy exists
            if selection.get("renames"):
                mapping = {it["old"]: it["new"] for it in selection["renames"] if not it.get("is_dir")}
                transformed_binaries = []
                for p in allowed_binary_paths:
                    transformed_binaries.append(mapping.get(p, p))
                allowed_binary_paths = transformed_binaries

            replace_in_files(
                verified_dir,
                inputs["old_padded"],
                inputs["new_padded"],
                inputs["old_stripped"],
                inputs["new_stripped"],
                inputs["slot_key"],
                inputs["race_code"],
                extensions={".json", ".mtrl", ".mdl"},
                logger=self.log,
                allowed_json_paths=allowed_json_paths,
                allowed_binary_paths=allowed_binary_paths,
                json_scope=json_scope,
                delete_scope=None,
            )

            # Optional post-conversion cleanup of wrong-race assets
            try:
                plan = plan_cleanup_wrong_race_assets(verified_dir, inputs["old_padded"], inputs["new_padded"], inputs["race_code"], logger=self.log)
                total = len(plan.get("models", [])) + len(plan.get("materials", [])) + len(plan.get("textures", []))
                if total > 0:
                    proceed_cleanup = messagebox.askyesno(
                        "Post-Conversion Cleanup",
                        f"Found {len(plan['models'])} model(s), {len(plan['materials'])} material(s), {len(plan['textures'])} texture(s)\n"
                        f"from other races that are not required by the converted assets.\n\n"
                        f"Do you want to delete these and clean up JSON references?",
                    )
                    if proceed_cleanup:
                        apply_cleanup_wrong_race_assets(verified_dir, plan, logger=self.log)
                        self.log("✅ Cleanup complete.")
            except Exception as e:
                self.log(f"⚠️ Cleanup planning failed: {e}")
        except Exception as exc:
            self.log(f"❌ Unexpected error: {exc}")
            messagebox.showerror("Conversion Failed", str(exc))
            self.set_status("Conversion failed.")
            return

        self.last_preview = None

        self.log("✅ Conversion complete.")
        self.set_status("Conversion complete.")
        messagebox.showinfo(
            "Conversion Complete",
            "Successfully completed conversion.\nClick \"Reload Mod\" in the \"Edit Mod\" tab in Penumbra and equip the new item to see the changes in-game."
        )


class ModelSelectDialog(tk.Toplevel):
    def __init__(self, parent, base_dir, mdl_paths):
        super().__init__(parent)
        self.title("Select Models")
        self.parent = parent
        self.base_dir = base_dir.rstrip("\\/")
        self.mdl_paths = mdl_paths
        self.vars = []
        self.result = None

        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.resizable(True, True)
        self.minsize(520, 380)
        self.focus()

    def _build_ui(self):
        frame = ttk.Frame(self, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        ttk.Label(frame, text="Choose which models to include in this conversion:").pack(anchor="w")

        canvas = tk.Canvas(frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, pady=(8, 6))
        scrollbar.pack(side="right", fill="y", padx=(0, 4), pady=(8, 6))

        for p in self.mdl_paths:
            var = tk.BooleanVar(value=True)
            rel = _relative_path(self.base_dir, p)
            cb = tk.Checkbutton(inner, text=rel, variable=var, anchor="w", justify="left", wraplength=480)
            cb.pack(anchor="w", fill="x")
            self.vars.append((var, p))

        btns = ttk.Frame(self)
        btns.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        ttk.Button(btns, text="Select All", command=self._select_all).pack(side="left")
        ttk.Button(btns, text="Clear All", command=self._clear_all).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="OK", command=self._on_ok).pack(side="right")
        ttk.Button(btns, text="Cancel", command=self._on_cancel).pack(side="right", padx=(8, 0))

    def _select_all(self):
        for v, _ in self.vars:
            v.set(True)

    def _clear_all(self):
        for v, _ in self.vars:
            v.set(False)

    def _on_ok(self):
        selected = [p for v, p in self.vars if v.get()]
        self.result = selected
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()

    def show(self):
        self.wait_window(self)
        return self.result


def apply_explicit_renames_entries(rename_entries, logger):
    if not rename_entries:
        return
    files = [e for e in rename_entries if not e.get("is_dir")]
    dirs = [e for e in rename_entries if e.get("is_dir")]

    seen = set()
    def _dedup(seq):
        out = []
        for it in seq:
            oldp = it.get("old")
            if oldp in seen:
                continue
            seen.add(oldp)
            out.append(it)
        return out

    files = _dedup(files)
    dirs = _dedup(dirs)

    # Files first
    for it in files:
        oldp, newp = it.get("old"), it.get("new")
        action = it.get("action", "move")
        try:
            if not os.path.exists(oldp):
                logger(f"⚠️ Skipping rename (missing): {oldp}")
                continue
            if os.path.exists(newp):
                logger(f"⚠️ Skipping {action} (target exists): {newp}")
                continue
            os.makedirs(os.path.dirname(newp), exist_ok=True)
            if action == "copy":
                try:
                    shutil.copy2(oldp, newp)
                    logger(f"Copied file: {oldp} -> {newp}")
                except Exception as e:
                    logger(f"❌ Error copying file {oldp} -> {newp}: {e}")
            else:
                os.rename(oldp, newp)
                logger(f"Renamed file: {oldp} -> {newp}")
        except Exception as e:
            logger(f"❌ Error renaming file {oldp} -> {newp}: {e}")

    # Then directories (deepest first)
    dirs_sorted = sorted(dirs, key=lambda it: it.get("old", "").count(os.sep), reverse=True)
    for it in dirs_sorted:
        oldp, newp = it.get("old"), it.get("new")
        try:
            if not os.path.isdir(oldp):
                logger(f"⚠️ Skipping dir rename (missing): {oldp}")
                continue
            if os.path.exists(newp):
                logger(f"⚠️ Skipping dir rename (target exists): {newp}")
                continue
            parent = os.path.dirname(newp)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            os.rename(oldp, newp)
            logger(f"Renamed directory: {oldp} -> {newp}")
        except Exception as e:
            logger(f"❌ Error renaming directory {oldp} -> {newp}: {e}")


if __name__ == "__main__":
    app = PenumbraConverterApp()
    if len(sys.argv) > 1:
        app.dir_var.set(sys.argv[1])
        app.schedule_preview()
    app.mainloop()
