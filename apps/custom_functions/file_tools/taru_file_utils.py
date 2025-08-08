# custom_functions/file_tools/file_utils.py

import os
import re
import difflib
from TaruAgent import tool

# --- Configuration ---
# This defines the safe directory for file operations.
# Using an environment variable is more flexible than hardcoding.
SANDBOX_ROOT = os.path.abspath(os.environ.get("TARU_SANDBOX_ROOT", "/apps/data"))

# Create the directory if it doesn't exist to avoid errors on first run
os.makedirs(SANDBOX_ROOT, exist_ok=True)


# --- Helper Functions ---

def safe_join(root, *paths):
    """
    Safely joins a root directory with path segments, preventing directory traversal.
    """
    p = os.path.abspath(os.path.join(root, *paths))
    if not p.startswith(root):
        raise ValueError("Access denied: Path is outside the designated sandbox directory.")
    return p

def privacy_filter(text):
    """
    Scrubs common secret patterns from text before returning it.
    """
    patterns = [
        (r'(?i)(api_key|apikey|secret|token|key|password|pw|pass|access_token)\s*[:=]\s*([\'\"]?)[^\s\'\"]+\2', r'\1=<\1>'),
        (r'(?i)\b(username|user|login)[\'\"]?\s*:\s*[\'\"][^\'\"]+[\'\"]', r'\1: "<username>"'),
        (r'([\'\"])([^\'\"]{4,128})([\'\"])\s*(#\s*password)', r'"<password>" \4'),
    ]
    for pat, repl in patterns:
        text = re.sub(pat, repl, text)
    return text

def find_files_fuzzy(root_dir, filename, max_results=5):
    """
    Performs a fuzzy search to find files similar to the requested filename.
    """
    matches = []
    filename_lower = filename.lower()
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            f_lower = f.lower()
            if filename_lower in f_lower:
                rel_path = os.path.relpath(os.path.join(dirpath, f), root_dir)
                matches.append("/" + rel_path.replace("\\", "/"))
            else:
                ratio = difflib.SequenceMatcher(None, filename_lower, f_lower).ratio()
                if ratio > 0.6:
                    rel_path = os.path.relpath(os.path.join(dirpath, f), root_dir)
                    matches.append("/" + rel_path.replace("\\", "/"))
            if len(matches) >= max_results:
                return matches
    return matches


# --- The Tool ---
@tool(
    name="file_list",
    description="List files and directories under a sandbox subdirectory, returning full paths for files.",
    tags={"sandbox", "file", "list"},
)
def sandbox_list(subdir: str = None):
    try:
        # Normalize subdir, strip leading slash for internal path join
        if not subdir or subdir.strip() == "":
            subdir = "."
        elif subdir.startswith("/"):
            subdir = subdir[1:]

        full_path = safe_join(SANDBOX_ROOT, subdir)
        entries = os.listdir(full_path)
        results = []
        for entry in entries:
            entry_path = os.path.join(full_path, entry)
            rel_path = os.path.relpath(entry_path, SANDBOX_ROOT)
            results.append({
                "name": entry,
                "full_path": "/" + rel_path.replace("\\", "/"),  # Prepend slash and normalize separator
                "is_dir": os.path.isdir(entry_path)
            })
        return {"result": results}
    except Exception as e:
        return {"error": str(e)}
    
@tool(
    name="file_read",
    description="Read one or more files from the sandbox  directory. Can suggest files if an exact match is not found.",
)
def file_read(path: str = None, paths: list[str] = None):
    """
    Reads one or more files from a sandboxed directory, with privacy filtering.

    Args:
        path: The path to a single file to read.
        paths: A list of paths to multiple files to read.
    
    Returns:
        A dictionary containing the results, errors, and suggestions.
    """
    results = {}
    errors = {}
    suggestions = []

    paths_to_read = []
    if path:
        paths_to_read = [path.lstrip("/")]
    elif paths:
        paths_to_read = [p.lstrip("/") for p in paths]
    else:
        return {"error": "No path or paths specified"}

    if len(paths_to_read) == 1:
        p = paths_to_read[0]
        try:
            full_path = safe_join(SANDBOX_ROOT, p)
            if not os.path.isfile(full_path):
                matches = find_files_fuzzy(SANDBOX_ROOT, os.path.basename(p))
                if matches:
                    return {
                        "error": f"File '{'/' + p}' not found.",
                        "suggestions": matches,
                        "message": "Please confirm exact file path to read."
                    }
                else:
                    return {"error": f"File '{'/' + p}' not found and no similar files found."}
            else:
                with open(full_path, "r") as f:
                    content = f.read()
                results["/" + p.replace("\\", "/")] = privacy_filter(content)
        except Exception as e:
            errors["/" + p] = str(e)
    else:
        for p in paths_to_read:
            try:
                full_path = safe_join(SANDBOX_ROOT, p)
                if not os.path.isfile(full_path):
                    errors["/" + p] = "File not found."
                    continue
                with open(full_path, "r") as f:
                    content = f.read()
                results["/" + p.replace("\\", "/")] = privacy_filter(content)
            except Exception as e:
                errors["/" + p] = str(e)

    response = {}
    if results:
        response["result"] = results
    if errors:
        response["errors"] = errors
    if suggestions:
        response["suggestions"] = suggestions
        response["message"] = "Please confirm exact file path to read."

    if not response:
        response = {"error": "No files found or read."}

    return response
