import os
import re
import yaml
import difflib
from fastmcp import FastMCP


SANDBOX_ROOT = "/testing/data"
def safe_join(root, *paths):
    p = os.path.abspath(os.path.join(root, *paths))
    if not p.startswith(root):
        raise Exception("Access denied!")
    return p

def privacy_filter(text):
    patterns = [
        (r'(?i)(api_key|apikey|secret|token|key|password|pw|pass|access_token)\s*[:=]\s*([\'"]?)[^\s\'"]+\2', r'\1=<\1>'),
        (r'(?i)\b(username|user|login)[\'"]?\s*:\s*[\'"][^\'"]+[\'"]', r'\1: "<username>"'),
        (r'([\'"])([^\'"]{4,128})([\'"])\s*(#\s*password)', r'"<password>" \4'),
    ]
    for pat, repl in patterns:
        text = re.sub(pat, repl, text)
    return text

def find_files_fuzzy(root_dir, filename, max_results=5):
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

mcp = FastMCP(
    "FileManager",
    host="0.0.0.0",
    port=8525,
)

@mcp.tool(
    name="file_read",
    description="Read one or more files from the sandbox with privacy filtering and user confirmation for fuzzy matches.",
    tags={"sandbox", "file", "read"},
)
def file_read(path: str = None, paths: list[str] = None):
    results = {}
    errors = {}
    suggestions = []

    paths_to_read = []
    if path:
        # Strip leading slash from single path input
        paths_to_read = [path.lstrip("/")]
    elif paths:
        # Strip leading slash from all paths
        paths_to_read = [p.lstrip("/") for p in paths]
    else:
        return {"error": "No path or paths specified"}

    if len(paths_to_read) == 1:
        p = paths_to_read[0]
        try:
            full_path = safe_join(SANDBOX_ROOT, p)
            if not os.path.isfile(full_path):
                # File not found, perform fuzzy search and return suggestions only
                matches = find_files_fuzzy(SANDBOX_ROOT, os.path.basename(p))
                if matches:
                    suggestions = matches
                    return {
                        "error": f"File '{'/' + p}' not found.",
                        "suggestions": suggestions,
                        "message": "Please confirm exact file path to read."
                    }
                else:
                    return {
                        "error": f"File '{'/' + p}' not found and no similar files found."
                    }
            else:
                # Exact file found, read it
                with open(full_path, "r") as f:
                    content = f.read()
                results["/" + p.replace("\\", "/")] = privacy_filter(content)
        except Exception as e:
            errors["/" + p] = str(e)
    else:
        # Multiple files: read only exact matches, report errors for missing
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


@mcp.tool(
    name="file_write",
    description="Write content to a file in the sandbox, with optional overwrite control.",
    tags={"sandbox", "file", "write"},
)
def file_write(path: str, content: str, overwrite: bool = True):
    try:
        # Strip leading slash from input path
        normalized_path = path.lstrip("/")
        full_path = safe_join(SANDBOX_ROOT, normalized_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if not overwrite and os.path.exists(full_path):
            return {"error": "File exists and overwrite not allowed"}
        with open(full_path, "w") as f:
            f.write(content)
        return {"result": "ok", "path": "/" + normalized_path.replace("\\", "/")}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool(
    name="file_list",
    description="List files and directories under a sandbox subdirectory, returning full paths for files.",
    tags={"sandbox", "file", "list"},
)
def file_list(subdir: str = None):
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

@mcp.tool(
    name="file_mkdir",
    description="Create a directory inside the sandbox.",
    tags={"sandbox", "file", "mkdir"},
)
def file_mkdir(path: str):
    try:
        # Strip leading slash from input path
        normalized_path = path.lstrip("/")
        full_path = safe_join(SANDBOX_ROOT, normalized_path)
        os.makedirs(full_path, exist_ok=True)
        return {"result": "created", "path": "/" + normalized_path.replace("\\", "/")}
    except Exception as e:
        return {"error": str(e)}
    

if __name__ == "__main__":
    mcp.run(host="0.0.0.0", port=8525, transport="sse")

