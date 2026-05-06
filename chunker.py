from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass

EXTENSION_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go",
    ".rs": "rust", ".java": "java", ".c": "c", ".cpp": "cpp",
    ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala", ".sh": "bash",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json",
    ".md": "markdown", ".html": "html", ".css": "css",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "vendor", "target", "bin", "obj",
}

MAX_CHUNK_CHARS = 1500
OVERLAP_CHARS   = 200


@dataclass
class Chunk:
    file_path: str
    language:  str
    start_line: int
    end_line:   int
    symbol:     str | None  # function/class name if extracted
    content:    str


def _sliding_window(lines: list[str], file_path: str, lang: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[str] = []
    start = 1
    char_count = 0

    for i, line in enumerate(lines, start=1):
        current.append(line)
        char_count += len(line)
        if char_count >= MAX_CHUNK_CHARS:
            chunks.append(Chunk(
                file_path=file_path, language=lang,
                start_line=start, end_line=i,
                symbol=None, content="".join(current),
            ))
            # keep overlap
            overlap: list[str] = []
            overlap_chars = 0
            for l in reversed(current):
                if overlap_chars + len(l) > OVERLAP_CHARS:
                    break
                overlap.insert(0, l)
                overlap_chars += len(l)
            current = overlap
            start = i - len(overlap) + 1
            char_count = overlap_chars

    if current:
        chunks.append(Chunk(
            file_path=file_path, language=lang,
            start_line=start, end_line=start + len(current) - 1,
            symbol=None, content="".join(current),
        ))
    return chunks


def _try_treesitter(source: str, lang: str, file_path: str) -> list[Chunk] | None:
    try:
        import tree_sitter_python as tspython
        import tree_sitter_javascript as tsjs
        import tree_sitter_typescript as tsts
        import tree_sitter_go as tsgo
        import tree_sitter_rust as tsrust
        import tree_sitter_java as tsjava
        from tree_sitter import Language, Parser

        lang_map = {
            "python":     Language(tspython.language()),
            "javascript": Language(tsjs.language()),
            "typescript": Language(tsts.language_typescript()),
            "go":         Language(tsgo.language()),
            "rust":       Language(tsrust.language()),
            "java":       Language(tsjava.language()),
        }
        if lang not in lang_map:
            return None

        parser = Parser(lang_map[lang])
        tree = parser.parse(source.encode())

        SYMBOL_TYPES = {
            "function_definition", "function_declaration", "method_definition",
            "class_definition", "class_declaration", "impl_item",
            "function_item", "method_declaration",
        }

        chunks: list[Chunk] = []
        lines = source.splitlines(keepends=True)

        def walk(node):
            if node.type in SYMBOL_TYPES:
                start = node.start_point[0]
                end   = node.end_point[0] + 1
                name_node = node.child_by_field_name("name")
                symbol = name_node.text.decode() if name_node else None
                content = "".join(lines[start:end])
                if len(content) > MAX_CHUNK_CHARS:
                    # too large — fall through to children
                    for child in node.children:
                        walk(child)
                else:
                    chunks.append(Chunk(
                        file_path=file_path, language=lang,
                        start_line=start + 1, end_line=end,
                        symbol=symbol, content=content,
                    ))
                return
            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return chunks if chunks else None

    except Exception:
        return None


def chunk_file(path: Path, repo_root: Path) -> list[Chunk]:
    lang = EXTENSION_LANG.get(path.suffix.lower())
    if not lang:
        return []

    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    if not source.strip():
        return []

    rel = str(path.relative_to(repo_root))
    lines = source.splitlines(keepends=True)

    chunks = _try_treesitter(source, lang, rel)
    if not chunks:
        chunks = _sliding_window(lines, rel, lang)

    return chunks


def iter_repo_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for p in repo_root.rglob("*"):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
            if p.suffix.lower() in EXTENSION_LANG:
                files.append(p)
    return sorted(files)
