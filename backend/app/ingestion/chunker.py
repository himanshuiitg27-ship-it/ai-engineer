"""AST-aware code chunking with Tree-sitter.

Produces function-, method- and class-level chunks with exact 1-based, inclusive line ranges, so
that answers can cite `path:start-end` precisely. Top-level code that is not part of any definition
becomes `module` chunks; prose files (README/docs) become `doc` chunks.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from tree_sitter import Node

from app.ingestion.languages import DOC_LANGUAGE, LangSpec, get_parser, is_doc_path, spec_for_path

log = logging.getLogger(__name__)

MAX_PRECEDING_COMMENT_LINES = 30
DOC_WINDOW_LINES = 40
WINDOW_OVERLAP = 4


@dataclass
class CodeChunk:
    path: str
    language: str
    kind: str  # function | method | class | module | doc
    name: str
    qualified_name: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    code: str
    symbol_type: str = ""  # raw tree-sitter node type, e.g. "impl_item"
    part: int = 0  # 1-based window index when a long definition was split, else 0
    parts: int = 0
    truncated: bool = False  # class chunk that shows only its header
    content_hash: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "CodeChunk":
        return cls(**d)

    def embedding_text(self) -> str:
        """Text fed to the embedding model: a small metadata header + the code."""
        label = self.qualified_name if self.qualified_name else self.name
        return f"{self.language} {self.kind} {label} in {self.path}\n{self.code}"


def _normalize_for_hash(code: str) -> str:
    return re.sub(r"\s+", " ", code).strip()


def _hash(code: str) -> str:
    return hashlib.sha1(_normalize_for_hash(code).encode("utf-8", "replace")).hexdigest()


# --------------------------------------------------------------------------------------
# Node helpers
# --------------------------------------------------------------------------------------

_IDENT_TYPES = {
    "identifier", "type_identifier", "simple_identifier", "field_identifier", "constant",
    "property_identifier", "name", "qualified_identifier", "scope_resolution",
}


def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _declarator_name(node: Node, src: bytes) -> Optional[str]:
    """C/C++ style: function_definition -> declarator -> ... -> identifier."""
    cur = node.child_by_field_name("declarator")
    depth = 0
    while cur is not None and depth < 12:
        nxt = cur.child_by_field_name("declarator")
        if nxt is None:
            return _text(cur, src)
        cur = nxt
        depth += 1
    return None


def _node_name(node: Node, src: bytes) -> str:
    n = node.child_by_field_name("name")
    if n is not None:
        return _text(n, src).strip()
    if node.type == "impl_item":
        t = node.child_by_field_name("type")
        if t is not None:
            tr = node.child_by_field_name("trait")
            label = _text(t, src)
            return f"{_text(tr, src)} for {label}" if tr is not None else label
    d = _declarator_name(node, src)
    if d:
        return d.strip()
    for child in node.named_children:
        if child.type in _IDENT_TYPES:
            return _text(child, src).strip()
    # typedef struct { ... } Name;
    parent = node.parent
    if parent is not None and parent.type == "type_definition":
        decl = parent.child_by_field_name("declarator")
        if decl is not None:
            return _text(decl, src).strip()
    return "<anonymous>"


def _go_type_name(node: Node, src: bytes) -> Optional[str]:
    """Only struct/interface type declarations count as classes in Go."""
    for spec in node.named_children:
        if spec.type == "type_spec":
            t = spec.child_by_field_name("type")
            if t is not None and t.type in ("struct_type", "interface_type"):
                n = spec.child_by_field_name("name")
                return _text(n, src) if n is not None else "<anonymous>"
    return None


_FUNC_VALUE_TYPES = {"arrow_function", "function_expression", "function", "generator_function"}


def _var_function_name(node: Node, src: bytes) -> Optional[str]:
    for decl in node.named_children:
        if decl.type == "variable_declarator":
            value = decl.child_by_field_name("value")
            if value is not None and value.type in _FUNC_VALUE_TYPES:
                n = decl.child_by_field_name("name")
                return _text(n, src) if n is not None else "<anonymous>"
    return None


def _classify(node: Node, spec: LangSpec, src: bytes, in_class: bool) -> Optional[Tuple[str, str]]:
    """Return (kind, name) if `node` is a definition we chunk, else None."""
    t = node.type
    if t in spec.class_types:
        if spec.name == "go":
            name = _go_type_name(node, src)
            return ("class", name) if name else None
        if spec.name in ("c", "cpp") and node.child_by_field_name("body") is None:
            return None  # forward declaration / bare type reference
        return "class", _node_name(node, src)
    if t in spec.method_types:
        return "method", _node_name(node, src)
    if t in spec.function_types:
        if spec.name in ("c", "cpp") and node.child_by_field_name("body") is None:
            return None
        return ("method" if in_class else "function"), _node_name(node, src)
    if t in spec.var_function_types:
        name = _var_function_name(node, src)
        if name:
            return ("method" if in_class else "function"), name
    return None


def _expand_range(node: Node, spec: LangSpec) -> Tuple[Node, int, int]:
    """Expand a definition to include decorators/exports wrappers and directly-attached comments."""
    outer = node
    while outer.parent is not None and outer.parent.type in spec.wrapper_types:
        outer = outer.parent
    start_row = outer.start_point.row
    end_row = outer.end_point.row
    if outer.end_point.column == 0 and end_row > start_row:
        end_row -= 1  # grammar included the trailing newline

    prev = outer.prev_sibling
    steps = 0
    while prev is not None and "comment" in prev.type and prev.end_point.row >= start_row - 1 and steps < 8:
        if start_row - prev.start_point.row > MAX_PRECEDING_COMMENT_LINES:
            break
        start_row = prev.start_point.row
        prev = prev.prev_sibling
        steps += 1
    return outer, start_row, end_row


# --------------------------------------------------------------------------------------
# Chunk assembly
# --------------------------------------------------------------------------------------

def _split_lines(text: str) -> List[str]:
    """Split into lines WITHOUT the phantom empty element after a trailing newline (keeps line numbers exact)."""
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()
    return lines


def _slice(lines: List[str], start_line: int, end_line: int) -> str:
    return "\n".join(lines[start_line - 1:end_line])


def _windows(start: int, end: int, size: int, overlap: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    cur = start
    while cur <= end:
        stop = min(cur + size - 1, end)
        out.append((cur, stop))
        if stop >= end:
            break
        cur = stop - overlap + 1
    return out


def _emit_definition(
    out: List[CodeChunk], path: str, spec: LangSpec, lines: List[str], kind: str, name: str,
    qualified: str, start: int, end: int, symbol_type: str, max_lines: int,
) -> None:
    span = end - start + 1
    if span <= max_lines:
        code = _slice(lines, start, end)
        out.append(CodeChunk(path, spec.name, kind, name, qualified, start, end, code, symbol_type,
                             content_hash=_hash(code)))
        return
    if kind == "class":
        # Header only: signature, attributes, leading docs. Members are chunked separately.
        stop = start + max_lines - 1
        code = _slice(lines, start, stop)
        out.append(CodeChunk(path, spec.name, kind, name, qualified, start, stop, code, symbol_type,
                             truncated=True, content_hash=_hash(code)))
        return
    wins = _windows(start, end, max_lines, WINDOW_OVERLAP)
    for i, (ws, we) in enumerate(wins, start=1):
        code = _slice(lines, ws, we)
        out.append(CodeChunk(path, spec.name, kind, name, qualified, ws, we, code, symbol_type,
                             part=i, parts=len(wins), content_hash=_hash(code)))


def _module_chunks(
    path: str, spec: LangSpec, lines: List[str], covered: List[bool], max_lines: int
) -> List[CodeChunk]:
    """Top-level code outside any definition (imports, constants, scripts) -> `module` chunks."""
    out: List[CodeChunk] = []
    n = len(lines)
    i = 0
    while i < n:
        if covered[i]:
            i += 1
            continue
        j = i
        blank_run = 0
        last_content = i
        while j < n and not covered[j]:
            if lines[j].strip():
                blank_run = 0
                last_content = j
            else:
                blank_run += 1
                if blank_run > 2:
                    break
            j += 1
        block_start, block_end = i + 1, last_content + 1  # 1-based inclusive
        text = _slice(lines, block_start, block_end)
        if len(re.sub(r"\s+", "", text)) >= 40:
            for ws, we in _windows(block_start, block_end, max_lines, WINDOW_OVERLAP):
                code = _slice(lines, ws, we)
                out.append(CodeChunk(path, spec.name, "module", "<module>", f"{path}", ws, we, code, "module",
                                     content_hash=_hash(code)))
        i = max(j, i + 1)
    return out


def chunk_source(path: str, source: bytes, max_lines: int = 60) -> List[CodeChunk]:
    """Chunk one file. Returns [] for unsupported or empty files."""
    if is_doc_path(path):
        return _chunk_doc(path, source)
    spec = spec_for_path(path)
    if spec is None or not source.strip():
        return []

    parser = get_parser(spec.name)
    tree = parser.parse(source)
    text = source.decode("utf-8", "replace")
    lines = _split_lines(text)
    covered = [False] * len(lines)
    chunks: List[CodeChunk] = []

    # Iterative DFS (deeply nested ASTs would overflow Python's recursion limit).
    stack: List[Tuple[Node, Tuple[str, ...]]] = [(tree.root_node, ())]
    while stack:
        node, class_stack = stack.pop()
        res = _classify(node, spec, source, in_class=bool(class_stack))
        if res is None:
            for child in reversed(node.children):
                stack.append((child, class_stack))
            continue
        kind, name = res
        outer, start_row, end_row = _expand_range(node, spec)
        start, end = start_row + 1, min(end_row + 1, len(lines))
        qualified = ".".join(class_stack + (name,))
        _emit_definition(chunks, path, spec, lines, kind, name, qualified, start, end, node.type, max_lines)
        for r in range(start - 1, min(end, len(lines))):
            covered[r] = True
        if kind == "class":
            for child in reversed(node.children):
                stack.append((child, class_stack + (name,)))
        # functions/methods: do not descend (nested functions stay inside their parent chunk)

    chunks.extend(_module_chunks(path, spec, lines, covered, max_lines))
    chunks.sort(key=lambda c: (c.start_line, c.end_line, c.part))
    return chunks


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)")


def _chunk_doc(path: str, source: bytes) -> List[CodeChunk]:
    text = source.decode("utf-8", "replace")
    if not text.strip():
        return []
    lines = _split_lines(text)
    out: List[CodeChunk] = []
    heading = ""
    heading_at: Dict[int, str] = {}
    for idx, ln in enumerate(lines, start=1):
        m = _HEADING.match(ln)
        if m:
            heading = m.group(1).strip()
        heading_at[idx] = heading
    for ws, we in _windows(1, len(lines), DOC_WINDOW_LINES, WINDOW_OVERLAP):
        code = _slice(lines, ws, we)
        if len(code.strip()) < 20:
            continue
        name = heading_at.get(ws) or path.rsplit("/", 1)[-1]
        out.append(CodeChunk(path, DOC_LANGUAGE, "doc", name, name, ws, we, code, "doc", content_hash=_hash(code)))
    return out


def chunk_files(files: Iterable[Tuple[str, bytes]], max_lines: int = 60) -> Iterable[CodeChunk]:
    """Chunk many files, isolating per-file failures."""
    for path, data in files:
        try:
            yield from chunk_source(path, data, max_lines=max_lines)
        except Exception:  # a single odd file must never abort indexing
            log.exception("Failed to chunk %s", path)
