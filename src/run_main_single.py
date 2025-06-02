"""
Framework for LLM-Specific Defect Localization using Multi-Agent Collaboration and Call Relationship Subgraph Aggregation.

Agent 1: Code Knowledge Graph Constructor - Builds function-level knowledge graph with static analysis and embeddings.
Agent 2: Defect Analyzer - Interprets defect description and performs full matching recall (static + embedding supplement).
Agent 3: Validator - Call relationship subgraph aggregation, hierarchical evaluation, and two-stage localization.
"""

import os
import json
import networkx as nx
import argparse
import re
import sys
import builtins
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
import openai
import logging
from tqdm import tqdm
from tree_sitter_languages import get_parser, get_language
from utils import setup_logging, get_llm_res, get_embedding,PROJECT_PATH
from datetime import datetime
import difflib

# get_embedding(text: str, base_url: Optional[str] = None, api_key: Optional[str] = None, model: str = "bge-m3:567m-fp16", output_dir: str = "./logs") -> List[float]

if TYPE_CHECKING:
    from tree_sitter import Tree, Node, Query  # type: ignore

# --- Constant definitions ---
# Directories and files to ignore during parsing
IGNORE_DIRS = {".git", "__pycache__", "build", "dist", ".venv", "venv", "env", "node_modules", "target", "docs", "doc", "test", "example", "examples", "samples", "migrations", "fixtures", "data", "static", "templates"}

IGNORE_FILES = {".DS_Store", " Pipfile.lock", "poetry.lock", "yarn.lock", "package-lock.json"}

# Predefined LLM roles for static analysis and LLM interaction
LLM_ROLES = ["API_Caller", "Prompt_Builder", "Response_Parser", "Context_Retriever", "History_Manager", "Embedding_Generator", "Knowledge_Updater", "Input_Validator", "Output_Processor", "Error_Handler", "Data_Sanitizer"]
# Known LLM-related configuration keys for static analysis
KNOWN_LLM_CONFIGS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HUGGINGFACE_TOKEN", "GOOGLE_API_KEY", "AZURE_OPENAI_API_KEY", "API_KEY", "API_TOKEN", "SECRET_KEY", "MODEL_NAME", "CHAT_MODEL", "COMPLETION_MODEL", "EMBEDDING_MODEL", "LLM_MODEL", "ENGINE", "DEPLOYMENT_ID", "API_BASE", "OPENAI_API_BASE", "BASE_URL", "ENDPOINT", "TEMPERATURE", "MAX_TOKENS", "TOP_P", "FREQUENCY_PENALTY", "PRESENCE_PENALTY", "STOP_SEQUENCES", "STOP", "MAX_OUTPUT_TOKENS", "MAX_CONTEXT_TOKENS", "MAX_PROMPT_LENGTH", "PROMPT_TRUNCATION_STRATEGY", "MAX_HISTORY_TURNS", "HISTORY_SUMMARY_METHOD", "CHAT_HISTORY", "MEMORY", "SYSTEM_PROMPT", "SYSTEM_MESSAGE_TEMPLATE", "USER_MESSAGE_TEMPLATE", "AI_MESSAGE_TEMPLATE", "PROMPT_TEMPLATE", "VECTOR_DB_URL", "INDEX_NAME", "TOP_K_RESULTS", "SIMILARITY_THRESHOLD", "TOP_K", "CHUNK_SIZE", "CHUNK_OVERLAP", "REQUEST_TIMEOUT", "MAX_RETRIES", "TIMEOUT", "RETRY_DELAY", "TOOL_CHOICE", "TOOLS", "FUNCTION_CALLING", "STREAMING", "STREAM", "LOG_LEVEL", "VERBOSE", "API_VERSION", "ORGANIZATION", "PROJECT_ID", "REGION", "TOKEN", "LLM_CONFIG", "MESSAGES", "PROMPT", "RESPONSE_FORMAT"]


# --- Data structure definitions ---
@dataclass
class FunctionNode:
    id: str
    name: str
    file_path: str
    start_line: int
    end_line: int
    signature: str
    code_snippet: str
    docstring: str = ""
    module_id: str = ""
    containing_class_id: Optional[str] = None
    static_analysis_hints: Dict[str, List[str]] = field(default_factory=lambda: {"potential_roles": [], "potential_configs": []})
    imports: List[str] = field(default_factory=list)
    llm_roles: List[str] = field(default_factory=list)
    uses_configs: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, FunctionNode) and self.id == other.id

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class CodeKnowledgeGraph:
    nodes: Dict[str, FunctionNode] = field(default_factory=dict)
    call_graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": {nid: node.to_dict() for nid, node in self.nodes.items()}, "call_graph": nx.node_link_data(self.call_graph)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeKnowledgeGraph":
        kg = cls()
        kg.nodes = {nid: FunctionNode(**node_data) for nid, node_data in data["nodes"].items()}
        kg.call_graph = nx.node_link_graph(data["call_graph"])
        return kg


@dataclass
class DefectAnalysisSummary:
    """Stores the LLM's initial analysis of the defect description."""

    relevant_llm_roles: List[str] = field(default_factory=list)
    relevant_configs: List[str] = field(default_factory=list)
    potential_locations: List[str] = field(default_factory=list)
    code_keywords: List[str] = field(default_factory=list)
    raw_llm_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class CallRelationSubgraphAnalysisUnit:
    """Represents a call relationship subgraph for hierarchical analysis."""

    subgraph_id: str
    connected_functions: List[FunctionNode]
    call_relationships: List[Tuple[str, str]]
    involved_files: Set[str]
    file_summaries: Dict[str, str]
    llm_features_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"subgraph_id": self.subgraph_id, "connected_functions": [f.to_dict() for f in self.connected_functions], "call_relationships": self.call_relationships, "involved_files": list(self.involved_files), "file_summaries": self.file_summaries, "llm_features_summary": self.llm_features_summary}


@dataclass
class SearchResultEntry:
    """Represents a candidate function with its context for LLM validation."""

    target_function_node: FunctionNode
    target_code_snippet: str
    contextual_snippets: Dict[str, Any]
    initial_match_score: Optional[float] = None
    initial_match_reason: Optional[str] = None
    embedding_similarity_score: Optional[float] = None
    file_importance_score: Optional[float] = None
    llm_validation_score: Optional[float] = None
    llm_validation_reason: Optional[str] = None
    subgraph_id: Optional[str] = None
    intra_file_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"target_function_node_id": self.target_function_node.id, "target_code_snippet": self.target_code_snippet, "contextual_snippets": self.contextual_snippets, "initial_match_score": self.initial_match_score, "initial_match_reason": self.initial_match_reason, "embedding_similarity_score": self.embedding_similarity_score, "file_importance_score": self.file_importance_score, "llm_validation_score": self.llm_validation_score, "llm_validation_reason": self.llm_validation_reason, "subgraph_id": self.subgraph_id, "intra_file_rank": self.intra_file_rank, "target_function_summary": {"name": self.target_function_node.name, "file_path": self.target_function_node.file_path, "start_line": self.target_function_node.start_line, "static_hints": self.target_function_node.static_analysis_hints}}


# --- Helper functions ---
def safe_read_file(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None


def get_ts_node_text(node: "Node", encoding: str = "utf-8") -> str:
    """Extracts the full text of a node, properly handling composite nodes like dotted_name."""
    if node is None:
        return ""

    if node.type == "dotted_name":
        parts = []
        for child in node.children:
            parts.append(get_ts_node_text(child, encoding))
        return "".join(parts)

    return node.text.decode(encoding)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0

    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1_np, vec2_np) / (norm1 * norm2))


def limit_user_message(user_message: str, system_prompt: str, max_tokens: int = 32768) -> str:
    # reserve tokens for system prompt, response, tool calls, and safety margin
    system_tokens_estimate = len(system_prompt) // 3  # approximate 1 token = 3-4 chars
    reserved_tokens = 2000  # token for response and tool calls
    safety_margin = 500  # token for safet

    available_tokens = max_tokens - system_tokens_estimate - reserved_tokens - safety_margin
    max_user_chars = available_tokens * 3  # change to chars

    if len(user_message) > max_user_chars:
        user_message = user_message[:max_user_chars] + "...[content truncated]"

    return user_message


def get_std_lib_symbols() -> Set[str]:
    symbols = set(dir(builtins))
    # A more comprehensive list of common standard library modules
    common_stdlib_modules = ["os", "sys", "re", "json", "math", "datetime", "collections", "itertools", "functools", "typing", "pathlib", "io", "subprocess", "argparse", "logging", "random", "string", "time", "pickle", "inspect", "warnings", "copy", "dataclasses", "enum", "threading", "multiprocessing", "asyncio", "contextlib", "glob", "shutil", "tempfile", "uuid", "base64", "hashlib", "hmac", "socket", "select", "struct", "decimal", "fractions", "csv", "configparser", "urllib.parse", "urllib.request", "http.client", "http.server", "xml.etree.ElementTree", "zipfile", "tarfile", "gzip"]
    for mod_name in common_stdlib_modules:
        try:
            module = __import__(mod_name)
            symbols.update(dir(module))
        except ImportError:
            logging.info(f"Standard library module {mod_name} not found or not importable.")
        except Exception as e:
            logging.info(f"Could not import or list dir for stdlib module {mod_name}: {e}")

    # Add attributes of common built-in types
    for t in [list, dict, set, str, tuple, int, float, bool, bytes, bytearray, type(None), type]:
        symbols.update(dir(t))
    return symbols


STD_LIB_SYMBOLS = get_std_lib_symbols()


# --- Agent 1: Code Knowledge Graph Constructor ---
class CodeKGConstructor:
    """Agent 1: Builds a function-level Code Knowledge Graph with static analysis and embeddings."""

    def __init__(self, repo_path: str, args: argparse.Namespace):
        self.repo_path = Path(repo_path).resolve()
        self.args = args
        self.kg = CodeKnowledgeGraph()
        self.parsed_files_ts: Dict[str, Optional["Tree"]] = {}
        self.file_lines_cache: Dict[str, List[str]] = {}
        self.import_map: Dict[str, Dict[str, str]] = {}

        logging.info(f"[Agent 1] Initializing CodeKGConstructor for repository: {self.repo_path}")
        self.ts_parser = get_parser("python")
        self.python_language = get_language("python")
        self._initialize_ts_queries()

    def _initialize_ts_queries(self):
        """Initializes Tree-sitter queries for extracting code elements."""
        # Query for imports (comprehensive)
        self.ts_query_imports = self.python_language.query(
            """
        (import_statement 
            name: (dotted_name) @import_name) @import_stmt
        (import_statement 
            name: (aliased_import 
                name: (dotted_name) @import_name 
                alias: (identifier) @import_alias)) @import_stmt
        (import_from_statement
            module_name: [(dotted_name) @import_module (relative_import) @relative_import]
            name: (dotted_name) @import_name) @import_from_stmt
        (import_from_statement
            module_name: [(dotted_name) @import_module (relative_import) @relative_import]
            name: (aliased_import 
                name: (dotted_name) @import_name 
                alias: (identifier) @import_alias)) @import_from_stmt
        (import_from_statement
            module_name: [(dotted_name) @import_module (relative_import) @relative_import]
            (wildcard_import) @wildcard) @import_from_stmt
        """
        )
        # Query for function and class definitions
        self.ts_query_func_def = self.python_language.query("(function_definition) @func")
        self.ts_query_class_def = self.python_language.query("(class_definition) @class")
        # Query for function calls
        self.ts_query_call = self.python_language.query(
            """
            (call
                function: [
                    (identifier) @func_name_direct
                    (attribute object: (_) @obj attribute: (identifier) @attr_name)
                ]
            ) @call_node
        """
        )
        # Query for assignments (useful for config detection)
        self.ts_query_assignment = self.python_language.query("(assignment left: (identifier) @var_name right: (_) @value_node) @assign_stmt")
        # Query for try-except blocks (Error_Handler role)
        self.ts_query_try_stmt = self.python_language.query("(try_statement) @try_stmt")
        # Query for string literals and f-strings (Prompt_Builder role)
        self.ts_query_string_literal = self.python_language.query("(string) @string")
        self.ts_query_fstring_content = self.python_language.query("(interpolation expression: (_) @expression) @interpolation")

    def _find_python_files(self) -> List[Path]:
        """Scans the repository for Python files, respecting IGNORE_DIRS and IGNORE_FILES."""
        py_files = []
        logging.info(f"[Agent 1] Scanning for Python files in {self.repo_path}...")
        for item in self.repo_path.rglob("*.py"):
            if item.is_file():
                try:
                    relative_parts = item.relative_to(self.repo_path).parts
                    if any(part in IGNORE_DIRS for part in relative_parts) or item.name in IGNORE_FILES:
                        logging.info(f"Ignoring file due to rule: {item}")
                        continue
                    py_files.append(item)
                except ValueError:
                    logging.warning(f"Could not get relative path for {item}, skipping.")
        logging.info(f"[Agent 1] Found {len(py_files)} Python files for analysis.")
        return py_files

    def _parse_file_ts(self, file_path: Path) -> Optional["Tree"]:
        """Parses a single Python file using Tree-sitter and caches the result."""
        rel_path_str = str(file_path.relative_to(self.repo_path))
        if rel_path_str in self.parsed_files_ts:
            return self.parsed_files_ts[rel_path_str]

        content = safe_read_file(str(file_path))
        if content is None:
            self.file_lines_cache[rel_path_str] = []
            self.parsed_files_ts[rel_path_str] = None
            return None

        self.file_lines_cache[rel_path_str] = content.splitlines()
        try:
            tree = self.ts_parser.parse(bytes(content, "utf8"))
            self.parsed_files_ts[rel_path_str] = tree
            self._extract_imports_ts(tree.root_node, rel_path_str)
            return tree
        except Exception as e:
            logging.error(f"Failed to parse {rel_path_str} with Tree-sitter: {e}", exc_info=True)
            self.parsed_files_ts[rel_path_str] = None
            return None

    def _extract_imports_ts(self, root_node: "Node", rel_file_path: str):
        """Extracts and resolves import statements from a file's AST root."""
        imports: Dict[str, str] = {}
        if not root_node:
            return

        captures = self.ts_query_imports.captures(root_node)

        # Group captures by parent import statement
        statements: Dict[int, Dict[str, List["Node"]]] = {}
        for node, tag in captures:
            parent = node
            while parent and parent.type not in ["import_statement", "import_from_statement"]:
                parent = parent.parent

            if not parent:
                continue

            if parent.id not in statements:
                statements[parent.id] = {"nodes": [], "tags": []}
            statements[parent.id]["nodes"].append(node)
            statements[parent.id]["tags"].append(tag)

        # Process each import statement
        for stmt_id, stmt_info in statements.items():
            nodes = stmt_info["nodes"]
            tags = stmt_info["tags"]

            # Find statement type
            parent_node = None
            for node in nodes:
                parent = node.parent
                while parent and parent.type not in ["import_statement", "import_from_statement"]:
                    parent = parent.parent
                if parent:
                    parent_node = parent
                    break

            if not parent_node:
                continue

            stmt_type = parent_node.type

            if stmt_type == "import_statement":
                # Process import statement (import x, import x as y)
                module_names = [n for n, t in zip(nodes, tags) if t == "import_name"]
                aliases = [n for n, t in zip(nodes, tags) if t == "import_alias"]

                for i, module_node in enumerate(module_names):
                    module_path = get_ts_node_text(module_node)
                    # Use alias if present
                    alias = None
                    if i < len(aliases):
                        alias = get_ts_node_text(aliases[i])

                    if not alias:
                        # For "import x.y.z", add both "z" -> "x.y.z" and "x" -> "x"
                        parts = module_path.split(".")
                        imports[parts[-1]] = module_path
                        imports[parts[0]] = parts[0]
                    else:
                        imports[alias] = module_path

            elif stmt_type == "import_from_statement":
                # Process from statement (from x import y, from x import y as z)
                module_nodes = [n for n, t in zip(nodes, tags) if t == "import_module"]
                relative_nodes = [n for n, t in zip(nodes, tags) if t == "relative_import"]
                import_names = [n for n, t in zip(nodes, tags) if t == "import_name"]
                aliases = [n for n, t in zip(nodes, tags) if t == "import_alias"]
                wildcards = [n for n, t in zip(nodes, tags) if t == "wildcard"]

                # Get module path (from X import y)
                module_path = ""
                if module_nodes:
                    module_path = get_ts_node_text(module_nodes[0])
                elif relative_nodes:
                    rel_node = relative_nodes[0]
                    # Handle relative imports (from . import x, from .. import y)
                    dots = ""
                    module_name = ""

                    for child in rel_node.children:
                        if child.type == ".":
                            dots += "."
                        elif child.type == "dotted_name":
                            module_name = get_ts_node_text(child)

                    # Resolve relative path based on current file
                    if dots:
                        current_parts = Path(rel_file_path).parent.parts
                        level = len(dots)
                        if level > len(current_parts):
                            # Going beyond repo root
                            module_path = dots + module_name
                        else:
                            base_parts = current_parts[:-level] if level > 0 else current_parts
                            if module_name:
                                base_parts = list(base_parts) + [module_name]
                            module_path = ".".join(base_parts)
                    else:
                        module_path = module_name

                # Process import names (the Y part in "from X import Y")
                if wildcards:
                    # from module import *
                    imports["*"] = module_path
                else:
                    for i, name_node in enumerate(import_names):
                        name = get_ts_node_text(name_node)
                        # Check if this name has an alias
                        alias = None
                        for alias_node in aliases:
                            if alias_node.parent and name_node.parent and alias_node.parent.id == name_node.parent.id:
                                alias = get_ts_node_text(alias_node)
                                break

                        display_name = alias if alias else name
                        full_path = f"{module_path}.{name}" if module_path else name
                        imports[display_name] = full_path

        self.import_map[rel_file_path] = imports

    def _perform_static_analysis_ts(self, func_def_node: "Node", code_snippet: str, file_imports_flat: List[str]) -> Dict[str, List[str]]:
        """Performs static analysis on a function's AST node and code snippet to identify potential LLM roles and configurations."""
        hints: Dict[str, List[str]] = {"potential_roles": [], "potential_configs": []}
        potential_roles: Set[str] = set()
        potential_configs: Set[str] = set()

        # 1. Analyze file-level imports
        llm_library_keywords = ["openai", "anthropic", "ollama", "huggingface_hub", "transformers", "google.generativeai", "vertexai", "langchain", "llama_index", "semantic_kernel", "litellm", "guidance", "cohere", "ai21", "gemini", "mistralai", "instructor", "sentence_transformers"]
        for imp_module_name in file_imports_flat:
            base_module = imp_module_name.split(".")[0].lower()
            if any(lib_kw == base_module for lib_kw in llm_library_keywords):
                potential_roles.add("API_Caller")
                if any(adv_lib == base_module for adv_lib in ["langchain", "llama_index", "semantic_kernel", "guidance"]):
                    potential_roles.add("Prompt_Builder")
                    potential_roles.add("Context_Retriever")

        # 2. Tree-sitter AST analysis for function body
        # API_Caller: Look for function calls that suggest LLM interaction
        for call_cap_node, call_tag in self.ts_query_call.captures(func_def_node):
            if call_tag == "call_node":
                func_name_str = ""
                func_sub_node = call_cap_node.child_by_field_name("function")
                if func_sub_node:
                    if func_sub_node.type == "identifier":
                        func_name_str = get_ts_node_text(func_sub_node)
                    elif func_sub_node.type == "attribute":
                        attr_node = func_sub_node.child_by_field_name("attribute")
                        if attr_node:
                            func_name_str = get_ts_node_text(attr_node)

                api_call_keywords = ["generate", "create", "invoke", "call", "run", "predict", "complete", "completion", "chat", "embed", "ask", "query", "infer", "send_request", "stream", "agenerate", "acompletion", "achat", "async_generate"]
                if func_name_str and any(keyword in func_name_str.lower() for keyword in api_call_keywords):
                    potential_roles.add("API_Caller")

                # Argument analysis for configs and roles
                args_node = call_cap_node.child_by_field_name("arguments")
                if args_node:
                    for child_arg in args_node.children:
                        if child_arg.type == "keyword_argument":
                            name_node = child_arg.child_by_field_name("name")
                            if name_node:
                                arg_name = get_ts_node_text(name_node)
                                # Config detection
                                if arg_name.upper() in KNOWN_LLM_CONFIGS or arg_name in KNOWN_LLM_CONFIGS:
                                    potential_configs.add(arg_name)
                                # Role detection from argument names
                                if arg_name in ["model", "prompt", "messages", "max_tokens", "temperature", "system_prompt", "tools", "tool_choice", "engine", "deployment_id", "api_key", "api_base", "model_name"]:
                                    potential_roles.add("API_Caller")
                                if arg_name in ["history", "chat_history", "memory", "conversation", "previous_messages", "message_history"]:
                                    potential_roles.add("History_Manager")

        # Assignment analysis for Prompt_Builder and Configurations
        for assign_node, assign_tag in self.ts_query_assignment.captures(func_def_node):
            if assign_tag == "assign_stmt":
                var_name_node = assign_node.child_by_field_name("left")
                value_node = assign_node.child_by_field_name("right")

                if var_name_node and var_name_node.type == "identifier":
                    var_name = get_ts_node_text(var_name_node)
                    # Prompt_Builder from variable names
                    prompt_var_keywords = ["prompt", "messages", "query", "template", "instruction", "context_str", "system_message", "user_message", "assistant_message", "system_prompt", "user_prompt"]
                    if any(keyword in var_name.lower() for keyword in prompt_var_keywords):
                        if value_node and value_node.type in ["string", "binary_operator", "call", "f_string_content", "concatenation"]:
                            potential_roles.add("Prompt_Builder")

                    # Configurations from variable names
                    if var_name.upper() in KNOWN_LLM_CONFIGS or var_name in KNOWN_LLM_CONFIGS:
                        potential_configs.add(var_name)
                    # General config-like variable names
                    config_var_keywords = ["api_key", "model", "temperature", "token", "endpoint", "base_url", "max_tokens", "secret", "engine", "deployment_id", "llm_config", "top_p", "frequency_penalty", "presence_penalty", "timeout", "api_version", "organization", "project_id"]
                    if any(k_conf.lower() in var_name.lower() for k_conf in config_var_keywords):
                        potential_configs.add(var_name)

        # String literals and f-strings for Prompt_Builder
        for str_node_cap, tag_name in self.ts_query_string_literal.captures(func_def_node) + self.ts_query_fstring_content.captures(func_def_node):
            node_text = get_ts_node_text(str_node_cap)
            # More precise f-string detection
            is_fstring = tag_name == "f_content" or str_node_cap.type == "f_string_content" or (str_node_cap.type == "string" and (node_text.startswith("f'") or node_text.startswith('f"')))

            # Template detection for Prompt_Builder
            has_placeholders = ("{" in node_text and "}" in node_text) or "%s" in node_text or "%(" in node_text

            if (is_fstring and has_placeholders) or (has_placeholders and ".format(" in code_snippet):
                potential_roles.add("Prompt_Builder")

            # Keywords in strings indicating prompt templates
            prompt_template_indicators = ["user:", "assistant:", "system:", "context:", "question:", "answer:", "human:", "ai:", "<user>", "<assistant>", "<system>"]
            if len(node_text) > 50 and any(kw in node_text.lower() for kw in prompt_template_indicators):
                potential_roles.add("Prompt_Builder")

        # Template libraries indicate Prompt_Builder
        template_libs_pattern = r"(template|MessagePromptTemplate|ChatPromptTemplate|PromptTemplate|jinja2|mustache|FewShotPromptTemplate|PromptDialog)"
        if re.search(template_libs_pattern, code_snippet, re.IGNORECASE):
            potential_roles.add("Prompt_Builder")

        # Response_Parser detection
        parser_keywords = [r"json\.loads", r"xml\.etree", r"BeautifulSoup", r"parse_xml_response", r"OutputParser", r"PydanticOutputParser", r"loads_json_response", r"\.parse\b", r"\.extract\b", r"parser\.(?:parse|parse_partial)", r"response\.", r"response\.get\b", r"json\.dump", r"\.to_json\b", r"parse_result", r"extract_content", r"\.content\b", r"\.text\b"]
        if any(re.search(kw, code_snippet, re.IGNORECASE) for kw in parser_keywords):
            potential_roles.add("Response_Parser")

        # Error_Handler detection
        if self.ts_query_try_stmt.captures(func_def_node) or re.search(r"except\s+(\w+Error|Exception)", code_snippet):
            error_keywords = ["APIError", "RateLimitError", "AuthenticationError", "timeout", "retry", "backoff", "ConnectionError", "ServiceUnavailableError", "QuotaExceededError", "InvalidRequestError", "ServerOverloadedError", "TokenLimitError"]
            if any(re.search(rf"\b{kw}\b", code_snippet, re.IGNORECASE) for kw in error_keywords):
                potential_roles.add("Error_Handler")

        # 3. Regex-based analysis for remaining roles
        # LLM library direct mentions in code
        llm_lib_pattern = r"\b(openai|anthropic|huggingface|langchain|llama_index|semantic_kernel|google\.generativeai|vertexai|litellm|guidance|cohere|ai21|gemini|mistralai|instructor)\b"
        if re.search(llm_lib_pattern, code_snippet, re.IGNORECASE):
            potential_roles.add("API_Caller")

        # Environment variables for configs
        env_matches = re.findall(r"os\.(?:environ\.get|getenv)\s*$\s*['\"]([^'\"]+)['\"]", code_snippet)
        for key in env_matches:
            if key.upper() in KNOWN_LLM_CONFIGS or key in KNOWN_LLM_CONFIGS:
                potential_configs.add(key)
            elif key.isupper() and any(k_part in key for k_part in ["KEY", "TOKEN", "ENDPOINT", "SECRET", "URL", "ID", "MODEL", "PROJECT", "REGION", "CONFIG", "API"]):
                potential_configs.add(key)

        # Direct config string literals
        for config_key in KNOWN_LLM_CONFIGS:
            if re.search(rf"['\"]({re.escape(config_key)})['\"]", code_snippet) or re.search(rf"\b{re.escape(config_key)}\s*[:=]", code_snippet, re.IGNORECASE):
                potential_configs.add(config_key)

        # Context_Retriever detection
        retriever_keywords = [r"search\b", r"retrieve\b", r"vector_store", r"knowledge_base", r"db\.query", r"similarity_search", r"retriever\.", r"annoy", r"faiss", r"chroma", r"pinecone", r"weaviate", r"elasticsearch", r"rag", r"top_k", r"similarity_threshold", r"nearest_neighbors", r"document_retrieval", r"query_documents", r"fetch_context"]
        if any(re.search(kw, code_snippet, re.IGNORECASE) for kw in retriever_keywords):
            potential_roles.add("Context_Retriever")

        # History_Manager detection
        history_keywords = [r"history\.append", r"memory\.save_context", r"conversation_buffer", r"chat_history", r"ConversationChain", r"load_memory_variables", r"ChatMessageHistory", r"ConversationBufferMemory", r"previous_messages", r"message_history", r"append_message", r"add_message", r"update_history", r"store_conversation", r"get_chat_history", r"conversation_state"]
        if any(re.search(kw, code_snippet, re.IGNORECASE) for kw in history_keywords):
            potential_roles.add("History_Manager")

        # Embedding_Generator detection
        embedding_keywords = [r"\.embed_documents", r"\.embed_query", r"get_embedding", r"embeddings?\.", r"Embedding", r"SentenceTransformer", r"CohereEmbeddings", r"OpenAIEmbeddings", r"similarity\b", r"vector\b", r"encode\b", r"embedding_function", r"text_to_embedding", r"create_embeddings", r"vectorize", r"ada_embedding"]
        if any(re.search(kw, code_snippet, re.IGNORECASE) for kw in embedding_keywords):
            potential_roles.add("Embedding_Generator")

        # Knowledge_Updater detection
        knowledge_keywords = [r"update_knowledge", r"add_document", r"index_document", r"upsert", r"update_index", r"add_texts", r"vectorstore\.add", r"knowledge_base\.add", r"db\.insert", r"update_vectorstore", r"index\.add", r"store_document", r"ingest_documents", r"create_index", r"update_database"]
        if any(re.search(kw, code_snippet, re.IGNORECASE) for kw in knowledge_keywords):
            potential_roles.add("Knowledge_Updater")

        # Input_Validator detection
        input_validation_keywords = [r"validate_input", r"check_input", r"is_valid", r"validator", r"schema\.validate", r"input_validation", r"validate\b", r"assert\s+\w+", r"if\s+not\s+\w+:", r"raise\s+ValueError", r"input_verification", r"sanitize_input", r"check_parameters", r"verify_inputs"]
        if any(re.search(kw, code_snippet, re.IGNORECASE) for kw in input_validation_keywords):
            potential_roles.add("Input_Validator")

        # Output_Processor detection
        output_keywords = [r"process_response", r"format_output", r"transform_result", r"postprocess", r"output_formatter", r"response_handler", r"clean_response", r"format_result", r"prepare_output", r"structure_output", r"enhance_response", r"filter_response", r"transform_output", r"beautify_output"]
        if any(re.search(kw, code_snippet, re.IGNORECASE) for kw in output_keywords):
            potential_roles.add("Output_Processor")

        # Data_Sanitizer detection
        sanitizer_keywords = [r"sanitize", r"clean_data", r"remove_pii", r"scrub_data", r"filter_sensitive", r"escape_html", r"strip_tags", r"secure_data", r"normalize_data", r"redact", r"anonymize", r"mask_data", r"clean_text", r"filter_content", r"content_filter"]
        if any(re.search(kw, code_snippet, re.IGNORECASE) for kw in sanitizer_keywords):
            potential_roles.add("Data_Sanitizer")

        hints["potential_roles"] = sorted(list(potential_roles & set(LLM_ROLES)))
        hints["potential_configs"] = sorted(list(set(potential_configs)))
        return hints

    def _get_ts_docstring(self, func_def_node: "Node") -> str:
        """Extracts the docstring from a function definition node."""
        body_node = func_def_node.child_by_field_name("body")
        if body_node and body_node.named_child_count > 0:
            first_statement_in_body = body_node.named_children[0]
            if first_statement_in_body.type == "expression_statement":
                string_node_candidate = first_statement_in_body.child(0)
                if string_node_candidate and string_node_candidate.type == "string":
                    docstring_text = get_ts_node_text(string_node_candidate)
                    match = re.match(r"^[urfURF]*['\"]{3}(.*?)['\"]{3}$", docstring_text, re.DOTALL)
                    if match:
                        return match.group(1).strip()
                    elif (docstring_text.startswith("'") and docstring_text.endswith("'") and "\n" in docstring_text) or (docstring_text.startswith('"') and docstring_text.endswith('"') and "\n" in docstring_text):
                        return docstring_text[1:-1].strip()
        return ""

    def _get_ts_simplified_signature(self, func_def_node: "Node") -> str:
        """Extracts a simplified signature string from a function node."""
        name_node = func_def_node.child_by_field_name("name")
        params_node = func_def_node.child_by_field_name("parameters")
        func_name = get_ts_node_text(name_node) if name_node else "anonymous_function"

        args_list = []
        if params_node:
            for param_child_node in params_node.named_children:
                param_text = ""
                if param_child_node.type == "identifier":
                    param_text = get_ts_node_text(param_child_node)
                elif param_child_node.type == "typed_parameter":
                    id_node = param_child_node.child(0)
                    if id_node and id_node.type == "identifier":
                        param_text = get_ts_node_text(id_node)
                elif param_child_node.type == "default_parameter":
                    id_node = param_child_node.child_by_field_name("name")
                    if id_node:
                        param_text = get_ts_node_text(id_node)
                elif param_child_node.type in ["list_splat_pattern", "tuple_splat_pattern"]:
                    actual_name_node = param_child_node.child_by_field_name("name")
                    if actual_name_node:
                        param_text = f"*{get_ts_node_text(actual_name_node)}"
                    elif len(param_child_node.children) > 0 and param_child_node.children[-1].type == "identifier":
                        param_text = f"*{get_ts_node_text(param_child_node.children[-1])}"
                elif param_child_node.type == "dictionary_splat_pattern":
                    actual_name_node = param_child_node.child_by_field_name("name")
                    if actual_name_node:
                        param_text = f"**{get_ts_node_text(actual_name_node)}"
                    elif len(param_child_node.children) > 0 and param_child_node.children[-1].type == "identifier":
                        param_text = f"**{get_ts_node_text(param_child_node.children[-1])}"
                elif param_child_node.type == "keyword_separator":
                    param_text = "*"
                elif param_child_node.type == "positional_separator":
                    param_text = "/"

                if param_text:
                    args_list.append(param_text)

        return f"def {func_name}({', '.join(args_list)})"

    def _generate_function_embedding(self, func_node: FunctionNode) -> Optional[List[float]]:
        """Generate embedding for a function based on its code and docstring."""
        # Combine function signature, docstring, and code snippet for embedding
        embedding_text = f"{func_node.signature}\n{func_node.docstring}\n{func_node.code_snippet}"
        embedding_res = get_embedding(text=embedding_text, base_url=self.args.embedding_url, api_key=self.args.embedding_key, model=self.args.embedding_model, output_dir=self.args.output_dir)
        return embedding_res.data[0].embedding

    def _extract_function_nodes_ts(self, file_path: Path, tree: "Tree"):
        """Recursively extracts function and method definitions from a file's AST, populates their static_analysis_hints and embeddings."""
        rel_path_str = str(file_path.relative_to(self.repo_path))

        # Get simplified top-level import module names for this file
        file_import_details = self.import_map.get(rel_path_str, {})
        simplified_imports_for_file = list(set(val for val in file_import_details.values() if val and val != "*"))

        def recurse_extract(node: "Node", current_class_id: Optional[str] = None, current_class_name: Optional[str] = None):
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if not name_node:
                    return

                func_name = get_ts_node_text(name_node)
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Get full code snippet for the function
                file_lines = self.file_lines_cache.get(rel_path_str, [])
                code_snippet_lines = file_lines[start_line - 1 : end_line]
                code_snippet = "\n".join(code_snippet_lines)

                # Construct unique ID
                func_id = f"{current_class_id}:{func_name}" if current_class_id else f"{rel_path_str}::{func_name}"

                if func_id in self.kg.nodes:
                    return
                signature = self._get_ts_simplified_signature(node)
                docstring = self._get_ts_docstring(node)

                # Perform static analysis for LLM-related hints
                static_hints = self._perform_static_analysis_ts(node, code_snippet, simplified_imports_for_file)

                # ... existing code ...

                func_node_obj = FunctionNode(id=func_id, name=func_name, file_path=rel_path_str, start_line=start_line, end_line=end_line, signature=signature, code_snippet=code_snippet, docstring=docstring, module_id=rel_path_str, containing_class_id=current_class_id, static_analysis_hints=static_hints, imports=simplified_imports_for_file, llm_roles=[], uses_configs=[])

                # Generate embedding for the function
                func_node_obj.embedding = self._generate_function_embedding(func_node_obj)

                self.kg.nodes[func_id] = func_node_obj
                self.kg.call_graph.add_node(func_id)

            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    class_name_str = get_ts_node_text(name_node)
                    new_class_id = f"{rel_path_str}:{class_name_str}"
                    body_node = node.child_by_field_name("body")
                    if body_node:
                        for child in body_node.children:
                            recurse_extract(child, new_class_id, class_name_str)

            # Recurse into children that might contain definitions
            if node.type not in ["function_definition", "class_definition"]:
                for child_node in node.children:
                    recurse_extract(child_node, current_class_id, current_class_name)
            elif node.type == "function_definition":
                body_node = node.child_by_field_name("body")
                if body_node:
                    for child_node in body_node.children:
                        recurse_extract(child_node, current_class_id, current_class_name)

        if tree and tree.root_node:
            for child_node in tree.root_node.children:
                recurse_extract(child_node)

    def _find_ts_node_for_function(self, file_tree_root: "Node", func_node_obj: FunctionNode) -> Optional["Node"]:
        """Finds the Tree-sitter AST node corresponding to a FunctionNode object."""
        if not file_tree_root:
            return None

        for capture_node, _ in self.ts_query_func_def.captures(file_tree_root):
            if not (capture_node.start_point[0] <= func_node_obj.start_line - 1 <= capture_node.end_point[0]):
                continue

            name_node = capture_node.child_by_field_name("name")
            if name_node and get_ts_node_text(name_node) == func_node_obj.name:
                if func_node_obj.containing_class_id:
                    parent_class_node = capture_node.parent
                    while parent_class_node and parent_class_node.type != "class_definition":
                        parent_class_node = parent_class_node.parent

                    if parent_class_node:
                        class_name_node = parent_class_node.child_by_field_name("name")
                        if class_name_node:
                            class_name_from_ast = get_ts_node_text(class_name_node)
                            expected_class_id = f"{func_node_obj.file_path}:{class_name_from_ast}"
                            if func_node_obj.containing_class_id == expected_class_id:
                                return capture_node
                    continue
                else:
                    return capture_node
        return None

    def _resolve_call_ts(self, call_node_ts: "Node", caller_node_obj: FunctionNode) -> Optional[str]:
        """Attempts to resolve a Tree-sitter call_node to a FunctionNode.id in the KG."""
        current_file_path_str = caller_node_obj.file_path
        file_imports_map = self.import_map.get(current_file_path_str, {})

        function_sub_node = call_node_ts.child_by_field_name("function")
        if not function_sub_node:
            return None

        invoked_name: Optional[str] = None
        base_object_name: Optional[str] = None
        full_call_path: Optional[str] = None

        if function_sub_node.type == "identifier":
            invoked_name = get_ts_node_text(function_sub_node)
        elif function_sub_node.type == "attribute":
            attr_node = function_sub_node.child_by_field_name("attribute")
            if attr_node:
                invoked_name = get_ts_node_text(attr_node)

            # Handle nested attribute chains
            obj_chain = []
            current_node = function_sub_node
            while current_node and current_node.type == "attribute":
                attr = current_node.child_by_field_name("attribute")
                if attr:
                    obj_chain.insert(0, get_ts_node_text(attr))

                obj_node = current_node.child_by_field_name("object")
                if not obj_node:
                    break

                if obj_node.type == "identifier":
                    obj_chain.insert(0, get_ts_node_text(obj_node))
                    base_object_name = get_ts_node_text(obj_node)
                    break

                current_node = obj_node

            if len(obj_chain) > 1:
                full_call_path = ".".join(obj_chain)
                invoked_name = obj_chain[-1]
                base_object_name = obj_chain[0]

        if not invoked_name:
            return None

        # Filter out stdlib/builtin calls
        if invoked_name in STD_LIB_SYMBOLS and not base_object_name and invoked_name not in file_imports_map:
            return None

        # Enhanced import resolution for cross-file calls
        if base_object_name:
            # Case 1: module.function() or class.method()
            if base_object_name in file_imports_map:
                imported_path = file_imports_map[base_object_name]

                # Try finding the full function path in our knowledge graph
                module_path = imported_path.replace(".", "/")
                if not module_path.endswith(".py"):
                    module_path += ".py"

                # Try as a module function
                function_id = f"{module_path}::{invoked_name}"
                if function_id in self.kg.nodes:
                    return function_id

                # Or it might be a class method
                parts = imported_path.split(".")
                if parts and parts[-1][0].isupper():
                    class_path = "/".join(parts[:-1]) if len(parts) > 1 else ""
                    class_name = parts[-1]
                    if class_path:
                        class_path += ".py"
                        method_id = f"{class_path}:{class_name}:{invoked_name}"
                        if method_id in self.kg.nodes:
                            return method_id

            # Case 2: self.method()
            elif base_object_name == "self" and caller_node_obj.containing_class_id:
                class_id = caller_node_obj.containing_class_id
                method_id = f"{class_id}:{invoked_name}"
                if method_id in self.kg.nodes:
                    return method_id
        else:
            # Direct function call without object/module prefix

            # Check if it's an imported function
            if invoked_name in file_imports_map:
                imported_path = file_imports_map[invoked_name]

                if "." in imported_path:
                    parts = imported_path.split(".")
                    function_name = parts[-1]
                    module_path = "/".join(parts[:-1]) + ".py"
                    function_id = f"{module_path}::{function_name}"
                    if function_id in self.kg.nodes:
                        return function_id
                else:
                    for node_id in self.kg.nodes:
                        node = self.kg.nodes[node_id]
                        if node.name == invoked_name:
                            file_path = node.file_path
                            if imported_path in file_path:
                                return node_id

            # Check for functions in the current file
            current_file_function_id = f"{current_file_path_str}::{invoked_name}"
            if current_file_function_id in self.kg.nodes:
                return current_file_function_id

            # Check if it's a method in the current class
            if caller_node_obj.containing_class_id:
                method_id = f"{caller_node_obj.containing_class_id}:{invoked_name}"
                if method_id in self.kg.nodes:
                    return method_id

        # If no match found, attempt a deeper search across all nodes
        for node_id, node in self.kg.nodes.items():
            if node.name == invoked_name:
                if node.file_path != current_file_path_str:
                    return node_id

        return None

    def _build_call_graph(self):
        """Builds the call graph by resolving function calls within each FunctionNode."""
        logging.info("[Agent 1] Building function call graph...")
        call_edges_added = 0
        total_potential_calls_analyzed = 0

        for caller_id, caller_node_obj in tqdm(self.kg.nodes.items(), desc="Building Call Graph"):
            file_rel_path = caller_node_obj.file_path
            if file_rel_path not in self.parsed_files_ts or not self.parsed_files_ts[file_rel_path]:
                logging.info(f"Skipping call graph for {caller_id}, file AST {file_rel_path} not found/parsed.")
                continue

            file_tree = self.parsed_files_ts[file_rel_path]
            assert file_tree is not None

            # Find the AST node for the caller function
            caller_ts_node = self._find_ts_node_for_function(file_tree.root_node, caller_node_obj)
            if not caller_ts_node:
                logging.warning(f"Could not find AST node for function {caller_id} in {file_rel_path}. Skipping its internal calls.")
                continue

            # Look for calls only within the body of this specific function
            function_body_node = caller_ts_node.child_by_field_name("body")
            if not function_body_node:
                continue

            try:
                for call_ts_node_capture, call_tag_capture in self.ts_query_call.captures(function_body_node):
                    if call_tag_capture == "call_node":
                        total_potential_calls_analyzed += 1
                        callee_id = self._resolve_call_ts(call_ts_node_capture, caller_node_obj)
                        if callee_id and callee_id in self.kg.nodes and caller_id != callee_id:
                            if not self.kg.call_graph.has_edge(caller_id, callee_id):
                                self.kg.call_graph.add_edge(caller_id, callee_id, type="CALLS")
                                call_edges_added += 1
            except Exception as e:
                logging.error(f"Error processing calls within function {caller_node_obj.id}: {e}", exc_info=True)

        logging.info(f"[Agent 1] Call graph built. Analyzed ~{total_potential_calls_analyzed} call sites, added {call_edges_added} unique internal call edges.")

    def build_kg_with_static_analysis(self) -> CodeKnowledgeGraph:
        """Main method for Agent 1. Orchestrates scanning, parsing, FunctionNode creation (with static hints and embeddings), and call graph construction."""
        logging.info("[Agent 1] Starting Code Knowledge Graph construction with static analysis and embeddings...")

        python_files = self._find_python_files()
        for file_path in tqdm(python_files, desc="Agent 1: Parsing Files & Extracting Nodes"):
            tree = self._parse_file_ts(file_path)
            if tree:
                self._extract_function_nodes_ts(file_path, tree)

        if not self.kg.nodes:
            logging.error("[Agent 1] No function nodes extracted. KG building aborted.")
            return self.kg

        logging.info(f"[Agent 1] Extracted {len(self.kg.nodes)} function/method nodes with static hints and embeddings.")

        self._build_call_graph()

        logging.info("[Agent 1] Code Knowledge Graph construction with static analysis and embeddings completed.")
        return self.kg


# --- Agent 2: Defect Analyzer ---
class DefectAnalyzer:
    """Agent 2: Interprets the defect description using an LLM and performs full matching recall (static + embedding supplement)."""

    def __init__(self, kg: CodeKnowledgeGraph, args: argparse.Namespace):
        self.kg = kg
        self.args = args
        self.llm_roles_set = set(LLM_ROLES)

    def _get_repository_overview(self) -> Dict[str, Any]:
        """Extracts a concise JSON-like repository structure from the KG."""
        repo_structure = {}

        if not self.kg.nodes:
            repo_structure["summary"] = "No code elements found in the knowledge graph."
            return repo_structure

        # Build the hierarchical structure
        for node_id, node in self.kg.nodes.items():
            file_path = node.file_path
            if file_path not in repo_structure:
                repo_structure[file_path] = {"functions": set(), "classes": {}}

            if node.containing_class_id:
                class_name = node.containing_class_id.split(":")[-1]
                if class_name not in repo_structure[file_path]["classes"]:
                    repo_structure[file_path]["classes"][class_name] = set()
                repo_structure[file_path]["classes"][class_name].add(node.name)
            else:
                repo_structure[file_path]["functions"].add(node.name)

        # Convert to JSON-serializable format with limits
        MAX_FILES = 75
        MAX_FUNCTIONS = 20
        MAX_CLASSES = 15
        MAX_METHODS = 15

        result = {}
        for i, (file_path, file_data) in enumerate(sorted(repo_structure.items())):
            if i >= MAX_FILES:
                break

            file_entry = {}

            # Add functions
            functions = sorted(list(file_data["functions"]))
            file_entry["functions"] = functions[:MAX_FUNCTIONS]
            if len(functions) > MAX_FUNCTIONS:
                file_entry["functions_truncated"] = True

            # Add classes with methods
            file_entry["classes"] = {}
            for j, (class_name, methods) in enumerate(sorted(file_data["classes"].items())):
                if j >= MAX_CLASSES:
                    file_entry["classes_truncated"] = True
                    break

                methods_list = sorted(list(methods))
                file_entry["classes"][class_name] = methods_list[:MAX_METHODS]
                if len(methods_list) > MAX_METHODS:
                    file_entry["classes"][class_name + "_truncated"] = True

            result[file_path] = file_entry

        # Add stats
        result["_stats"] = {"total_files": len(repo_structure), "files_shown": min(len(repo_structure), MAX_FILES), "note": "Some elements may be truncated due to size limits"}

        return result

    def _llm_analyze_defect_for_summary(self, explanation: str, consequences: str, tests: str, repo_overview: Dict[str, Any]) -> DefectAnalysisSummary:
        """Uses LLM (once) to analyze defect description and repo overview. Returns a structured DefectAnalysisSummary."""
        logging.info("[Agent 2] Calling LLM to analyze defect description for summary...")

        system_prompt = f"""You are an expert at localizing LLM-specific software defects.
Analyze the defect description and repository overview to identify code location clues.

Think about:
1. Interaction boundary issues (API calls, data mapping)
2. Prompt engineering flaws (unclear prompts, context issues)
3. Input/Output format problems (validation, formats)
4. Context management defects (dialogue history, context windows)
5. Tool usage errors (function calling issues, parsing)
6. Configuration errors (API keys, model settings)

Output ONLY a JSON via the `extract_defect_analysis_summary` function call with:
- "relevant_llm_roles": List of involved roles from: {LLM_ROLES}
- "relevant_configs": List of related LLM configuration items (e.g., API keys, model names)
- "potential_locations": List of *EXACTLY 5* likely bug locations in KG-compatible format:
  - For functions: "file_path.py::function_name"
  - For methods: "file_path.py:ClassName:method_name"
  Example: ["src/utils.py::process_data", "src/api.py:ApiClient:send_request"]
- "code_keywords": List of keywords likely near the bug (e.g., "api_key", "response.choices[0]")"""

        user_message = f"""Defect:
{explanation}

Consequences:
{consequences}

Reproduction steps:
{tests}

Repository structure:
{json.dumps(repo_overview, indent=2)}

ONLY call function `extract_defect_analysis_summary` to output your results."""

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_defect_analysis_summary",
                    "description": "Extracts a structured summary of the defect analysis to help locate the bug.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "relevant_llm_roles": {"type": "array", "items": {"type": "string"}, "description": "List of most relevant LLM roles."},
                            "relevant_configs": {"type": "array", "items": {"type": "string"}, "description": "List of relevant LLM configuration items."},
                            "potential_locations": {"type": "array", "items": {"type": "string"}, "description": "List of potential locations in format 'file_path.py::function_name' or 'file_path.py:ClassName:method_name'"},
                            "code_keywords": {"type": "array", "items": {"type": "string"}, "description": "Specific code keywords or short patterns related to the defect."},
                        },
                        "required": ["relevant_llm_roles", "relevant_configs", "potential_locations", "code_keywords"],
                    },
                },
            }
        ]

        # truncate user message to 32768 tokens
        user_message = limit_user_message(user_message=user_message, system_prompt=system_prompt, max_tokens=32768)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

        summary = DefectAnalysisSummary()
        raw_llm_response_json_str = None
        try:
            response = get_llm_res(messages=messages, base_url=self.args.base_url, api_key=self.args.api_key, model=self.args.model, temperature=0.0, tools=tools, tool_choice={"type": "function", "function": {"name": "extract_defect_analysis_summary"}}, output_dir=self.args.output_dir)

            tool_call = response.choices[0].message.tool_calls[0]
            raw_llm_response_json_str = tool_call.function.arguments
            logging.info(f"[Agent 2] LLM raw response for defect: {response}")
            parsed_args = json.loads(raw_llm_response_json_str)

            summary.relevant_llm_roles = [r for r in parsed_args.get("relevant_llm_roles", []) if r in self.llm_roles_set]
            summary.relevant_configs = parsed_args.get("relevant_configs", [])
            summary.potential_locations = parsed_args.get("potential_locations", [])
            summary.code_keywords = parsed_args.get("code_keywords", [])
            summary.raw_llm_response = raw_llm_response_json_str

        except Exception as e:
            logging.error(f"[Agent 2] Error processing LLM response for defect summary: {e}. Raw: {raw_llm_response_json_str}", exc_info=True)

        # Correct class in last position
        corrected_locations = []
        for location in summary.potential_locations:
            if location in self.kg.nodes:
                corrected_locations.append(location)
                continue

            # correct class in last position
            if "::" in location:
                parts = location.split("::")
                file_path = parts[0]
                entity_name = parts[1] if len(parts) > 1 else None
                class_methods = [node for node in self.kg.nodes if node.startswith(f"{file_path}:{entity_name}:")]
                if class_methods:
                    # add top3
                    for method_node in class_methods[:3]:
                        corrected_locations.append(method_node)
        summary.potential_locations = corrected_locations

        logging.info(f"[Agent 2] Defect Analysis Summary: Roles={summary.relevant_llm_roles}, Configs={summary.relevant_configs}, Locations={len(summary.potential_locations)}, Keywords={len(summary.code_keywords)}")
        return summary

    def _calculate_static_match_score(self, func_node: FunctionNode, defect_summary: DefectAnalysisSummary) -> Tuple[float, str, Dict[str, float]]:
        """Calculates a match score between a FunctionNode's static hints and the DefectAnalysisSummary."""
        score = 0.0
        reasons: List[str] = []

        node_hints = func_node.static_analysis_hints

        # record the score components
        score_components = {"role_match_score": 0.0, "config_match_score": 0.0, "location_match_score": 0.0, "keyword_match_score": 0.0}

        # 1. Match potential roles
        if node_hints.get("potential_roles") and defect_summary.relevant_llm_roles:
            role_matches = set(node_hints["potential_roles"]) & set(defect_summary.relevant_llm_roles)
            if role_matches:
                role_score = len(role_matches) * 2.5
                score += role_score
                score_components["role_match_score"] = role_score
                reasons.append(f"Role match: {', '.join(role_matches)}")

        # 2. Match potential configs
        if node_hints.get("potential_configs") and defect_summary.relevant_configs:
            node_configs_lower = {c.lower() for c in node_hints["potential_configs"]}
            summary_configs_lower = {c.lower() for c in defect_summary.relevant_configs}
            config_matches = node_configs_lower & summary_configs_lower
            if config_matches:
                config_score = len(config_matches) * 1.5
                score += config_score
                score_components["config_match_score"] = config_score
                reasons.append(f"Config match: {', '.join(config_matches)}")

        # 3. Match FunctionNode's location with potential_locations
        for loc_guess in defect_summary.potential_locations:
            if "::" in loc_guess:
                parts = loc_guess.split("::")
                guess_file = parts[0]
                guess_class = None
                guess_method = parts[1] if len(parts) > 1 else None
            else:
                parts = loc_guess.split(":")
                guess_file = parts[0]
                guess_class = parts[1] if len(parts) > 1 else None
                guess_method = parts[2] if len(parts) > 2 else None

            # File match
            file_match_score = 0
            if guess_file and guess_file in func_node.file_path:
                file_match_score = 1.0
                if func_node.file_path.endswith(guess_file):
                    file_match_score = 1.5

            if not file_match_score > 0 and guess_file:
                continue

            method_match_score = 0
            if guess_method and guess_method == func_node.name:
                method_match_score = 2.0

            class_match_score = 0
            if guess_class and func_node.containing_class_id:
                node_class_name = func_node.containing_class_id.split(":")[-1]
                if guess_class == node_class_name:
                    class_match_score = 1.5

            current_loc_score = 0
            temp_loc_reasons = []

            if file_match_score > 0:
                current_loc_score += file_match_score
                temp_loc_reasons.append(f"File({guess_file or 'any'})")
                if class_match_score > 0:
                    current_loc_score += class_match_score
                    temp_loc_reasons.append(f"Class({guess_class})")
                    if method_match_score > 0:
                        current_loc_score += method_match_score
                        temp_loc_reasons.append(f"Method({guess_method})")
                elif guess_class is None and method_match_score > 0:
                    current_loc_score += method_match_score
                    temp_loc_reasons.append(f"Method({guess_method})")
                elif guess_class is None and guess_method is None:
                    pass

            if current_loc_score > 0:
                loc_score = current_loc_score * 1.5
                score += loc_score
                score_components["location_match_score"] = loc_score
                reasons.append(f"Location hint match ({', '.join(temp_loc_reasons)}) for guess: {loc_guess}")

        # 4. Match code_keywords
        if defect_summary.code_keywords:
            keyword_hits = []
            combined_text = func_node.code_snippet.lower() + "\n" + func_node.docstring.lower()
            for kw in defect_summary.code_keywords:
                if kw.lower() in combined_text:
                    keyword_hits.append(kw)
            if keyword_hits:
                keyword_hits_score = len(keyword_hits) * 0.75
                score += keyword_hits_score
                score_components["keyword_match_score"] = keyword_hits_score
                reasons.append(f"Keyword match: {', '.join(keyword_hits)}")

        return score, "; ".join(reasons) if reasons else "No specific static matches found.", score_components

    def _calculate_embedding_similarity(self, func_node: FunctionNode, defect_embedding: List[float]) -> float:
        """Calculate embedding similarity between function and defect description."""
        if not func_node.embedding or not defect_embedding:
            return 0.0

        return cosine_similarity(func_node.embedding, defect_embedding)

    def interpret_defect_and_perform_full_matching_recall(self, explanation: str, consequences: str, tests: str, static_threshold: float = 0.01, embedding_threshold: float = 0.3) -> Tuple[List[Tuple[str, float, str]], DefectAnalysisSummary]:
        """Orchestrates Agent 2: Gets LLM defect summary, then performs full matching recall (static + embedding supplement)."""
        logging.info("[Agent 2] Starting defect interpretation and full matching recall.")
        repo_overview = self._get_repository_overview()

        # Save repository overview for inspection
        repo_overview_path = Path(self.args.output_dir) / "intermediate_repo_overview.json"
        with open(repo_overview_path, "w", encoding="utf-8") as f:
            json.dump(repo_overview, f, indent=2, ensure_ascii=False)
        logging.info(f"[Agent 2] Saved repository overview to {repo_overview_path}")

        defect_summary = self._llm_analyze_defect_for_summary(explanation, consequences, tests, repo_overview)

        # Generate defect embedding for semantic recall
        defect_text = f"{explanation}\n{consequences}\n{tests}"
        defect_embedding_res = get_embedding(text=defect_text, base_url=self.args.embedding_url, api_key=self.args.embedding_key, model=self.args.embedding_model, output_dir=self.args.output_dir)
        defect_embedding = defect_embedding_res.data[0].embedding

        # Phase 1: Full static feature matching
        matched_candidates: List[Tuple[str, float, str]] = []
        matched_node_ids: Set[str] = set()

        if not self.kg.nodes:
            logging.warning("[Agent 2] No function nodes in KG to filter.")
            return [], defect_summary

        logging.info(f"[Agent 2] Phase 1: Full static feature matching on {len(self.kg.nodes)} FunctionNodes...")
        detailed_scoring_records = []
        for func_id, func_node in tqdm(self.kg.nodes.items(), desc="Agent 2: Static Feature Matching"):
            match_score, match_reason, score_components = self._calculate_static_match_score(func_node, defect_summary)
            if match_score > static_threshold:
                matched_candidates.append((func_id, match_score, match_reason))
                matched_node_ids.add(func_id)
                detailed_scoring_records.append({"function_id": func_id, "total_score": match_score, "reason": match_reason, "score_components": score_components})

        matched_candidates.sort(key=lambda x: x[1], reverse=True)

        # Phase 2: Embedding supplement recall for unmatched nodes
        supplement_candidates: List[Tuple[str, float, str]] = []
        if defect_embedding:
            logging.info(f"[Agent 2] Phase 2: Embedding supplement recall for unmatched nodes...")

            for func_id, func_node in tqdm(self.kg.nodes.items(), desc="Agent 2: Embedding Supplement Recall"):
                if func_id in matched_node_ids:
                    continue  # Skip already matched candidates

                similarity = self._calculate_embedding_similarity(func_node, defect_embedding)
                if similarity > embedding_threshold:
                    reason = f"Embedding similarity: {similarity:.3f}"
                    supplement_candidates.append((func_id, similarity, reason))

            supplement_candidates.sort(key=lambda x: x[1], reverse=True)

        # Merge matched and supplement candidates
        all_candidates = matched_candidates + supplement_candidates

        # Save detailed scoring
        scoring_details_path = Path(self.args.output_dir) / "agent2_detailed_scoring.json"
        with open(scoring_details_path, "w", encoding="utf-8") as f:
            json.dump(detailed_scoring_records, f, indent=2, ensure_ascii=False)

        logging.info(f"[Agent 2] Generated {len(matched_candidates)} matched candidates and {len(supplement_candidates)} supplement candidates.")

        return all_candidates, defect_summary


# --- Agent 3: Validator ---
class Validator:
    """Agent 3: Call relationship subgraph aggregation, hierarchical evaluation, and two-stage localization."""

    def __init__(self, kg: CodeKnowledgeGraph, defect_summary: DefectAnalysisSummary, args: argparse.Namespace):
        self.kg = kg
        self.defect_summary = defect_summary
        self.args = args

        # Track evaluated files and function nodes to prevent duplicate processing
        self.evaluated_files: Set[str] = set()
        self.evaluated_function_ids: Set[str] = set()

    def _get_context_nodes(self, func_id: str, context_hops: int = 1) -> Dict[str, Any]:
        """Get context nodes within specified hops from the target function."""
        context = {"callers": [], "callees": [], "imports_in_file": [], "class_definition": "N/A"}

        if func_id not in self.kg.nodes:
            return context

        target_node = self.kg.nodes[func_id]

        # Get callers and callees within context_hops
        current_level_nodes = {func_id}
        all_context_nodes = {func_id}

        for hop in range(context_hops):
            next_level_nodes = set()

            for node_id in current_level_nodes:
                # Get callers (predecessors)
                for caller_id in self.kg.call_graph.predecessors(node_id):
                    if caller_id not in all_context_nodes:
                        next_level_nodes.add(caller_id)
                        caller_node = self.kg.nodes.get(caller_id)
                        if caller_node:
                            context["callers"].append({"id": caller_id, "name": caller_node.name, "file": caller_node.file_path, "signature": caller_node.signature})

                # Get callees (successors)
                for callee_id in self.kg.call_graph.successors(node_id):
                    if callee_id not in all_context_nodes:
                        next_level_nodes.add(callee_id)
                        callee_node = self.kg.nodes.get(callee_id)
                        if callee_node:
                            context["callees"].append({"id": callee_id, "name": callee_node.name, "file": callee_node.file_path, "signature": callee_node.signature})

            all_context_nodes.update(next_level_nodes)
            current_level_nodes = next_level_nodes

            if not current_level_nodes:
                break

        # Get file-level imports
        context["imports_in_file"] = target_node.imports

        # Get class definition if the function is a method
        if target_node.containing_class_id:
            # Find class definition in the same file
            class_name = target_node.containing_class_id.split(":")[-1]
            context["class_definition"] = f"class {class_name}:"

        return context

    def _build_call_relationship_subgraphs(self, all_candidates: List[Tuple[str, float, str]], max_candidates_per_file: int = 3, max_candidates_for_context: int = 50) -> List[CallRelationSubgraphAnalysisUnit]:
        """Build call relationship subgraphs by connecting candidate functions through call relationships."""
        logging.info("[Agent 3] Building call relationship subgraphs...")

        # Apply diversity constraints
        candidate_ids_for_subgraph: Set[str] = set()
        file_candidate_counts: Dict[str, int] = {}

        # Sort by score and apply constraints
        for func_id, score, reason in all_candidates[:max_candidates_for_context]:
            if func_id in self.kg.nodes:
                file_path = self.kg.nodes[func_id].file_path
                current_file_count = file_candidate_counts.get(file_path, 0)

                if current_file_count < max_candidates_per_file:
                    candidate_ids_for_subgraph.add(func_id)
                    file_candidate_counts[file_path] = current_file_count + 1

        # Add LLM-suggested locations
        for location in self.defect_summary.potential_locations:
            if location in self.kg.nodes:
                candidate_ids_for_subgraph.add(location)

        # Build connected components using call relationships
        subgraph = self.kg.call_graph.subgraph(candidate_ids_for_subgraph)
        connected_components = list(nx.connected_components(subgraph.to_undirected()))

        # Create CallRelationSubgraphAnalysisUnit objects
        subgraph_units = []
        for i, component in enumerate(connected_components):
            subgraph_id = f"call_subgraph_{i+1}"

            # Get function nodes in this component
            connected_functions = [self.kg.nodes[func_id] for func_id in component if func_id in self.kg.nodes]

            # Extract call relationships within this component
            call_relationships = []
            for func_id in component:
                if self.kg.call_graph.has_node(func_id):
                    for successor in self.kg.call_graph.successors(func_id):
                        if successor in component:
                            call_relationships.append((func_id, successor))

            # Get involved files
            involved_files = set(func.file_path for func in connected_functions)

            # Generate file summaries
            file_summaries = {}
            for file_path in involved_files:
                file_functions = [func for func in connected_functions if func.file_path == file_path]
                file_summaries[file_path] = self._generate_file_summary(file_path, file_functions)

            # Generate LLM features summary
            llm_features_summary = self._generate_llm_features_summary(connected_functions)

            subgraph_unit = CallRelationSubgraphAnalysisUnit(subgraph_id=subgraph_id, connected_functions=connected_functions, call_relationships=call_relationships, involved_files=involved_files, file_summaries=file_summaries, llm_features_summary=llm_features_summary)
            subgraph_units.append(subgraph_unit)

        logging.info(f"[Agent 3] Created {len(subgraph_units)} call relationship subgraphs.")
        return subgraph_units

    def _generate_file_summary(self, file_path: str, functions: List[FunctionNode]) -> str:
        """Generate a summary of the file based on its functions."""
        if not functions:
            return f"File {file_path} contains no candidate functions."

        class_names = set()
        function_names = []
        llm_roles = set()

        for func in functions:
            if func.containing_class_id:
                class_name = func.containing_class_id.split(":")[-1]
                class_names.add(class_name)
            else:
                function_names.append(func.name)

            llm_roles.update(func.static_analysis_hints.get("potential_roles", []))

        summary_parts = [f"File: {file_path}"]
        if class_names:
            summary_parts.append(f"Classes: {', '.join(sorted(class_names))}")
        if function_names:
            summary_parts.append(f"Functions: {', '.join(function_names[:5])}")
        if llm_roles:
            summary_parts.append(f"LLM Roles: {', '.join(sorted(llm_roles))}")

        return "; ".join(summary_parts)

    def _generate_llm_features_summary(self, functions: List[FunctionNode]) -> Dict[str, Any]:
        """Generate a summary of LLM-related features and global file dependencies in the functions."""
        all_roles = set()
        all_configs = set()

        # Group functions by file
        files_with_functions = {}
        for func in functions:
            file_path = func.file_path
            if file_path not in files_with_functions:
                files_with_functions[file_path] = []
            files_with_functions[file_path].append(func)

        # Create file index mapping
        file_paths = sorted(list(files_with_functions.keys()))
        file_to_index = {file_path: i for i, file_path in enumerate(file_paths)}

        # Build global file dependencies
        global_dependencies = {}

        for file_path, file_funcs in files_with_functions.items():
            file_index = file_to_index[file_path]
            dependencies = set()

            for func in file_funcs:
                func_id = func.id
                if self.kg.call_graph.has_node(func_id):
                    # Check callees (functions this file calls)
                    for callee_id in self.kg.call_graph.successors(func_id):
                        if callee_id in self.kg.nodes:
                            callee_file = self.kg.nodes[callee_id].file_path
                            if callee_file != file_path and callee_file in file_to_index:
                                dependencies.add(file_to_index[callee_file])

                    # Check callers (functions that call this file)
                    for caller_id in self.kg.call_graph.predecessors(func_id):
                        if caller_id in self.kg.nodes:
                            caller_file = self.kg.nodes[caller_id].file_path
                            if caller_file != file_path and caller_file in file_to_index:
                                dependencies.add(file_to_index[caller_file])

            if dependencies:
                global_dependencies[file_index] = sorted(list(dependencies))

        # Collect detailed file information
        file_details = {}
        for file_path, file_funcs in files_with_functions.items():
            file_index = file_to_index[file_path]

            # Collect all unique imports
            all_imports = set()
            for func in file_funcs:
                all_imports.update(func.imports)

            # Collect all function code snippets
            function_codes = []
            class_definitions = set()
            function_signatures = []

            for func in file_funcs:
                # Add full code snippet
                function_codes.append(f"{func.name}:\n{func.code_snippet}")

                # Add signature
                function_signatures.append(func.signature)

                # Add class info
                if func.containing_class_id:
                    class_name = func.containing_class_id.split(":")[-1]
                    class_definitions.add(class_name)

                # Collect LLM features
                all_roles.update(func.static_analysis_hints.get("potential_roles", []))
                all_configs.update(func.static_analysis_hints.get("potential_configs", []))

            file_details[file_index] = {
                "file_path": file_path,
                "imports": sorted(list(all_imports)),
                "classes": sorted(list(class_definitions)),
                "function_signatures": function_signatures,
                "function_codes": function_codes,
                "function_count": len(file_funcs),
            }

        return {
            "llm_roles": sorted(list(all_roles)),
            "llm_configs": sorted(list(all_configs)),
            "function_count": len(functions),
            "file_count": len(files_with_functions),
            "file_index_mapping": file_to_index,
            "global_file_dependencies": global_dependencies,
            "file_details": file_details,
        }

    def _llm_evaluate_file_importance_batch(self, subgraph_units: List[CallRelationSubgraphAnalysisUnit], defect_explanation: str, defect_consequences: str, defect_tests: str) -> List[Tuple[str, float]]:
        """Stage 1: Batch LLM evaluation of file importance across subgraphs with unique scores."""
        # Extract all unique files from subgraphs
        all_files = set()
        for unit in subgraph_units:
            all_files.update(unit.involved_files)

        # Check if any files have already been evaluated
        unevaluated_files = all_files - self.evaluated_files

        if not unevaluated_files:
            logging.info("[Agent 3] All files already evaluated, skipping file importance evaluation.")
            return []

        logging.info(f"[Agent 3] Stage 1: Batch LLM evaluating file importance for {len(unevaluated_files)} files...")

        system_prompt = f"""You are a debugging expert for LLM applications.
Assess which files are most likely to contain the ROOT CAUSE of a described defect.

You will receive:
1. Defect details (explanation, consequences, reproduction steps)
2. File details with complete candidate function code and global dependencies
3. Defect analysis hypothesis

Analysis process (ordered by impact):
1. File-Defect Match: Does file functionality directly relate to defect behavior?
2. LLM Pattern Match: Do file's LLM features align with defect characteristics?
3. Dependency Support: Do global file dependencies suggest involvement?

Score guide:
- 0.8-1.0: Primary file likely containing defect
- 0.6-0.8: Important supporting file
- 0.4-0.6: Secondary contributor
- 0.2-0.4: Peripheral involvement
- 0.0-0.2: Little/no connection

IMPORTANT: You MUST assign DISTINCT scores to each file - no two files can have the same score!

ONLY use the `evaluate_file_importance` function call to output results."""

        # Collect comprehensive file information
        file_to_functions = {}
        all_global_dependencies = {}
        all_file_index_mapping = {}

        # Collect all functions for each file across all subgraph units
        for unit in subgraph_units:
            for func in unit.connected_functions:
                file_path = func.file_path
                if file_path in unevaluated_files:
                    if file_path not in file_to_functions:
                        file_to_functions[file_path] = []
                    file_to_functions[file_path].append(func)

            # Merge global dependencies and file mappings
            llm_features = unit.llm_features_summary
            unit_dependencies = llm_features.get("global_file_dependencies", {})
            unit_file_mapping = llm_features.get("file_index_mapping", {})

            all_global_dependencies.update(unit_dependencies)
            all_file_index_mapping.update(unit_file_mapping)

        # Build comprehensive global file dependencies using function call relationships
        comprehensive_dependencies = {}
        file_paths = sorted(list(file_to_functions.keys()))
        file_to_index = {file_path: i for i, file_path in enumerate(file_paths)}

        for file_path, file_funcs in file_to_functions.items():
            file_index = file_to_index[file_path]
            dependencies = set()

            for func in file_funcs:
                func_id = func.id
                if self.kg.call_graph.has_node(func_id):
                    # Check callees (functions this file calls)
                    for callee_id in self.kg.call_graph.successors(func_id):
                        if callee_id in self.kg.nodes:
                            callee_file = self.kg.nodes[callee_id].file_path
                            if callee_file != file_path and callee_file in file_to_index:
                                dependencies.add(file_to_index[callee_file])

                    # Check callers (functions that call this file)
                    for caller_id in self.kg.call_graph.predecessors(func_id):
                        if caller_id in self.kg.nodes:
                            caller_file = self.kg.nodes[caller_id].file_path
                            if caller_file != file_path and caller_file in file_to_index:
                                dependencies.add(file_to_index[caller_file])

            if dependencies:
                comprehensive_dependencies[file_index] = sorted(list(dependencies))

        # Prepare detailed file information
        files_info = []
        for i, (file_path, file_funcs) in enumerate(sorted(file_to_functions.items())):
            file_index = file_to_index[file_path]
            dependencies = comprehensive_dependencies.get(file_index, [])

            # Get dependency file names
            dep_files = [file_paths[dep_idx] for dep_idx in dependencies if dep_idx < len(file_paths)]

            # Collect all unique imports
            all_imports = set()
            for func in file_funcs:
                all_imports.update(func.imports)

            # Collect class information
            class_definitions = set()
            for func in file_funcs:
                if func.containing_class_id:
                    class_name = func.containing_class_id.split(":")[-1]
                    class_definitions.add(class_name)

            # Collect LLM roles and configs
            llm_roles = set()
            llm_configs = set()
            for func in file_funcs:
                llm_roles.update(func.static_analysis_hints.get("potential_roles", []))
                llm_configs.update(func.static_analysis_hints.get("potential_configs", []))

            # Collect all function signatures and code snippets
            function_signatures = []
            function_codes = []
            for j, func in enumerate(file_funcs):
                function_signatures.append(func.signature)
                # function_codes.append(f"# Function: {func.name}\n{func.code_snippet}")
                # only first 5 lines of code
                code_lines = func.code_snippet.split("\n")
                limited_code = "\n".join(code_lines[:3])
                if len(code_lines) > 3:
                    limited_code += "\n..."
                function_codes.append(f"# Function {j}: {limited_code}")

            file_info = f"""File {file_index}:
- File Path: {file_path}
- Function Count: {len(file_funcs)}
- Imports: {', '.join(sorted(list(all_imports))[:8]) if all_imports else 'None'}
- Classes: {', '.join(sorted(class_definitions)) if class_definitions else 'None'}
- LLM Roles: {', '.join(sorted(llm_roles)) if llm_roles else 'None'}
- LLM Configs: {', '.join(sorted(llm_configs)) if llm_configs else 'None'}
- Complete Function Code: {chr(10).join(function_codes)}"""

            # Function Signatures:
            # {chr(10).join(function_signatures)}

            files_info.append(file_info)

        # sort files_info by file_index
        files_info.sort(key=lambda x: int(re.search(r"File (\d+)", x).group(1)))

        # Build global dependency overview in JSON format
        dependency_overview = {"file_index_mapping": file_to_index, "global_file_dependencies": comprehensive_dependencies, "dependency_summary": f"Total files: {len(file_paths)}, Files with dependencies: {len(comprehensive_dependencies)}"}

        user_message = f"""Defect:
- Explanation: {defect_explanation}
- Consequences: {defect_consequences}
- Reproduction Steps: {defect_tests}

Defect Analysis Hypothesis:
- Relevant LLM Roles: {self.defect_summary.relevant_llm_roles}
- Relevant LLM Configs: {self.defect_summary.relevant_configs}
- Potential Keywords: {self.defect_summary.code_keywords}

Global File Dependencies:
{json.dumps(dependency_overview, indent=2)}

--- Files to Evaluate ---
{chr(10).join(files_info)}
---

Evaluate each file's importance using the `evaluate_file_importance` function tool."""

        tools = [{"type": "function", "function": {"name": "evaluate_file_importance", "description": "Evaluates file importance for defect localization with unique scores", "parameters": {"type": "object", "properties": {"results": {"type": "array", "items": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Path of the file being evaluated"}, "score": {"type": "number", "description": "Unique importance score (0.0-1.0)"}}, "required": ["file_path", "score"]}}}, "required": ["results"]}}}]

        # truncate user message to 32768 tokens
        user_message = limit_user_message(user_message=user_message, system_prompt=system_prompt, max_tokens=32768)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

        file_scores = []
        try:
            response = get_llm_res(messages=messages, base_url=self.args.base_url, api_key=self.args.api_key, model=self.args.model, temperature=0.0, tools=tools, tool_choice={"type": "function", "function": {"name": "evaluate_file_importance"}}, output_dir=self.args.output_dir)

            tool_call = response.choices[0].message.tool_calls[0]
            evaluation_results = json.loads(tool_call.function.arguments)

            # Sort results by LLM score
            llm_results = evaluation_results.get("results", [])
            llm_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            for result in llm_results:
                file_path = result.get("file_path")
                score = result.get("score", 0.0)
                if file_path:
                    file_scores.append((file_path, score))
                    # Mark this file as evaluated
                    self.evaluated_files.add(file_path)
                    logging.info(f"[Agent 3] File importance: {file_path}, Score={score:.2f}")

        except Exception as e:
            logging.error(f"[Agent 3] Error during file importance evaluation: {e}", exc_info=True)
            # Fallback: assign default scores in descending order
            for i, file_path in enumerate(sorted(unevaluated_files)):
                score = max(0.01, 0.9 - (i * 0.1))  # Simple descending scores starting from 0.9
                file_scores.append((file_path, score))
                self.evaluated_files.add(file_path)

        return file_scores

    def _llm_evaluate_functions_in_top_files_batch(self, top_files: List[str], subgraph_units: List[CallRelationSubgraphAnalysisUnit], defect_explanation: str, defect_consequences: str, defect_tests: str) -> List[Tuple[str, float, str, int]]:
        """Stage 2: Batch LLM evaluation of functions within top-ranked files with unique scores."""
        logging.info(f"[Agent 3] Stage 2: Batch LLM evaluating functions in top {len(top_files)} files...")

        # Step 1: Collect all functions from top files
        all_file_functions = []
        for file_path in top_files:
            # Find functions in this file from subgraph units
            file_functions = []
            for unit in subgraph_units:
                for func in unit.connected_functions:
                    if func.file_path == file_path:
                        file_functions.append(func)
            all_file_functions.extend(file_functions)

        # Step 2: Extend candidate nodes by one hop (parents and children)
        extended_function_ids = set()
        for func in all_file_functions:
            extended_function_ids.add(func.id)

            # Add one hop callers (parents)
            if self.kg.call_graph.has_node(func.id):
                for caller_id in self.kg.call_graph.predecessors(func.id):
                    if caller_id in self.kg.nodes:
                        extended_function_ids.add(caller_id)

                # Add one hop callees (children)
                for callee_id in self.kg.call_graph.successors(func.id):
                    if callee_id in self.kg.nodes:
                        extended_function_ids.add(callee_id)

        # Step 3: Get all extended functions
        extended_functions = [self.kg.nodes[func_id] for func_id in extended_function_ids if func_id in self.kg.nodes]

        # Check if functions have already been evaluated
        unevaluated_func_ids = extended_function_ids - self.evaluated_function_ids

        if not unevaluated_func_ids:
            logging.info(f"[Agent 3] All extended functions already evaluated, skipping.")
            return []

        unevaluated_functions = [func for func in extended_functions if func.id in unevaluated_func_ids]

        logging.info(f"[Agent 3] Extended from {len(all_file_functions)} to {len(extended_functions)} functions, evaluating {len(unevaluated_functions)} unevaluated functions...")

        system_prompt = f"""You are a debugging expert for LLM applications.
Assess which functions are most likely to contain the ROOT CAUSE of a described defect.

You will receive:
1. Defect details (explanation, consequences, reproduction steps)
2. Function details with complete candidate function code and global dependencies
3. Defect analysis hypothesis

Analysis process (ordered by impact):
1. Function-Defect Match: Does function logic directly implement defect behavior?
2. LLM Pattern Match: Do function's LLM features align with defect characteristics?
3. Dependency Support: Do global function dependencies suggest involvement?

Score guide:
- 0.8-1.0: Direct implementation of defect
- 0.6-0.8: Clear part of causal chain
- 0.4-0.6: Secondary contributor
- 0.2-0.4: Peripheral involvement
- 0.0-0.2: Little/no connection

IMPORTANT: You MUST assign DISTINCT scores to each function - no two functions can have the same score!

ONLY use the `evaluate_functions_batch` function call to output results."""

        # Step 4: Build comprehensive global function dependencies using call relationships
        function_paths = sorted([func.id for func in unevaluated_functions])
        function_to_index = {func_id: i for i, func_id in enumerate(function_paths)}

        comprehensive_dependencies = {}
        for func in unevaluated_functions:
            func_id = func.id
            func_index = function_to_index[func_id]
            dependencies = set()

            if self.kg.call_graph.has_node(func_id):
                # Check callees (functions this function calls)
                for callee_id in self.kg.call_graph.successors(func_id):
                    if callee_id in function_to_index:
                        dependencies.add(function_to_index[callee_id])

                # Check callers (functions that call this function)
                for caller_id in self.kg.call_graph.predecessors(func_id):
                    if caller_id in function_to_index:
                        dependencies.add(function_to_index[caller_id])

            if dependencies:
                comprehensive_dependencies[func_index] = sorted(list(dependencies))

        # Step 5: Prepare detailed function information
        functions_info = []
        for i, func in enumerate(unevaluated_functions):
            func_index = function_to_index[func.id]
            dependencies = comprehensive_dependencies.get(func_index, [])

            # Get dependency function names
            dep_functions = [function_paths[dep_idx] for dep_idx in dependencies if dep_idx < len(function_paths)]

            # Collect LLM roles and configs
            llm_roles = set(func.static_analysis_hints.get("potential_roles", []))
            llm_configs = set(func.static_analysis_hints.get("potential_configs", []))

            # Limit code snippet to first 5 lines
            code_lines = func.code_snippet.split("\n")
            limited_code = "\n".join(code_lines[:5])
            if len(code_lines) > 5:
                limited_code += "\n..."

            func_info = f"""Function {func_index}: 
- Function ID: {func.id}
- File: {func.file_path}
- Signature: {func.signature}
- Imports: {', '.join(func.imports[:5]) if func.imports else 'None'}
- Class: {func.containing_class_id.split(':')[-1] if func.containing_class_id else 'None'}
- LLM Roles: {', '.join(sorted(llm_roles)) if llm_roles else 'None'}
- LLM Configs: {', '.join(sorted(llm_configs)) if llm_configs else 'None'}
- Complete Function Code: {limited_code}"""

            functions_info.append(func_info)

        # sort functions_info by function_index
        functions_info.sort(key=lambda x: int(re.search(r"Function (\d+)", x).group(1)))

        # Build global dependency overview in JSON format
        dependency_overview = {"function_index_mapping": function_to_index, "global_function_dependencies": comprehensive_dependencies, "dependency_summary": f"Total functions: {len(function_paths)}, Functions with dependencies: {len(comprehensive_dependencies)}"}

        user_message = f"""Defect:
- Explanation: {defect_explanation}
- Consequences: {defect_consequences}
- Reproduction Steps: {defect_tests}

Defect Analysis Hypothesis:
- Relevant LLM Roles: {self.defect_summary.relevant_llm_roles}
- Relevant LLM Configs: {self.defect_summary.relevant_configs}
- Potential Keywords: {self.defect_summary.code_keywords}

Global Function Dependencies:
{json.dumps(dependency_overview, indent=2)}

--- Functions to Evaluate ---
{chr(10).join(functions_info)}
---

Evaluate each function using the `evaluate_functions_batch` function tool."""

        tools = [{"type": "function", "function": {"name": "evaluate_functions_batch", "description": "Evaluates functions across multiple files with unique scores", "parameters": {"type": "object", "properties": {"results": {"type": "array", "items": {"type": "object", "properties": {"function_id": {"type": "string", "description": "ID of the function being evaluated"}, "score": {"type": "number", "description": "Unique confidence score (0.0-1.0)"}}, "required": ["function_id", "score"]}}}, "required": ["results"]}}}]

        # truncate user message to 32768 tokens
        user_message = limit_user_message(user_message=user_message, system_prompt=system_prompt, max_tokens=32768)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

        all_results = []
        try:
            response = get_llm_res(messages=messages, base_url=self.args.base_url, api_key=self.args.api_key, model=self.args.model, temperature=0.0, tools=tools, tool_choice={"type": "function", "function": {"name": "evaluate_functions_batch"}}, output_dir=self.args.output_dir)

            tool_call = response.choices[0].message.tool_calls[0]
            evaluation_results = json.loads(tool_call.function.arguments)

            # Sort results by LLM score (descending)
            llm_results = evaluation_results.get("results", [])
            llm_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            for i, result in enumerate(llm_results):
                func_id = result.get("function_id")
                score = result.get("score", 0.0)
                if func_id:
                    reason = f"Batch function evaluation with extended dependencies"
                    level = i + 1  # Rank based on position after sorting
                    all_results.append((func_id, score, reason, level))

                    # Mark this function as evaluated
                    self.evaluated_function_ids.add(func_id)

                    logging.info(f"[Agent 3] Function evaluation: {func_id}, Level={level}, Score={score:.2f}")

        except Exception as e:
            logging.error(f"[Agent 3] Error during batch function evaluation: {e}", exc_info=True)
            # Fallback: assign default scores
            for i, func in enumerate(unevaluated_functions):
                score = max(0.01, 0.9 - (i * 0.01))  # Descending scores
                all_results.append((func.id, score, f"Error in batch evaluation", i + 1))
                self.evaluated_function_ids.add(func.id)

        return all_results

    def find_best_matching_node_simple(self, func_id: str) -> Tuple[Optional[str], Optional[object]]:
        """Simplified fuzzy matching"""
        # exact match
        if func_id in self.kg.nodes:
            return func_id, self.kg.nodes[func_id]

        # common variants
        variants = [func_id.rstrip(":"), func_id + ":", func_id.strip()]

        for variant in variants:
            if variant in self.kg.nodes:
                return variant, self.kg.nodes[variant]

        # find the most similar match
        best_match = None
        best_score = 0

        for node_id in self.kg.nodes.keys():
            # calculate simple similarity
            similarity = difflib.SequenceMatcher(None, func_id, node_id).ratio()
            if similarity > best_score and similarity > 0.8:  # threshold can be adjusted
                best_score = similarity
                best_match = node_id

        if best_match:
            return best_match, self.kg.nodes[best_match]

        return None, None

    def perform_call_relationship_subgraph_analysis_and_two_stage_localization(self, all_candidates: List[Tuple[str, float, str]], defect_explanation: str, defect_consequences: str, defect_tests: str, max_candidates_per_file: int = 3, max_candidates_for_context: int = 50) -> List[SearchResultEntry]:
        """Orchestrates Agent 3: Call relationship subgraph aggregation, hierarchical evaluation, and two-stage localization."""
        logging.info("[Agent 3] Starting call relationship subgraph analysis and two-stage localization.")

        # Step 1: Build call relationship subgraphs
        subgraph_units = self._build_call_relationship_subgraphs(all_candidates, max_candidates_per_file, max_candidates_for_context)

        # Save subgraph units for inspection
        subgraph_units_path = Path(self.args.output_dir) / "agent3_call_relationship_subgraphs.json"
        with open(subgraph_units_path, "w", encoding="utf-8") as f:
            json.dump([unit.to_dict() for unit in subgraph_units], f, indent=2, ensure_ascii=False)

        # Step 2: Stage 1 - Horizontal file importance evaluation across subgraphs
        file_scores = self._llm_evaluate_file_importance_batch(subgraph_units, defect_explanation, defect_consequences, defect_tests)

        # Sort files by importance and select top files
        file_scores.sort(key=lambda x: x[1], reverse=True)
        top_files = [file_path for file_path, score in file_scores[: self.args.top_files_count]]

        logging.info(f"[Agent 3] Top {len(top_files)} important files: {top_files}")
        all_evaluated_entries = []
        # =====
        # Step 3: Stage 2 - Vertical function evaluation within top files
        function_results = self._llm_evaluate_functions_in_top_files_batch(top_files, subgraph_units, defect_explanation, defect_consequences, defect_tests)

        function_results = [(self.find_best_matching_node_simple(func_id)[0] or func_id, score, reason, level) for func_id, score, reason, level in function_results]

        # Convert results to SearchResultEntry objects
        all_evaluated_entries = []
        file_score_map = {file_path: score for file_path, score in file_scores}

        logging.info(f"[Agent 3] Converting {len(function_results)} function results to SearchResultEntry objects...")

        for func_id, score, reason, level in function_results:
            logging.info(f"[Agent 3] Processing function result: {func_id}, score={score}")

            if func_id in self.kg.nodes:
                func_node = self.kg.nodes[func_id]
                file_importance_score = file_score_map.get(func_node.file_path, 0.0)

                # Find the corresponding subgraph unit
                subgraph_unit = None
                for unit in subgraph_units:
                    if any(func.id == func_id for func in unit.connected_functions):
                        subgraph_unit = unit
                        break

                # Build context with call relationships and context nodes
                context_snippets = {"subgraph_context": {"subgraph_id": subgraph_unit.subgraph_id if subgraph_unit else "", "call_relationships": subgraph_unit.call_relationships if subgraph_unit else [], "involved_files": list(subgraph_unit.involved_files) if subgraph_unit else [], "llm_features_summary": subgraph_unit.llm_features_summary if subgraph_unit else {}}}

                # Add context nodes
                context_nodes = self._get_context_nodes(func_id, self.args.context_hops)
                context_snippets.update(context_nodes)

                entry = SearchResultEntry(target_function_node=func_node, target_code_snippet=func_node.code_snippet, contextual_snippets=context_snippets, file_importance_score=file_importance_score, llm_validation_score=score, llm_validation_reason=reason, subgraph_id=subgraph_unit.subgraph_id if subgraph_unit else None, intra_file_rank=level)
                all_evaluated_entries.append(entry)
                logging.info(f"[Agent 3] Successfully created SearchResultEntry for {func_id}")
            else:
                logging.warning(f"[Agent 3] Function {func_id} not found in knowledge graph nodes, skipping...")

        logging.info(f"[Agent 3] Created {len(all_evaluated_entries)} SearchResultEntry objects from {len(function_results)} function results")

        # Sort by combined score (function score)
        all_evaluated_entries.sort(key=lambda x: (x.llm_validation_score or 0.0), reverse=True)

        # Save detailed results
        detailed_results_path = Path(self.args.output_dir) / "agent3_two_stage_detailed_results.json"
        with open(detailed_results_path, "w", encoding="utf-8") as f:
            json.dump([entry.to_dict() for entry in all_evaluated_entries], f, indent=2, ensure_ascii=False)

        # =======
        # Save file importance scores
        file_importance_path = Path(self.args.output_dir) / "agent3_file_importance_scores.json"
        with open(file_importance_path, "w", encoding="utf-8") as f:
            json.dump([{"file_path": fp, "importance_score": score} for fp, score in file_scores], f, indent=2, ensure_ascii=False)

        # logging.info(f"[Agent 3] Completed call relationship subgraph analysis and two-stage localization. Generated {len(all_evaluated_entries)} ranked results.")
        return all_evaluated_entries


# --- Main execution logic ---
def main():
    parser = argparse.ArgumentParser(description="Multi-Agent LLM-Specific Defect Localization with Call Relationship Subgraph Aggregation")

    # Core inputs
    parser.add_argument("--repo_dir", default=f"{PROJECT_PATH}/datasets/llmdd/0__0__Shaunwei__RealChar", help="Repository path")
    parser.add_argument("--defect_explanation", default="The LLM's answers sometimes contradict the truth or the content of the uploaded file", help="Defect description")
    parser.add_argument("--defect_consequences", default="incorrectness", help="Defect consequences")
    parser.add_argument("--defect_tests", default="1.In the RealChar UI, select a character to converse with.\n2.Ask ambiguous questions.", help="Trigger test steps")
    parser.add_argument("--output_dir", default="results/localization", help="Output directory")

    # LLM Configuration
    parser.add_argument("--base_url", default="http://33.33.33.123:8794/v1", help="LLM API base URL")
    parser.add_argument("--api_key", default="token-123", help="LLM API key")
    parser.add_argument("--model", default="qwen2.5-14b-instruct", help="LLM model")
    parser.add_argument("--embedding_url", default="http://localhost:11434/v1", help="Embedding service URL")
    parser.add_argument("--embedding_key", default="ollama", help="Embedding service API key")
    parser.add_argument("--embedding_model", default="bge-m3:567m-fp16", help="Embedding model to use")

    # Framework Tunables
    parser.add_argument("--static_threshold", type=float, default=0.01, help="Threshold for static feature matching")
    parser.add_argument("--embedding_threshold", type=float, default=0.3, help="Threshold for embedding similarity")
    parser.add_argument("--max_candidates_per_file", type=int, default=1000, help="Max candidates per file for subgraph analysis")
    parser.add_argument("--max_candidates_for_context", type=int, default=1000, help="Max total candidates for context analysis")
    parser.add_argument("--context_hops", type=int, default=1, help="Number of hops for context node extraction")
    parser.add_argument("--top_files_count", type=int, default=3, help="Number of top files to analyze in detail")
    parser.add_argument("--force_rebuild_kg", type=int, choices=[0, 1], default=0, help="Force rebuild of the Code Knowledge Graph")

    args = parser.parse_args()
    args.repo_dir = Path(args.repo_dir)
    args.output_dir = Path(args.output_dir)

    # --- Setup ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(os.path.join(args.output_dir, "running.log"))
    logging.info(f"Starting multi-agent defect localization run. Arguments: {args}")
    stage_times = {}

    # --- Agent 1: Code Knowledge Graph Constructor ---
    agent1_start = datetime.now()
    repo_dir_name = os.path.basename(args.repo_dir)
    if "__" in repo_dir_name:
        repo_name = "__".join(repo_dir_name.split("__")[2:])

    parent_dir = os.path.dirname(args.output_dir)
    code_kg_dir = os.path.join(parent_dir, "_code_kgs", repo_name)
    output_code_kg = os.path.join(code_kg_dir, "code_kg.json")
    os.makedirs(code_kg_dir, exist_ok=True)

    code_kg: Optional[CodeKnowledgeGraph] = None
    if not args.force_rebuild_kg and Path(output_code_kg).exists():
        logging.info(f"Loading Code Knowledge Graph from cache: {output_code_kg}")
        with open(output_code_kg, "r", encoding="utf-8") as f:
            data = json.load(f)
        code_kg = CodeKnowledgeGraph.from_dict(data)
        logging.info(f"Successfully loaded KG with {len(code_kg.nodes)} nodes and {len(code_kg.call_graph.edges)} call edges.")

    if code_kg is None:
        logging.info("Building new Code Knowledge Graph with static analysis and embeddings.")
        kg_constructor = CodeKGConstructor(str(args.repo_dir), args)
        code_kg = kg_constructor.build_kg_with_static_analysis()
        with open(output_code_kg, "w", encoding="utf-8") as f:
            json.dump(code_kg.to_dict(), f, indent=2, ensure_ascii=False)
        logging.info(f"Saved KG to cache: {output_code_kg}")

    agent1_end = datetime.now()
    stage_times["agent1_seconds"] = (agent1_end - agent1_start).total_seconds()

    # --- Agent 2: Defect Analyzer ---
    agent2_start = datetime.now()

    defect_analyzer = DefectAnalyzer(code_kg, args)
    all_candidates, defect_summary = defect_analyzer.interpret_defect_and_perform_full_matching_recall(args.defect_explanation, args.defect_consequences, args.defect_tests, args.static_threshold, args.embedding_threshold)

    agent2_end = datetime.now()
    stage_times["agent2_seconds"] = (agent2_end - agent2_start).total_seconds()

    # Save Agent 2 intermediate results
    defect_summary_path = args.output_dir / "1_defect_analysis_summary.json"
    with open(defect_summary_path, "w", encoding="utf-8") as f:
        json.dump(defect_summary.to_dict(), f, indent=2, ensure_ascii=False)

    all_candidates_summary = []
    for func_id, score, reason in all_candidates:
        node = code_kg.nodes.get(func_id)
        all_candidates_summary.append({"rank": len(all_candidates_summary) + 1, "function_id": func_id, "file_path": node.file_path if node else "N/A", "name": node.name if node else "N/A", "score": score, "reason": reason, "static_hints": node.static_analysis_hints if node else {}})

    all_candidates_path = args.output_dir / "2_all_candidates.json"
    with open(all_candidates_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates_summary, f, indent=2, ensure_ascii=False)

    # --- Agent 3: Validator ---
    agent3_start = datetime.now()

    validator = Validator(code_kg, defect_summary, args)
    final_results_entries = validator.perform_call_relationship_subgraph_analysis_and_two_stage_localization(all_candidates, args.defect_explanation, args.defect_consequences, args.defect_tests, args.max_candidates_per_file, args.max_candidates_for_context)

    agent3_end = datetime.now()
    stage_times["agent3_seconds"] = (agent3_end - agent3_start).total_seconds()

    # Save timing
    timing_path = args.output_dir / "execution_timing.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(stage_times, f, indent=2, ensure_ascii=False)

    # --- Result Formatting and Saving ---
    final_report = []
    for i, entry in enumerate(final_results_entries):
        node = entry.target_function_node
        final_report.append({"rank": i + 1, "function_id": node.id, "file_path": node.file_path, "function_name": node.name, "start_line": node.start_line, "end_line": node.end_line, "file_importance_score": entry.file_importance_score, "final_llm_score": entry.llm_validation_score, "final_llm_reason": entry.llm_validation_reason, "subgraph_id": entry.subgraph_id, "intra_file_rank": entry.intra_file_rank, "initial_static_score": entry.initial_match_score, "initial_static_reason": entry.initial_match_reason, "embedding_similarity_score": entry.embedding_similarity_score, "static_analysis_hints": node.static_analysis_hints, "code_snippet_preview": node.code_snippet.splitlines()[:15]})

    final_results_json_path = args.output_dir / "defect_results.json"
    with open(final_results_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved final Top-{len(final_report)} defect localization results to {final_results_json_path}")

    if not final_report:
        logging.warning("No defect locations identified after all agents.")
    else:
        logging.info("Multi-agent defect localization process completed successfully.")
        for item in final_report[:10]:  # Show top 10
            logging.info(f"  Rank {item['rank']}: {item['function_id']} (File Score: {item['file_importance_score']:.2f}, Function Score: {item['final_llm_score']:.2f}, Subgraph: {item['subgraph_id']})")


if __name__ == "__main__":
    main()
