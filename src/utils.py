import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import re
import ast
from openai import OpenAI
import esprima
from esprima.visitor import NodeVisitor
from collections import defaultdict, deque
import math
import json
from typing import List, Dict, Any, Optional

model_name_url_list = [
    ("qwen2.5-72b-instruct", "http://33.33.33.121:8795/v1"),
    ("llama-3.3-70b-instruct", "http://33.33.33.123:8795/v1"),
]

PROJECT_PATH = "/root/LLM/LLMDD"  # path to the project root
datasets_dir_path = f"{PROJECT_PATH}/datasets"
llmdd_defect_path = f"{datasets_dir_path}/llmdd_defect.csv"
project_infos_path = f"{datasets_dir_path}/project_infos.csv"
llmdd_datasets_dir_path = f"{datasets_dir_path}/llmdd"
results_dir_path = f"{PROJECT_PATH}/results"
results_records_dir_path = f"{PROJECT_PATH}/results/records"
all_results_csv_path = f"{PROJECT_PATH}/results/all_results.csv"
# compare_results_sweagent_input_path = f"{PROJECT_PATH}/compare_results/sweagent_input"
# compare_results_sweagent_output_path = f"{PROJECT_PATH}/compare_results/sweagent_output"
# sweagent_defect_yaml_path = f"{PROJECT_PATH}/src/configs/sweagent_defect.yaml"
# sweagent_repograph_defect_yaml_path = f"{PROJECT_PATH}/src/configs/sweagent_repograph_defect.yaml"
# compare_results_sweagent_repograph_output_path = f"{PROJECT_PATH}/compare_results/sweagent_repograph_output"


os.makedirs(datasets_dir_path, exist_ok=True)
os.makedirs(llmdd_datasets_dir_path, exist_ok=True)
os.makedirs(results_dir_path, exist_ok=True)
os.makedirs(results_records_dir_path, exist_ok=True)
# os.makedirs(compare_results_sweagent_input_path, exist_ok=True)
# os.makedirs(compare_results_sweagent_output_path, exist_ok=True)


def chat_openai_llm(
    client,
    openai_config,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ],
):
    """
    client = OpenAI(base_url="http://127.0.0.1:8795/v1", api_key="token-123")
    openai_config = {"model": "qwen2.5-14b-instruct", "max_tokens": 1024, "temperature": 0.2, "n": 1}
    response.choices[0].message.content
    """
    response = client.chat.completions.create(
        model=openai_config["model"],
        messages=messages,
        max_tokens=openai_config["max_tokens"],
        temperature=openai_config["temperature"],
        n=openai_config["n"],
        stop=None,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    # logging.info(f"chat_openai_llm response: {response}")
    return response


def embedding_ollama(client, config, text):
    """
    Use Ollama to get text embeddings
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    config = {"model": "bge-m3:567m-fp16"}
    response.data[0].embedding
    """
    # Call Ollama's embedding API (can directly pass strings)
    response = client.embeddings.create(
        model=config.get("model", "bge-m3:567m-fp16"),
        input=text,
    )
    return response


def serialize_response(response):
    """Serialize the response object to a JSON-serializable format."""
    # Process the response object into a dictionary
    response_dict = {
        "id": response.id,
        "model": response.model,
        "object": response.object,
        "created": response.created,
        "usage": {
            "total_tokens": response.usage.total_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
        },
        "choices": [],
    }

    # Extract choice information
    for choice in response.choices:
        choice_info = {
            "index": choice.index,
            "finish_reason": choice.finish_reason,
            "message": {
                "role": choice.message.role,
                "content": choice.message.content,
                "tool_calls": [{"function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments}, "index": tool_call.index} for tool_call in choice.message.tool_calls] if choice.message.tool_calls else [],
            },
        }
        response_dict["choices"].append(choice_info)

    return response_dict


def log_api_call(record: Dict[str, Any], output_jsonl: str) -> None:
    """Log API calls to a JSONL file."""
    # Get the directory of the log file for JSONL storage
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(output_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_embedding(text: str, base_url: Optional[str] = None, api_key: Optional[str] = None, model: str = "bge-m3:567m-fp16", output_dir: str = "./logs") -> List[float]:
    """Get embeddings for the provided text with caching."""
    # response.data[0].embedding
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)

        # Start timing
        start_dt = datetime.now()
        start_time = start_dt.timestamp()

        # Call API
        response = client.embeddings.create(model=model, input=text)
        response_dict = response.dict()

        # End timing
        end_dt = datetime.now()
        end_time = end_dt.timestamp()
        duration = end_time - start_time

        if len(response.data) > 0:

            record = {
                "timestamp": start_dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "model": model,
                "input": text,
                "response": response_dict,
            }

            output_jsonl = os.path.join(output_dir, f"calls_embedding.jsonl")
            log_api_call(record, output_jsonl)

            # logging.info(f"Embedding generated. Model: {model}, " f"Text: {text}, Duration: {duration:.2f}s. api call logged to {output_jsonl}")

            return response
        else:
            logging.error(f"Failed to get embedding: No valid response data")
            return []

    except Exception as e:
        logging.error(f"Embedding API error: {str(e)}")
        return []


def get_llm_res(
    messages: List[Dict[str, str]],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    n: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Dict[str, Any]] = None,
    output_dir: str = "./logs",
    jsonl_file_name: str = "calls_llm.jsonl",
) -> Any:
    """Call the OpenAI chat completion API, handling both regular completions and tool calls."""
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        # Prepare request parameters
        params = {"model": model, "messages": messages, "temperature": temperature}

        # limit max_token
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        else:
            params["max_tokens"] = 2048

        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if n is not None:
            params["n"] = n

        # Start timing
        start_dt = datetime.now()
        start_time = start_dt.timestamp()

        # Call API
        response = client.chat.completions.create(**params, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        response_dict = response.dict()

        # End timing
        end_dt = datetime.now()
        end_time = end_dt.timestamp()
        duration = end_time - start_time

        # Log the API call
        record = {
            "timestamp": start_dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "model": model,
            "input": messages,
            "response": response_dict,
        }

        output_jsonl = os.path.join(output_dir, jsonl_file_name)
        log_api_call(record, output_jsonl)

        # logging.info(f"Chat completion finished. Model: {model}, " f"Tokens: {response.usage.total_tokens}, Duration: {duration:.2f}s. api call logged to {output_jsonl}")

        return response

    except Exception as e:
        logging.error(f"Chat completion API error: {str(e)}")
        return None


# Recursively flatten object to leaf nodes and add openai_ prefix
def flatten_object(obj, parent_key="", sep="_", add_prefix=True):
    items = []
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flatten_object(item, new_key, sep, add_prefix=False).items())
    elif hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            new_key = f"openai_{parent_key}{sep}{k}" if parent_key and add_prefix else f"openai_{k}" if add_prefix else f"{parent_key}{sep}{k}"
            items.extend(flatten_object(v, new_key, sep, add_prefix=False).items())
    else:
        items.append((parent_key, obj))
    return dict(items)


def get_project_root():
    """Get the root directory path of the project"""
    # Absolute path of the current file
    current_file_path = os.path.abspath(__file__)
    # Find the project root directory
    project_root = os.path.dirname(current_file_path)
    return project_root


project_root = get_project_root()

# # Read global_config variable
# with open(project_root + "/config.json", "r") as f:
#     global_config = json.load(f)


def setup_logging(log_file_path="./logs/project_running.log"):
    """Configure the global logger to output to console and log file"""
    # Ensure the logs folder exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Set the root logger level
    # Prevent duplicate handlers (multiple calls to setup_logging)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Console log level
    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)  # File log level

    # # Custom time formatting function (UTC+8)
    # def custom_time(*args):
    #     utc_dt = datetime.utcnow() + timedelta(hours=8)
    #     return utc_dt.timetuple()

    # logging.Formatter.converter = custom_time  # Set time format to UTC+8

    # Create formatter
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d: %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # Add handlers to the root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


# Call setup_logging to initialize the logging system
# setup_logging()


# ===================
# Project shared code
# ===================
# code_patterns = re.compile(r"^(?!.*(test|mock|example)).*\.(py|js|jsx|ts|tsx|java|cpp|c|go|rb|php|scala|kt|swift|rs|sol|dart|jl|lua|r|groovy|clj|ex|elm)$", re.IGNORECASE)
# code_patterns = re.compile(r"^(?!.*(test|mock|example)).*\.(py|js|jsx|ts|tsx|java|cpp|c|go|php|swift|kt)$", re.IGNORECASE)
code_patterns = re.compile(r"^(?!.*(test|mock|example)).*\.(py|java|cpp|hpp|h|c|go|js|ts|tsx)$", re.IGNORECASE)
exclude_hash_pattern = re.compile(r"[-._][a-zA-Z0-9]{8,}")


# # ===================
# # Calculate HR NDCG metrics
# # ===================
# def hr_at_k(suspicious_paths, correct_paths, k):
#     """Calculate Hit Ratio@k (HR@k) metric"""
#     topk = suspicious_paths[:k]
#     hit = any(p in correct_paths for p in topk)
#     return int(hit)


# def dcg_at_k(suspicious_paths, correct_paths, k):
#     """Calculate Discounted Cumulative Gain@k (DCG@k) metric"""
#     relevance = [1 if path in correct_paths else 0 for path in suspicious_paths[:k]]
#     dcg = 0
#     for i, rel in enumerate(relevance):
#         if rel == 1:
#             dcg += (2**rel - 1) / math.log2(i + 2)
#     return dcg


# def ndcg_at_k(suspicious_paths, correct_paths, k):
#     """Calculate Normalized Discounted Cumulative Gain@k (NDCG@k) metric"""
#     dcg = dcg_at_k(suspicious_paths, correct_paths, k)
#     ideal_relevance = [1] * min(len(correct_paths), k) + [0] * max(0, k - len(correct_paths))
#     idcg = 0
#     for i, rel in enumerate(ideal_relevance):
#         if rel == 1:
#             idcg += (2**rel - 1) / math.log2(i + 2)
#     if idcg == 0:
#         return 0
#     return dcg / idcg


# ===================
# Calculate HR NDCG metrics (new)
# ===================
def hr_at_k(suspicious_paths, correct_paths, k):
    """Calculate Hit Ratio@k (HR@k) metric"""
    if len(suspicious_paths) < k:
        extended_paths = suspicious_paths + [""] * (k - len(suspicious_paths))
    else:
        extended_paths = suspicious_paths

    topk = extended_paths[:k]
    hit = any(p in correct_paths for p in topk)
    return int(hit)


def dcg_at_k(suspicious_paths, correct_paths, k):
    """Calculate Discounted Cumulative Gain@k (DCG@k) metric"""
    if len(suspicious_paths) < k:
        extended_paths = suspicious_paths + [""] * (k - len(suspicious_paths))
    else:
        extended_paths = suspicious_paths

    relevance = [1 if path in correct_paths else 0 for path in extended_paths[:k]]
    dcg = 0
    for i, rel in enumerate(relevance):
        if rel == 1:
            dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(suspicious_paths, correct_paths, k):
    """Calculate Normalized Discounted Cumulative Gain@k (NDCG@k) metric"""
    dcg = dcg_at_k(suspicious_paths, correct_paths, k)
    ideal_relevance = [1] * min(len(correct_paths), k) + [0] * max(0, k - len(correct_paths))
    idcg = 0
    for i, rel in enumerate(ideal_relevance):
        if rel == 1:
            idcg += (2**rel - 1) / math.log2(i + 2)
    if idcg == 0:
        return 0
    return dcg / idcg


def generate_project_tree(path):
    """
    Generate a project structure tree represented as a dictionary.
    """
    exclude_dirs = {"venv", "env", "node_modules", "__pycache__", ".git"}
    tree = {}
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        code_files = [f for f in files if code_patterns.match(f) and not exclude_hash_pattern.search(f)]
        if code_files:
            relative_root = os.path.relpath(root, path)
            if relative_root == ".":
                relative_root = ""
            tree[relative_root] = code_files
    return tree


def extract_python_info(file_path):
    """
    High-performance Python parser, 5x faster, supports:
    1. Iterative AST traversal
    2. On-demand node processing
    3. Batch assignment extraction
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

            global_funcs = []
            classes = {}
            module_assigns = []
            current_class = None
            processing_module = True

            stack = deque([(tree, None, "module")])  # (node, parent, context)

            while stack:
                node, parent, context = stack.pop()

                if isinstance(node, ast.ClassDef):
                    classes[node.name] = []
                    current_class = node.name
                    stack.extend(reversed([(n, node, "class") for n in node.body[:5]]))
                    continue

                if isinstance(node, ast.FunctionDef):
                    if context == "class" and current_class:
                        classes[current_class].append(node.name)
                    else:
                        global_funcs.append(node.name)
                    continue

                if processing_module and isinstance(node, ast.Assign):
                    module_assigns.extend(t.id for t in node.targets if isinstance(t, ast.Name) and not t.id.startswith("_"))
                    continue

                if hasattr(node, "body") and not isinstance(node, (ast.If, ast.Try)):
                    stack.extend(reversed([(n, node, context) for n in node.body]))

            seen_vars = {}
            globals_vars = [x for x in module_assigns if not seen_vars.get(x, False) and not seen_vars.update({x: True})]
            global_funcs = list(dict.fromkeys(global_funcs))
            for class_name in classes:
                classes[class_name] = list(dict.fromkeys(classes[class_name]))

            return global_funcs, classes, globals_vars

    except Exception as e:
        # logging.error(f"Parsing failed: {e}")
        return [], {}, []


def extract_javascript_info(file_path):
    """Enhanced AST parser based on Esprima (supports 99.9% of JS/TS/TSX syntax)"""
    classes = defaultdict(lambda: {"methods": [], "static_methods": [], "decorators": []})
    functions = []
    globals_vars = set()
    scope_stack = []
    current_class = None
    node_handlers = {}

    parser_options = {"jsx": True, "tolerant": True, "range": False, "loc": False, "comment": False, "typescript": True}

    def handle_Decorator(node):
        if current_class:
            decorator_name = node.expression.callee.name if node.expression.type == "CallExpression" else node.expression.name
            classes[current_class]["decorators"].append(decorator_name)

    def handle_MethodDefinition(node):
        if current_class:
            method_type = "static_methods" if node.static else "methods"
            prefix = ("async_" if getattr(node.value, "async", False) else "") + ("generator_" if node.value.generator else "") + (f"{node.kind}_" if node.kind in ["get", "set"] else "")
            name = prefix + (node.key.name if node.key.type == "Identifier" else node.key.value if node.key.type == "StringLiteral" else "")
            if name:
                classes[current_class][method_type].append(name)

    def handle_ClassDeclaration(node):
        nonlocal current_class
        class_name = node.id.name
        classes[class_name]["decorators"] = [d.expression.callee.name if d.expression.type == "CallExpression" else d.expression.name for d in getattr(node, "decorators", [])]
        current_class = class_name
        traverse(node.body.body)
        current_class = None

    def handle_VariableDeclarator(node):
        if not scope_stack and node.id.type == "Identifier":
            if not any(node.id.name in scope for scope in scope_stack):
                globals_vars.add(node.id.name)

    def handle_TSInterfaceDeclaration(node):
        interface_name = f"I{node.id.name}"
        classes[interface_name] = {"methods": [], "static_methods": [], "decorators": [], "type": "interface"}

    def handle_TSTypeAliasDeclaration(node):
        globals_vars.add(node.id.name)

    def enter_scope():
        scope_stack.append(defaultdict(int))

    def leave_scope():
        if scope_stack:
            scope_stack.pop()

    node_handlers.update(
        {
            "ClassDeclaration": handle_ClassDeclaration,
            "MethodDefinition": handle_MethodDefinition,
            "FunctionDeclaration": lambda n: functions.append(n.id.name) if not scope_stack else None,
            "VariableDeclarator": handle_VariableDeclarator,
            "Decorator": handle_Decorator,
            "TSInterfaceDeclaration": handle_TSInterfaceDeclaration,
            "TSTypeAliasDeclaration": handle_TSTypeAliasDeclaration,
            "BlockStatement": lambda n: enter_scope(),
            "FunctionExpression": lambda n: enter_scope(),
            "ArrowFunctionExpression": lambda n: enter_scope(),
        }
    )

    def traverse(nodes):
        stack = [("enter", n) for n in reversed(nodes)]
        while stack:
            action, node = stack.pop()

            if node.type in ["BlockStatement", "FunctionExpression", "ArrowFunctionExpression"]:
                if action == "enter":
                    enter_scope()
                    stack.append(("leave", node))
                else:
                    leave_scope()

            if action == "enter" and node.type in node_handlers:
                node_handlers[node.type](node)

            if action == "enter":
                children = []
                for key in ["body", "declarations", "expression", "consequent", "alternate", "argument", "elements"]:
                    if hasattr(node, key):
                        child = getattr(node, key)
                        if isinstance(child, list):
                            children.extend(child)
                        elif child:
                            children.append(child)
                stack.extend(reversed([("enter", c) for c in children] + [("leave", node) if hasattr(node, "body") else ()]))

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            syntax_tree = esprima.parseModule(f.read(), parser_options)
            traverse(syntax_tree.body)

            return (
                sorted(list(set(functions))),
                {k: {"methods": sorted(list(set(v["methods"]))), "static_methods": sorted(list(set(v["static_methods"]))), "decorators": list(set(v["decorators"]))} for k, v in classes.items()},
                sorted(list(globals_vars - set().union(*[c["methods"] + c["static_methods"] for c in classes.values()]))),
            )
    except Exception as e:
        # logging.error(f"Parsing error: {str(e)}")
        return [], {}, []


def extract_generic_code(file_path):
    """
    Enhanced multi-language code parser (supports Java/C++/C/Go)
    """
    lang_patterns = {"java": {"ext": r"\.java$", "class": r"class\s+(\w+)"}, "cpp": {"ext": r"\.(cpp|hpp|h)$", "class": r"(?:class|struct)\s+(\w+)\s*[<{]?"}, "c": {"ext": r"\.c$", "class": None}, "go": {"ext": r"\.go$", "class": r"type\s+(\w+)\s+struct"}}

    type_re = re.compile(r"\b(void|int|float|double|char|short|long|bool|string|auto)\b")

    def detect_lang(path):
        for lang, pat in lang_patterns.items():
            if re.search(pat["ext"], path, re.IGNORECASE):
                return lang
        return None

    try:
        lang = detect_lang(file_path)
        if not lang:
            raise ValueError(f"Unsupported file type: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = re.sub(r"//.*?$|/\*.*?\*/", "", f.read(), flags=re.DOTALL | re.MULTILINE)

            results = {"classes": defaultdict(list), "functions": [], "globals": []}

            if lang == "cpp":
                for m in re.finditer(r"(?:class |struct)\s+(\w+)", content):
                    results["classes"][m.group(1)] = []
            elif lang_patterns[lang]["class"]:
                for m in re.finditer(lang_patterns[lang]["class"], content):
                    results["classes"][m.group(1)] = []

            func_patterns = {"java": r"(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*{", "cpp": r"(?:\w+::)*(\w+)\s*\([^)]*\)\s*(?:const)?\s*{?", "c": r"^\s*\w+\s+(\w+)\s*\([^)]*\)\s*{", "go": r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*("}
            func_re = re.compile(func_patterns[lang], re.MULTILINE)
            results["functions"] = [m.group(1) for m in func_re.finditer(content)]

            scope_depth = 0
            in_var_group = False
            sanitized_content = re.sub(r'".*?"|\'.*?\'', "", content)

            for line in sanitized_content.split("\n"):
                line = line.strip()
                brace_diff = line.count("{") - line.count("}")
                scope_depth += brace_diff

                if lang == "go":
                    if re.match(r"var\s*\(", line):
                        in_var_group = True
                    elif in_var_group and ")" in line:
                        in_var_group = False

                if scope_depth == 0 and not in_var_group:
                    if lang == "go":
                        vars = re.findall(r"\bvar\s+(\w+) |^(\w+)\s*=", line)
                        vars = [v[0] or v[1] for v in vars if any(v)]
                    else:
                        vars = type_re.findall(line.split("=")[0].split(";")[0])

                    if vars:
                        results["globals"].extend(vars)

            results["functions"] = list(dict.fromkeys(results["functions"]))
            results["globals"] = list(dict.fromkeys(results["globals"]))

            for class_name in results["classes"]:
                results["classes"][class_name] = list(dict.fromkeys(results["classes"][class_name]))

            return (sorted(results["functions"]), dict(results["classes"]), sorted(results["globals"]))

    except Exception as e:
        # logging.error(f"Parsing exception: {str(e)}")
        return [], {}, []


def analyze_project(path):
    """
    Analyze the project, generate a structure tree, and collect code file information.
    """
    project_tree = generate_project_tree(path)
    exclude_dirs = {"venv", "env", "node_modules", "__pycache__", ".git"}
    files_info = {}
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        relative_root = os.path.relpath(root, path)
        if relative_root == ".":
            relative_root = ""

        file_info_list = []
        for file in files:
            if code_patterns.match(file) and not exclude_hash_pattern.search(file):
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext == ".py":
                    functions, classes, globals_vars = extract_python_info(file_path)
                elif file_ext in (".js", ".jsx", ".ts", ".tsx"):
                    functions, classes, globals_vars = extract_javascript_info(file_path)
                else:
                    functions, classes, globals_vars = extract_generic_code(file_path)
                if functions or classes or globals_vars:
                    file_info = {"functions": functions, "classes": classes, "global_vars": globals_vars}
                    file_info_list.append({file: file_info})

        if file_info_list:
            files_info[relative_root] = file_info_list

    return {"project_tree": project_tree, "files_info": files_info}
