# LLM-Specific Defect Localization Details

## Overview

The framework consists of three specialized agents that work sequentially to identify and localize defects in LLM-integrated applications:

1. **CodeKG Constructor**: Code Knowledge Graph Construction
2. **Defect Analyzer**: Defect Analysis and Candidate Retrieval  
3. **Validator**: Hierarchical Validation and Localization

---

## CodeKG Constructor: Code Knowledge Graph Construction

### Purpose
Parses source code to extract structural elements and identify LLM interaction features, creating a `CodeKnowledgeGraph` with function nodes enriched with static analysis hints and semantic embeddings.

### 1. Code Parsing and Node Extraction

#### FunctionNode Data Structure
Each function generates a `FunctionNode` containing:

- **`id`**: Unique identifier
  - Functions: `file_path.py::function_name`
  - Methods: `file_path.py:ClassName:method_name`
- **`name`**: Function or method name
- **`file_path`**: Relative file path
- **`start_line`, `end_line`**: Function scope line numbers
- **`signature`**: Simplified signature (e.g., `def my_func(arg1, *args)`)
- **`code_snippet`**: Complete function source code
- **`docstring`**: Function docstring (if present)
- **`imports`**: List of imported modules within the file

### 2. Static Analysis for LLM Interaction Features

- Mapping of LLM-specific Defects to LLM interaction Roles and Configurations: https://anonymous.4open.science/r/LLMSDL/paper/ASE_Rebuttal/DefectMapping.md

- Patterns for LLM Role and Configuration Extraction: https://anonymous.4open.science/r/LLMSDL/paper/ASE_Rebuttal/ExtractionPatterns.md


### 3. Embedding Generation and Call Graph Construction

#### Embedding Generation
- **Input Text**: Concatenation of function signature + docstring + full code snippet
- **Model**: Sentence-transformer model
- **Purpose**: Captures semantic purpose and implementation details

#### Call Graph Construction
- **Structure**: Directed graph with function IDs as nodes
- **Edge Creation**: Function A → Function B if A's code contains call to B
- **Resolution Method**: AST analysis + import map for cross-file/class tracing

---

## Defect Analyzer: Defect Analysis and Candidate Retrieval

### Purpose
Interprets natural language defect reports and retrieves candidate functions likely to contain the defect.

### 1. LLM-based Defect Interpretation

#### System Prompt
```
You are an expert at localizing LLM-specific software defects.
Analyze the defect description and repository overview to identify code location clues.

Think about:
1. Interaction boundary issues (API calls, data mapping)
2. Prompt engineering flaws (unclear prompts, context issues)
3. Input/Output format problems (validation, formats)
4. Context management defects (dialogue history, context windows)
5. Tool usage errors (function calling issues, parsing)
6. Configuration errors (API keys, model settings)

Output ONLY a JSON via the `extract_defect_analysis_summary` function call with:
- "relevant_llm_roles": List of involved roles from: ['API_Caller', 'Prompt_Builder', 'Response_Parser', 'Context_Retriever', 'History_Manager', 'Embedding_Generator', 'Knowledge_Updater', 'Input_Validator', 'Output_Processor', 'Error_Handler', 'Data_Sanitizer']
- "relevant_configs": List of related LLM configuration items (e.g., API keys, model names)
- "potential_locations": List of *EXACTLY 5* likely bug locations in KG-compatible format:
  - For functions: "file_path.py::function_name"
  - For methods: "file_path.py:ClassName:method_name"
  Example: ["src/utils.py::process_data", "src/api.py:ApiClient:send_request"]
- "code_keywords": List of keywords likely near the bug (e.g., "api_key", "response.choices[0]")
```

#### User Message
```
Defect:
{explanation}

Consequences:
{consequences}

Reproduction steps:
{tests}

Repository structure:
{json.dumps(repo_overview, indent=2)}

ONLY call function `extract_defect_analysis_summary` to output your results.
```

#### Context Management
- **Function**: `limit_user_message`
- **Reserved Tokens**: 2000 (for response + tool calls)
- **Safety Margin**: 500 tokens
- **Approximation**: 1 token ≈ 3 characters
- **Truncation**: Appends `...[content truncated]` if exceeded

#### Output Tool Schema
```json
{
    "type": "function",
    "function": {
        "name": "extract_defect_analysis_summary",
        "description": "Extracts a structured summary of the defect analysis to help locate the bug.",
        "parameters": {
            "type": "object",
            "properties": {
                "relevant_llm_roles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of most relevant LLM roles."
                },
                "relevant_configs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of relevant LLM configuration items."
                },
                "potential_locations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of potential locations in format 'file_path.py::function_name' or 'file_path.py:ClassName:method_name'"
                },
                "code_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific code keywords or short patterns related to the defect."
                }
            },
            "required": ["relevant_llm_roles", "relevant_configs", "potential_locations", "code_keywords"]
        }
    }
}
```

### 2. Full Matching Recall with Weighted Scoring

#### Phase 1: Static Match Score

**Scoring Components**:

1. **Role Match (Weight: 2.5)**
   - **Formula**: `2.5 × overlap_count`
   - **Comparison**: Function's `potential_roles` ∩ Defect's `relevant_llm_roles`

2. **Configuration Match (Weight: 1.5)**
   - **Formula**: `1.5 × overlap_count`
   - **Comparison**: Function's `potential_configs` ∩ Defect's `relevant_configs`

3. **Location Match (Weight: 1.5)**
   - **Formula**: `1.5 × location_score`
   - **Condition**: Function ID matches defect's `potential_locations`
   - **Location Score**: Sum of partial matches (file path + class name + function name)

4. **Keyword Match (Weight: 0.75)**
   - **Formula**: `0.75 × keyword_count`
   - **Search Target**: Function's code snippet and docstring
   - **Keywords**: From defect's `code_keywords`

#### Phase 2: Embedding Similarity Score

**Configuration**:
- **Input**: Combined defect explanation + consequences + tests
- **Threshold**: `embedding_threshold = 0.3`
- **Method**: Cosine similarity with function embeddings
- **Purpose**: Supplement functions below minimum static score threshold

---

## Validator: Hierarchical Validation and Localization

### Purpose
Performs context-aware validation using LLM to pinpoint the most likely defective function from Defect Analyzer's candidates.

### 1. Call Relationship Subgraph Aggregation

**Structure**: `CallRelationSubgraphAnalysisUnit`
**Contents**:
- Grouped candidate functions
- Direct call relationships
- Involved files
- Collective LLM feature summary

### 2. Stage 1: File-Level Importance Evaluation

#### System Prompt
```
You are a debugging expert for LLM applications.
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

ONLY use the `evaluate_file_importance` function call to output results.
```

#### User Message
```
Defect:
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

Evaluate each file's importance using the `evaluate_file_importance` function tool.
```

#### Context Management
- **Function**: `limit_user_message` (same as Defect Analyzer)

#### Output Tool Schema
```json
{
    "type": "function",
    "function": {
        "name": "evaluate_file_importance",
        "description": "Evaluates file importance for defect localization with unique scores",
        "parameters": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "score": {"type": "number", "description": "Unique importance score (0.0-1.0)"}
                        },
                        "required": ["file_path", "score"]
                    }
                }
            },
            "required": ["results"]
        }
    }
}
```

### 3. Stage 2: Function-Level Pinpointing

#### Candidate Expansion
- **Method**: One-hop expansion in call graph
- **Includes**: Direct callers and callees of initial candidates
- **Purpose**: Capture closely related functions that might contain root cause

#### System Prompt
```
You are a debugging expert for LLM applications.
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

ONLY use the `evaluate_functions_batch` function call to output results.
```

#### User Message
```
Defect:
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

Evaluate each function using the `evaluate_functions_batch` function tool.
```

#### Context Management
- **Function**: `limit_user_message` (same as previous agents)

#### Output Tool Schema
```json
{
    "type": "function",
    "function": {
        "name": "evaluate_functions_batch",
        "description": "Evaluates functions across multiple files with unique scores",
        "parameters": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "function_id": {"type": "string"},
                            "score": {"type": "number", "description": "Unique confidence score (0.0-1.0)"}
                        },
                        "required": ["function_id", "score"]
                    }
                }
            },
            "required": ["results"]
        }
    }
}
```

## Final Output

**Structure**: List of `SearchResultEntry` objects
**Ranking**: By `llm_validation_score` from Stage 2
**Contents**:
- Complete `FunctionNode`
- Function source code
- Scores from all analysis stages
- Contextual validation information