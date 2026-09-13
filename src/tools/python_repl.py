"""
Python REPL Tool: Secure execution environment for running generated quantitative cash flow scripts.
"""

import re
import traceback
import json
from typing import Dict, Any, Tuple, List
from langchain_experimental.utilities import PythonREPL

# Dangerous modules that the LLM-generated code must NOT import (per rules.md §4)
BLOCKED_IMPORTS = ["os", "sys", "subprocess", "shutil", "socket", "http", "urllib", "requests"]


def _check_for_dangerous_imports(code: str) -> str | None:
    """
    Scans the generated code for blocked imports.
    Returns an error message if a dangerous import is found, else None.
    """
    for module in BLOCKED_IMPORTS:
        # Match: import os | import os, sys | from os import ... | from os.path import ...
        pattern = rf'(?:^|\n)\s*(?:import\s+{module}(?:\s|,|$)|from\s+{module}[\s.])'
        if re.search(pattern, code):
            return (
                f"SECURITY VIOLATION: Your code imports '{module}', which is blocked. "
                f"Blocked modules: {BLOCKED_IMPORTS}. "
                f"Remove the import and rewrite without OS-level access."
            )
            
    if re.search(r'(?:^|[\s(])(?:exit|quit)\s*\(', code):
        return "SECURITY VIOLATION: Calls to exit() or quit() are not allowed. Print the JSON and let the script finish naturally."
        
    return None


def _extract_last_json(stdout: str) -> Dict[str, Any] | None:
    """
    Extracts the last valid JSON object from stdout.
    Handles cases where the LLM prints debug output before the final JSON line.
    """
    # Try each line from the end, looking for valid JSON
    lines = stdout.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def execute_simulation_code(code: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    Executes the generated Monte Carlo Python script in a controlled REPL environment.
    The LLM is instructed to print the final output as a JSON string.

    Returns:
        success (bool): True if execution passed, False if it crashed.
        results (Dict): The parsed JSON output from the simulation.
        error_msg (str): The traceback or error message if it failed.
    """
    # 1. Security check: block dangerous imports before executing
    security_error = _check_for_dangerous_imports(code)
    if security_error:
        return False, {}, security_error

    repl = PythonREPL()

    try:
        # 2. Run the code
        stdout = repl.run(code)

        # 3. Extract the JSON payload from stdout (robust: finds last JSON line)
        result_dict = _extract_last_json(stdout)
        if result_dict is not None:
            return True, result_dict, ""
        else:
            return False, {}, f"Execution succeeded, but no valid JSON object found in stdout:\n{stdout}"

    except Exception as e:
        # Capture full traceback for the Quant LLM to debug
        error_msg = f"{type(e).__name__}: {str(e)}\n"
        error_msg += traceback.format_exc()
        return False, {}, error_msg