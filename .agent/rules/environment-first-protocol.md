---
trigger: always_on
---

Rule: The "Environment-First" Protocol

1. Diagnose Before Action: When code execution fails, you must analyze the Traceback FIRST.

IF the error is ModuleNotFoundError, ImportError, or related to missing libraries/environment:

DO NOT modify the source code to avoid the dependency.

ACTION: Check the active environment or propose installing the package (e.g., pip install pandas).

STOP: If you cannot install it, pause and request user intervention.

ONLY refactor code if the error is a SyntaxError, LogicError, or RuntimeError caused by your implementation.

2. Environment Awareness: Assume the environment might be uninitialized. Before running complex scripts, you may verify dependencies using pip list or check python --version.
3.Correct: conda run -n [env] --no-capture-output python main.py