# Progressive Loading Protocol
1. **Addressing**: Always start from the root `ARCHITECTURE.md` to determine the Target Domain.
2. **Unpacking**: Enter the target directory and ONLY read the sibling `MODULE_DOC.md` to acquire I/O contracts. Parallel scanning across different modules is FORBIDDEN.
3. **Locking**: Open the specific `*.py` file. If the logic contains bottlenecks (IO/VRAM/GIL), review the Header Docstring first.
4. **Constraint**: DO NOT dump implementation details from multiple directories into the Prompt at once.
