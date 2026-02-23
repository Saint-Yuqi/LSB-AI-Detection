---
description: "Isomorphic Doc-as-Code Enforcement"
globs: ["src/**/*.py", "scripts/**/*.py"]
alwaysApply: true
---
# Fractal Sync Rules

- **Rule 1**: Only require in-file Header Docstring updates when public API, I/O schema, failure modes, or performance invariants change. Pure internal refactors should NOT force docstring edits.
- **Rule 2**: Class/Func Signature modified -> MUST evaluate impact on sibling MODULE_DOC.md.
- **Rule 3**: NEVER document code implementation in Pull Requests. Docs live in code or MODULE_DOC.md.
