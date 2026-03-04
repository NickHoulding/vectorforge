---
name: python-docs
description: Enforce consistent Google-style docstrings and minimal inline comments for all Python classes and functions in this repository.
compatibility: opencode
metadata:
  domain: documentation
  language: python
---

## Name & Trigger

**Skill:** `python-docs`
Triggered when a user asks to audit, write, fix, or enforce Python documentation — including module docstrings, function/class docstrings, or inline comments.

## The Goal

Ensure every Python file in the repository has consistent, high-quality documentation: a module-level docstring describing the file, Google-style docstrings on all functions and classes, and inline comments used sparingly and purposefully.

## The Process

1. Identify all Python files in scope (the entire repo unless the user specifies otherwise).
2. For each file, check for a module-level docstring at the top (before imports). Add or fix it if missing or non-compliant. See `reference/module-docstrings.md`.
3. For each function and method, check for a Google-style docstring. Add or fix it if missing or non-compliant. See `reference/function-docstrings.md`.
4. For each class, check for a class-level docstring. Add or fix it if missing or non-compliant.
5. Review inline comments. Remove noisy or redundant ones; ensure any that remain explain genuinely non-obvious logic.
6. Apply all changes, then confirm what was updated and why.

## Rules

- Every Python file must have a module-level docstring before any imports.
- Module docstrings must be 1–4 lines. No essays.
- Every function and method must have a Google-style docstring.
- Every class must have a docstring describing its purpose and key attributes/behaviour.
- Docstrings always use triple double quotes (`"""`).
- Google-style sections (`Args:`, `Returns:`, `Raises:`, `Example:`) are included only when relevant — never left empty.
- Inline comments are only acceptable for genuinely complex or esoteric logic that cannot be made clear by renaming or refactoring.
- Do not add comments that restate what the code already makes obvious.

## Progressive Updates

When a user identifies a documentation pattern that produces a bad outcome, or defines a clear preference for how something should be documented, **update the Rules section above** to capture that constraint permanently. This keeps the skill self-improving over time.
