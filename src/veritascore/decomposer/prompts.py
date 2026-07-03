"""Prompt templates for LLM-based claim decomposition.

Versioned so that prompt changes are tracked explicitly.
The active version is DECOMPOSITION_VERSION; callers can pin a version
by passing it to build_decomposition_prompt().

Prompt design principles (from literature review — FActScore, FacTool):
    1. Show diverse in-context examples (not just simple cases)
    2. Instruct co-reference resolution ("Einstein", not "he")
    3. Prohibit compound claims with "and"
    4. Request numbered output for reliable parsing
    5. Keep temperature low at inference time for determinism
"""

from __future__ import annotations

# ── Version Registry ──────────────────────────────────────────────────────────

DECOMPOSITION_VERSION = "v2"   # Bump when making substantive prompt changes


# ── v2 Prompts (current) ──────────────────────────────────────────────────────

_SYSTEM_V2 = """\
You are a precise fact extraction system. Your task is to decompose a given text \
into atomic factual claims — the smallest possible units that are independently verifiable.

RULES (follow all of them strictly):
1. Each claim must be ONE single fact. Do NOT combine two facts with "and", "but", or "while".
2. Claims must be SELF-CONTAINED: resolve all pronouns and references to their full form.
   Bad:  "He was born in 1879."
   Good: "Albert Einstein was born in 1879."
3. Do NOT include opinions, beliefs, or subjective assessments.
   Skip: "I think...", "It seems...", "arguably...", "reportedly..."
4. Do NOT include questions, commands, or rhetorical statements.
5. Preserve all numbers, dates, units, and proper nouns exactly as written.
6. If a single sentence contains N distinct facts, produce N separate claims.
7. Output ONLY the numbered claims — no preamble, no explanation.

EXAMPLES:

Input: "Marie Curie was born in Warsaw, Poland on November 7, 1867. She became \
the first woman to win a Nobel Prize and the only person to win Nobel Prizes \
in two different sciences."

Output:
1. Marie Curie was born in Warsaw, Poland.
2. Marie Curie was born on November 7, 1867.
3. Marie Curie was the first woman to win a Nobel Prize.
4. Marie Curie won Nobel Prizes in two different sciences.
5. Marie Curie is the only person to win Nobel Prizes in two different sciences.

---

Input: "The Python programming language was created by Guido van Rossum and was \
first released in 1991. Python emphasizes code readability, and its syntax allows \
programmers to express concepts in fewer lines of code than C++ or Java."

Output:
1. The Python programming language was created by Guido van Rossum.
2. The Python programming language was first released in 1991.
3. Python emphasizes code readability.
4. Python syntax allows programmers to express concepts in fewer lines of code than C++.
5. Python syntax allows programmers to express concepts in fewer lines of code than Java.

---

Input: "I think climate change is a serious issue. Global average temperatures have \
risen by approximately 1.1 degrees Celsius since pre-industrial times. Scientists \
believe we need to reduce emissions by 45% by 2030."

Output:
1. Global average temperatures have risen by approximately 1.1°C since pre-industrial times.
2. Scientists believe emissions need to be reduced by 45% by 2030.
"""

_USER_V2 = """\
{context_block}Decompose the following text into atomic factual claims.

Text:
\"\"\"
{response_text}
\"\"\"

Output each claim on a separate numbered line (1. 2. 3. etc.):"""


# ── v1 Prompts (kept for reproducibility) ────────────────────────────────────

_SYSTEM_V1 = """\
You are a precise fact extraction system. Your job is to decompose text into \
atomic factual claims.

Rules:
1. Each claim must be a SINGLE, independently verifiable factual statement.
2. Do NOT include opinions, subjective assessments, or rhetorical statements.
3. Do NOT include questions or imperative statements.
4. Each claim must be SELF-CONTAINED (understandable without surrounding context).
5. Resolve all pronouns and references to their full form.
6. If a sentence contains multiple facts, split them into separate claims.
7. Preserve numerical values, dates, and proper nouns exactly.
8. Output ONLY the claims, one per line, numbered.

Example:
Input: "Albert Einstein, who was born in Germany in 1879, developed the theory \
of relativity and won the Nobel Prize in Physics in 1921."

Output:
1. Albert Einstein was born in Germany.
2. Albert Einstein was born in 1879.
3. Albert Einstein developed the theory of relativity.
4. Albert Einstein won the Nobel Prize in Physics.
5. Albert Einstein won the Nobel Prize in Physics in 1921.
"""

_USER_V1 = """\
Decompose the following text into atomic factual claims.

{context_block}
Text to decompose:
\"\"\"
{response_text}
\"\"\"

Output each claim on a separate line, numbered (1. 2. 3. etc.):"""


# ── Registry ──────────────────────────────────────────────────────────────────

_VERSIONS: dict[str, tuple[str, str]] = {
    "v1": (_SYSTEM_V1, _USER_V1),
    "v2": (_SYSTEM_V2, _USER_V2),
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_system_prompt(version: str = DECOMPOSITION_VERSION) -> str:
    """Return the system prompt for a given version.

    Args:
        version: Prompt version string (e.g. "v2").

    Returns:
        System prompt string.

    Raises:
        KeyError: If the version does not exist.
    """
    if version not in _VERSIONS:
        raise KeyError(f"Unknown prompt version '{version}'. Available: {list(_VERSIONS)}")
    return _VERSIONS[version][0]


def build_decomposition_prompt(
    response_text: str,
    query: str | None = None,
    version: str = DECOMPOSITION_VERSION,
) -> str:
    """Build the user-turn prompt for claim decomposition.

    Args:
        response_text: The LLM response to decompose.
        query: Optional original user query for context.
        version: Prompt version to use. Defaults to DECOMPOSITION_VERSION.

    Returns:
        Formatted user-turn prompt string.

    Raises:
        KeyError: If the version does not exist.

    Example:
        >>> prompt = build_decomposition_prompt(
        ...     "Paris is the capital of France.",
        ...     query="What is the capital of France?",
        ... )
    """
    if version not in _VERSIONS:
        raise KeyError(f"Unknown prompt version '{version}'. Available: {list(_VERSIONS)}")

    _, user_template = _VERSIONS[version]

    context_block = ""
    if query:
        context_block = (
            f'Context — the text was generated in response to: "{query}"\n\n'
        )

    return user_template.format(
        context_block=context_block,
        response_text=response_text.strip(),
    )


def list_versions() -> list[str]:
    """Return all available prompt version strings."""
    return list(_VERSIONS.keys())
