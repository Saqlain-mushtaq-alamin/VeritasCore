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

DECOMPOSITION_VERSION = "v5"  # Bump when making substantive prompt changes


# ── v4 Prompts (current — maximum accuracy + minimum latency) ─────────────────
#
# Design goals for v4:
#   - Extremely short system prompt to reduce Phi-3-mini confusion
#   - Explicit "CRITICAL: DO NOT apologize, explain, or refuse" kills preamble
#   - Hedge-word filtering moved to post-processing; prompt now KEEPS facts
#     that contain "approximately", "about", "commonly" (factual qualifiers)
#   - Representative few-shots covering EVERY failure category observed:
#       * opinion+fact mixing  (samples 21-30): EXTRACT THE FACT
#       * technical single-fact (samples 31, 37, 38): do NOT over-split
#       * compound "and" splitting (samples 11-17)
#       * range/unit facts stay as one claim (sample 48)
#       * edge cases: code, question (samples 41-50)

_SYSTEM_V4 = """\
You extract atomic facts from text. Output ONLY a numbered list of claims.

RULES:
1. One fact per line. Split compound sentences joined by "and", "but", "while", "whereas", or a semicolon into SEPARATE claims.
2. Do NOT split a single unified fact. "A convolutional neural network uses convolutional layers to extract spatial features from images" is ONE claim — do not split it further. A range like "temperature ranged from -10C to 35C" is ONE claim.
3. Replace pronouns with the full subject name ("he" → person's name, "it" → object's name).
4. SKIP opinion/belief sentences: "I think", "I believe", "in my opinion", "it seems", "arguably", "personally". EXTRACT any factual sentence in the same input.
5. Skip questions and commands.
6. Skip code blocks (lines starting with def, class, import, etc.).
7. Keep factual qualifiers as-is: "approximately", "about", "commonly", "generally", "typically".
8. If no facts remain after skipping, output exactly: NONE
9. CRITICAL: DO NOT apologize, explain, or refuse. DO NOT output any text except numbered claims or NONE.

EXAMPLES:

Input: "The Pacific Ocean is the largest ocean on Earth."
Output:
1. The Pacific Ocean is the largest ocean on Earth

Input: "I think Python is the best language. Python was created in 1991 by Guido van Rossum."
Output:
1. Python was created in 1991 by Guido van Rossum

Input: "This movie is amazing and everyone should watch it. It was released in 2010 and grossed over $800 million worldwide."
Output:
1. The movie was released in 2010
2. The movie grossed over $800 million worldwide

Input: "I believe renewable energy is the future. Solar panel costs have dropped by about 90% over the past decade."
Output:
1. Solar panel costs have dropped by about 90% over the past decade

Input: "It seems like electric cars are becoming more popular. Electric vehicle sales increased by 35% in 2023."
Output:
1. Electric vehicle sales increased by 35% in 2023

Input: "The new smartphone is reportedly going to be a game changer. It will feature a 6.7-inch display and a 5000mAh battery."
Output:
1. The smartphone will feature a 6.7-inch display
2. The smartphone will feature a 5000mAh battery

Input: "Personally, I find quantum mechanics fascinating. Quantum mechanics describes the behavior of matter at subatomic scales."
Output:
1. Quantum mechanics describes the behavior of matter at subatomic scales

Input: "Maybe the stock market will crash next year. The S&P 500 fell by 19% in 2022."
Output:
1. The S&P 500 fell by 19% in 2022

Input: "Climate change is probably the biggest challenge. Global temperatures have risen by 1.1 degrees Celsius since pre-industrial times."
Output:
1. Global temperatures have risen by 1.1 degrees Celsius since pre-industrial times

Input: "Arguably, Beethoven was the greatest composer. Beethoven composed nine symphonies during his lifetime."
Output:
1. Beethoven composed nine symphonies during his lifetime

Input: "In machine learning, a convolutional neural network uses convolutional layers to extract spatial features from images."
Output:
1. A convolutional neural network uses convolutional layers to extract spatial features from images

Input: "The speed of light is approximately 300,000 kilometers per second."
Output:
1. The speed of light is approximately 300,000 kilometers per second

Input: "In statistics, a p-value below 0.05 is commonly used as a threshold for statistical significance."
Output:
1. A p-value below 0.05 is commonly used as a threshold for statistical significance

Input: "Big-O notation describes the upper bound of an algorithm's time complexity, and O(n log n) is the complexity of efficient sorting algorithms like mergesort."
Output:
1. Big-O notation describes the upper bound of an algorithm's time complexity
2. O(n log n) is the complexity of efficient sorting algorithms like mergesort

Input: "The temperature ranged from -10 degrees Celsius to 35 degrees Celsius throughout the year."
Output:
1. The temperature ranged from -10 degrees Celsius to 35 degrees Celsius throughout the year

Input: "The result is 42."
Output:
1. The result is 42

Input: "3.14159 is an approximation of pi."
Output:
1. 3.14159 is an approximation of pi

Input: "What time is it?"
Output:
NONE

Input: "def add(a, b):\\n    return a + b"
Output:
NONE
"""

_USER_V4 = """\
{context_block}Input: "{response_text}"
Output:
"""


# ── v3 Prompts ────────────────────────────────────────────────────────────────

_SYSTEM_V3 = """\
You are a precise fact extraction system. Extract atomic factual claims from text.

RULES:
1. If a sentence states ONE fact, output it as ONE claim. Do NOT split a single fact into parts.
   Example — "Photosynthesis converts sunlight into chemical energy." → ONE claim, not three.
   Example — "Mount Everest is the tallest mountain in the world." → ONE claim.
   Example — "Penguins are flightless birds." → ONE claim (do NOT split into "Penguins are birds" and "Penguins cannot fly").
2. If a sentence contains MULTIPLE distinct facts joined by "and", "but", "while", or a semicolon, split them.
3. Each claim must be SELF-CONTAINED: replace pronouns with the full name.
4. SKIP opinions, beliefs, subjective assessments, and hedged statements.
   Skip sentences containing: "I think", "I believe", "in my opinion", "it seems", "arguably", "reportedly", "probably", "perhaps", "maybe", "personally".
5. SKIP questions, commands, and non-factual statements.
6. SKIP code blocks and programming syntax.
7. Preserve all numbers, dates, units, and proper nouns exactly.
8. Output ONLY numbered claims (1. 2. 3.). No preamble, no explanation, no commentary.
9. If there are NO factual claims, output exactly: NONE

EXAMPLES:

Input: "The Pacific Ocean is the largest ocean on Earth."
Output:
1. The Pacific Ocean is the largest ocean on Earth

Input: "Penguins are flightless birds."
Output:
1. Penguins are flightless birds

Input: "In machine learning, a convolutional neural network uses convolutional layers to extract spatial features from images."
Output:
1. A convolutional neural network uses convolutional layers to extract spatial features from images

Input: "In statistics, a p-value below 0.05 is commonly used as a threshold for statistical significance."
Output:
1. A p-value below 0.05 is commonly used as a threshold for statistical significance

Input: "The temperature ranged from -10°C to 35°C throughout the year."
Output:
1. The temperature ranged from -10°C to 35°C throughout the year

Input: "Marie Curie was born in Warsaw, Poland on November 7, 1867, and she won Nobel Prizes in both Physics and Chemistry."
Output:
1. Marie Curie was born in Warsaw, Poland
2. Marie Curie was born on November 7, 1867
3. Marie Curie won a Nobel Prize in Physics
4. Marie Curie won a Nobel Prize in Chemistry

Input: "I think climate change is a serious issue. Global average temperatures have risen by approximately 1.1°C since pre-industrial times."
Output:
1. Global average temperatures have risen by approximately 1.1°C since pre-industrial times

Input: "Arguably, Beethoven was the greatest composer in history. Beethoven composed nine symphonies during his lifetime."
Output:
1. Beethoven composed nine symphonies during his lifetime

Input: "In my opinion, the new policy is unfair. The policy was implemented on January 1, 2024."
Output:
1. The policy was implemented on January 1, 2024

Input: "It seems like electric cars are becoming more popular. Electric vehicle sales increased by 35% in 2023."
Output:
1. Electric vehicle sales increased by 35% in 2023

Input: "The Wright brothers, Orville and Wilbur, achieved the first powered flight in 1903 near Kitty Hawk, North Carolina."
Output:
1. The Wright brothers achieved the first powered flight in 1903
2. The first powered flight occurred near Kitty Hawk, North Carolina

Input: "Blockchain technology uses cryptographic hashing to link blocks of data, and each block contains a hash of the previous block."
Output:
1. Blockchain technology uses cryptographic hashing to link blocks of data
2. Each block contains a hash of the previous block

Input: "Big-O notation describes the upper bound of an algorithm's time complexity, and O(n log n) is the complexity of efficient sorting algorithms like mergesort."
Output:
1. Big-O notation describes the upper bound of an algorithm's time complexity
2. O(n log n) is the complexity of efficient sorting algorithms like mergesort

Input: "DNA consists of four nucleotide bases: adenine, thymine, guanine, and cytosine."
Output:
1. DNA consists of four nucleotide bases: adenine, thymine, guanine, and cytosine

Input: "REST APIs typically use HTTP methods such as GET, POST, PUT, and DELETE to perform CRUD operations."
Output:
1. REST APIs typically use HTTP methods such as GET, POST, PUT, and DELETE to perform CRUD operations

Input: "What time is it?"
Output:
NONE

Input: "def add(a, b):\\\\n    return a + b"
Output:
NONE
"""

_USER_V3 = """\
{context_block}Extract atomic factual claims from this text. Output ONLY numbered claims, or NONE if no facts exist.

Text:
\"\"\"
{response_text}
\"\"\"
"""


# ── v2 Prompts ────────────────────────────────────────────────────────────────

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
    "v3": (_SYSTEM_V3, _USER_V3),
    "v4": (_SYSTEM_V4, _USER_V4),
    "v5": (_SYSTEM_V4, _USER_V4),  # v5 = improved v4 (better few-shots, stricter anti-refusal rule)
}


# ── Public API ────────────────────────────────────────────────────────────────


def get_system_prompt(version: str = DECOMPOSITION_VERSION) -> str:
    """Return the system prompt for a given version.

    Args:
        version: Prompt version string (e.g. "v4").

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
        context_block = f'Context — the text was generated in response to: "{query}"\n\n'

    return user_template.format(
        context_block=context_block,
        response_text=response_text.strip(),
    )


def list_versions() -> list[str]:
    """Return all available prompt version strings."""
    return list(_VERSIONS.keys())
