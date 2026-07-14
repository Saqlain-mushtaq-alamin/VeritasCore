"""Gradio demo for VeritasCore — HuggingFace Spaces deployment.

Launch locally:
    python demo/app.py

Deploy to HuggingFace Spaces:
    - Create a new Gradio Space
    - Push this file as app.py with demo/requirements.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gradio as gr

# ---------------------------------------------------------------------------
# Engine initialisation (lazy — loaded once on first request)
# ---------------------------------------------------------------------------
_engine = None


def _get_engine():
    global _engine  # noqa: PLW0603
    if _engine is None:
        from veritascore import VeritasCoreEngine
        _engine = VeritasCoreEngine()
    return _engine


# ---------------------------------------------------------------------------
# Core verification function
# ---------------------------------------------------------------------------
def verify_response(
    response_text: str,
    query: str,
    context: str,
    domain: str,
) -> tuple[str, str]:
    """Verify an LLM response and return formatted summary + claim detail."""
    if not response_text.strip():
        return "⚠️ Please enter an LLM response to verify.", ""

    engine = _get_engine()

    try:
        report = engine.verify(
            response=response_text,
            query=query.strip() if query.strip() else None,
            context=context.strip() if context.strip() else None,
            domain=domain,
        )
    except Exception as exc:  # noqa: BLE001
        return f"❌ Verification failed: {exc}", ""

    # ------------------------------------------------------------------
    # Summary panel
    # ------------------------------------------------------------------
    verdict_icon = {
        "supported": "✅",
        "contradicted": "❌",
        "unsupported": "⚠️",
    }.get(report.overall_verdict.value, "❓")

    score_pct = f"{report.overall_trust_score:.0%}"
    summary = (
        f"## {verdict_icon} Trust Score: {score_pct}\n\n"
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| **Overall Verdict** | `{report.overall_verdict.value.upper()}` |\n"
        f"| **Verification Mode** | `{report.verification_mode.value}` |\n"
        f"| **Domain Profile** | `{report.domain}` |\n"
        f"| **Claims analysed** | {len(report.claims)} |\n"
        f"| **Processing time** | {report.processing_time_ms:.0f} ms |\n"
    )

    # ------------------------------------------------------------------
    # Claims panel
    # ------------------------------------------------------------------
    verdict_icons = {"supported": "✅", "contradicted": "❌", "unsupported": "⚠️"}

    claims_md = "## Claim-Level Breakdown\n\n"
    for i, cv in enumerate(report.claims, 1):
        icon = verdict_icons.get(cv.verdict.value, "❓")
        claims_md += f"### {i}. {icon} `{cv.verdict.value.upper()}` — confidence {cv.confidence:.0%}\n\n"
        claims_md += f"> {cv.claim.text}\n\n"
        claims_md += f"**Reason:** {cv.reason}\n\n"
        if cv.evidence:
            snippet = cv.evidence[:250] + ("…" if len(cv.evidence) > 250 else "")
            claims_md += f"**Evidence:** _{snippet}_\n\n"
        claims_md += "---\n\n"

    return summary, claims_md


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
DESCRIPTION = """
**VeritasCore** decomposes any LLM response into atomic claims and verifies each one against evidence.

- 🟢 **Grounded mode**: paste a reference document in *Source Context* — fastest and most accurate
- 🌐 **Ungrounded mode**: leave *Source Context* empty — triggers web retrieval automatically
- Choose a **Domain Profile** for domain-specific thresholds (medical, legal, general)
"""

EXAMPLES = [
    [
        "The Eiffel Tower is 350 meters tall and was completed in 1889 for the World's Fair.",
        "Tell me about the Eiffel Tower",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
        "It is 330 metres (1,083 ft) tall and was constructed from 1887 to 1889 as the entrance arch "
        "for the 1889 World's Fair.",
        "general",
    ],
    [
        "Aspirin was first synthesized in 1897 by Felix Hoffmann at Bayer.",
        "Who invented aspirin?",
        "",
        "general",
    ],
    [
        "Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus "
        "in patients without contraindications.",
        "What is the first-line treatment for type 2 diabetes?",
        "Current guidelines recommend metformin as the preferred initial pharmacologic agent "
        "for type 2 diabetes management due to its efficacy, safety, and low cost.",
        "medical",
    ],
]

with gr.Blocks(
    title="🔍 VeritasCore — LLM Fact Checker",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.Markdown("# 🔍 VeritasCore — LLM Fact Checker")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            response_input = gr.Textbox(
                label="LLM Response to Verify",
                lines=6,
                placeholder="Paste the LLM-generated response here…",
                elem_id="response_input",
            )
            query_input = gr.Textbox(
                label="Original Query (optional)",
                lines=2,
                placeholder="What did you ask the LLM?",
                elem_id="query_input",
            )
            context_input = gr.Textbox(
                label="Source Context (optional — for grounded mode)",
                lines=6,
                placeholder="Paste reference documents, Wikipedia text, or source material here…",
                elem_id="context_input",
            )
            domain_input = gr.Dropdown(
                choices=["general", "medical", "legal"],
                value="general",
                label="Domain Profile",
                elem_id="domain_input",
            )
            verify_btn = gr.Button("🔍 Verify", variant="primary", elem_id="verify_btn")

        with gr.Column(scale=1):
            gr.Markdown("### Results")
            summary_output = gr.Markdown(label="Summary", elem_id="summary_output")
            claims_output = gr.Markdown(label="Claim Detail", elem_id="claims_output")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[response_input, query_input, context_input, domain_input],
        outputs=[summary_output, claims_output],
        fn=verify_response,
        cache_examples=False,
        label="Try an example",
    )

    verify_btn.click(
        fn=verify_response,
        inputs=[response_input, query_input, context_input, domain_input],
        outputs=[summary_output, claims_output],
    )

    gr.Markdown(
        "---\n"
        "**VeritasCore** · [GitHub](https://github.com/Saqlain-mushtaq-alamin/VeritasCore) · "
        "[Docs](https://github.com/Saqlain-mushtaq-alamin/VeritasCore/tree/main/docs) · "
        "MIT License"
    )


if __name__ == "__main__":
    demo.launch(share=False)
