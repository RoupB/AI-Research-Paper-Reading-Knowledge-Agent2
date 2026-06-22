# app/streamlit_app.py
from __future__ import annotations

import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.components.db_reader import get_audit_stats  # noqa: E402
from app.components.pipeline_state import get_progress, is_running, set_progress  # noqa: E402
from config import settings  # noqa: E402

st.set_page_config(
    page_title="ClaimCheck — LoRA Audit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
_KNOWN_VARIANTS = [
    "LoRA", "QLoRA", "AdaLoRA", "DoRA", "LoRA+", "VeRA",
    "DyLoRA", "LoftQ", "LoRA-FA", "GLoRA", "rsLoRA", "Flora",
]

_KNOWN_BENCHMARKS = [
    "MMLU", "HellaSwag", "WinoGrande", "ARC-Challenge",
    "TruthfulQA", "GSM8K", "HumanEval", "MBPP",
    "GLUE", "SuperGLUE", "MT-Bench", "AlpacaEval",
    "BBH", "DROP", "MATH",
]

# Node names from main.py's build_graph()
_NODE_TO_STEP: dict[str, int] = {
    "discover": 1,
    "resolve": 2,
    "extract": 3,
    "analyze": 4,
    "gap": 5,
    "contradictions": 6,
    "report": 7,
}

_STEP_MSGS = [
    "",                                             # 0 – initialising
    "Discovering papers on arXiv…",                # 1
    "Resolving GitHub repositories…",              # 2
    "Extracting benchmark claims from PDFs…",      # 3
    "Analysing released code…",                    # 4
    "Running gap analysis (claims vs code)…",      # 5
    "Mapping cross-paper contradictions…",         # 6
    "Generating audit report…",                    # 7
    "Pipeline complete!",                          # 8
]

# Approximate progress % at each step
_STEP_PCT = [0, 10, 22, 38, 54, 68, 82, 95, 100]

_STEP_LABELS = [
    "Discovery", "Repos", "Claims",
    "Code", "Gaps", "Contras.", "Report",
]


# ── Background thread ─────────────────────────────────────────────────────────
def _run_pipeline_thread(
    sid: str,
    text: str,
    variants: list[str],
    benchmarks: list[str],
    max_p: int,
    skip_repo: bool,
    min_conf: float,
    arxiv_n: int,
    temp: float,
    concurrency: int,
    discovery_rounds: int,
) -> None:
    """Runs the full pipeline in a daemon thread. Progress posted to pipeline_state."""
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        from db import init_db
        from main import build_graph
        from session.user_session import run_user_session_from_text

        # Apply user-chosen settings
        settings.pipeline_max_papers = max_p
        settings.skip_papers_without_repo = skip_repo
        settings.claim_min_confidence = min_conf
        settings.arxiv_max_results_per_query = arxiv_n
        settings.anthropic_temperature = temp
        settings.pipeline_concurrency = concurrency
        settings.discovery_max_rounds = discovery_rounds

        # Clear previous run: delete DB and old report files so results are
        # always scoped to the current audit inputs.
        set_progress(sid, step=0, msg="Clearing previous run data…", running=True)
        loop.run_until_complete(init_db.reset_database())
        _report_dir = Path(settings.report_output_dir)
        if _report_dir.exists():
            shutil.rmtree(_report_dir, ignore_errors=True)
        set_progress(sid, step=0, msg="Initialising database…", running=True)

        set_progress(sid, msg="Parsing research scope with LLM…")
        session_out = loop.run_until_complete(
            run_user_session_from_text(text, variants=variants if variants else "all")
        )
        if benchmarks:
            session_out = session_out.model_copy(
                update={"benchmarks_of_interest": benchmarks}
            )

        graph = build_graph().compile()
        initial_state = {
            "query_terms": session_out.search_queries,
            "variants_of_interest": session_out.variants_of_interest,
            "benchmarks_of_interest": session_out.benchmarks_of_interest,
            "research_question": session_out.research_question,
            "paper_ids": [],
            "current_paper_id": None,
            "papers_processed": 0,
            "errors": [],
            "report_path": None,
        }

        report_path: str | None = None

        async def _stream() -> None:
            nonlocal report_path
            async for event in graph.astream(initial_state):
                for node_name, node_state in event.items():
                    step = _NODE_TO_STEP.get(node_name, 0)
                    set_progress(sid, step=step, msg=_STEP_MSGS[step])
                    if isinstance(node_state, dict) and node_state.get("report_path"):
                        report_path = node_state["report_path"]

        loop.run_until_complete(_stream())
        set_progress(sid, step=8, msg=_STEP_MSGS[8], running=False, result=report_path)

    except Exception as exc:  # noqa: BLE001
        set_progress(sid, running=False, error=str(exc))
    finally:
        loop.close()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _render_step_tracker(active_step: int) -> None:
    cols = st.columns(7)
    for i, (col, label) in enumerate(zip(cols, _STEP_LABELS)):
        step_num = i + 1
        icon = "✅" if step_num < active_step else ("🔵" if step_num == active_step else "⬜")
        col.markdown(
            f"<div style='text-align:center;font-size:1.3rem'>{icon}</div>"
            f"<div style='text-align:center;font-size:0.68rem;color:#888'>{label}</div>",
            unsafe_allow_html=True,
        )


# ── Session state defaults ────────────────────────────────────────────────────
if "audit_sid" not in st.session_state:
    st.session_state.audit_sid = ""
if "audit_config_display" not in st.session_state:
    st.session_state.audit_config_display = {}

sid: str = st.session_state.audit_sid
progress = get_progress(sid)
pipeline_running = progress.get("running", False)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.title("🔬 ClaimCheck — LoRA Research Audit")
st.markdown(
    "Automated **reproducibility & consensus auditing** for LoRA-variant fine-tuning papers.  \n"
    "Configure your audit below → the 7-agent pipeline discovers papers, extracts benchmark "
    "claims, compares them against released code, and generates a downloadable report."
)

# ── Live DB stats bar ─────────────────────────────────────────────────────────
try:
    _stats = get_audit_stats()
    if _stats["papers_total"] > 0:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Papers in DB", _stats["papers_total"])
        s2.metric("Claims", _stats["claims_total"])
        s3.metric("Gaps", _stats["gaps_total"])
        s4.metric("Contradictions", _stats["contradictions_total"])
        st.caption("Figures above are from the last completed audit run.")
except Exception:  # noqa: BLE001
    pass

st.markdown("---")

# ── Configuration form (hidden while pipeline runs) ───────────────────────────
if not pipeline_running:
    with st.form("audit_form", clear_on_submit=False):
        st.subheader("Configure Your Audit")

        research_q = st.text_area(
            "Research question *",
            placeholder=(
                "Describe what you want to investigate. Be specific about the task, "
                "benchmarks, or model family you care about.\n\n"
                "Example: Which LoRA variants achieve the best accuracy-to-parameter "
                "trade-off on commonsense reasoning benchmarks for LLaMA models?"
            ),
            height=120,
            help="Required. The more specific your question, the better the paper search.",
        )

        col_left, col_right = st.columns(2)

        with col_left:
            max_papers = st.slider(
                "Maximum papers to audit",
                min_value=1, max_value=50, value=5,
                help=(
                    "Each paper requires ~30 K tokens of LLM processing. "
                    "Start with 3–5 to keep cost low; increase for broader coverage."
                ),
            )
            benchmarks_sel = st.multiselect(
                "Benchmarks / tasks of interest",
                options=_KNOWN_BENCHMARKS,
                default=[],
                placeholder="Leave empty to cover all benchmarks",
                help="Narrow the audit to specific evaluation benchmarks.",
            )

        with col_right:
            variants_sel = st.multiselect(
                "LoRA variants to focus on",
                options=_KNOWN_VARIANTS,
                default=[],
                placeholder="Leave empty to cover all variants",
                help="Narrow the search to specific LoRA variants.",
            )
            discovery_rounds = st.select_slider(
                "Discovery rounds",
                options=[1, 2, 3],
                value=2,
                help=(
                    "Round 1: precise queries.  "
                    "Round 2: widened queries.  "
                    "Round 3: broad fallback ('parameter efficient fine-tuning LoRA')."
                ),
            )

        with st.expander("Advanced options", expanded=False):
            adv_l, adv_r = st.columns(2)
            skip_no_repo = adv_l.toggle(
                "Skip papers without a public GitHub repo",
                value=True,
                help="Papers without a detectable repo skip Code Analysis and Gap Analysis.",
            )
            min_confidence = adv_l.slider(
                "Min claim confidence",
                min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                help="Claims below this confidence threshold are discarded.",
            )
            concurrency = adv_l.slider(
                "Pipeline concurrency",
                min_value=1, max_value=5, value=3, step=1,
                help="Papers processed in parallel. Higher = faster but more API load.",
            )
            arxiv_per_query = adv_r.slider(
                "ArXiv results per search query",
                min_value=5, max_value=50, value=20, step=5,
                help="More results increase coverage but slow the discovery step.",
            )
            temperature = adv_r.slider(
                "LLM temperature",
                min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                help="Lower = more deterministic extraction. 0.1 is recommended.",
            )

        submitted = st.form_submit_button(
            "🚀 Run Audit Pipeline",
            type="primary",
            use_container_width=True,
        )

    # ── Pipeline walkthrough ──────────────────────────────────────────────────
    with st.expander("What happens when you run the pipeline?", expanded=False):
        st.markdown("""
| # | Agent | What it does |
|---|---|---|
| 1 | **Paper Discovery** | Searches arXiv up to N rounds; filters with LLM relevance check |
| 2 | **Repo Resolution** | Finds the GitHub repo for each paper (confidence-scored) |
| 3 | **Claim Extraction** | Reads the PDF; extracts structured benchmark claims |
| 4 | **Code Analysis** | Reads released code; captures hyperparameters and training config |
| 5 | **Gap Analysis** | Compares paper claims vs code facts; flags Critical / Major / Minor gaps |
| 6 | **Contradiction Mapping** | Detects conflicts between claims across all papers |
| 7 | **Report Generation** | Produces a Markdown + HTML audit report |

> Papers that fail at any stage are marked `FAILED`; the pipeline continues for others.
> Code is never executed — read-only analysis only.
""")

    # ── Launch pipeline ───────────────────────────────────────────────────────
    if submitted and research_q.strip():
        new_sid = str(uuid.uuid4())
        st.session_state.audit_sid = new_sid
        st.session_state.audit_config_display = {
            "research_q": research_q.strip(),
            "variants": variants_sel,
            "benchmarks": benchmarks_sel,
            "max_papers": max_papers,
        }
        set_progress(
            new_sid,
            running=True, step=0, msg="Starting…", result=None, error=None,
        )
        threading.Thread(
            target=_run_pipeline_thread,
            args=(
                new_sid, research_q.strip(),
                [v.lower() for v in variants_sel],
                benchmarks_sel,
                max_papers,
                skip_no_repo,
                min_confidence,
                arxiv_per_query,
                temperature,
                concurrency,
                discovery_rounds,
            ),
            daemon=True,
        ).start()
        st.rerun()

    elif submitted:
        st.warning("Please fill in the **Research question** field before running.")


# ── Progress display (shown while pipeline is running) ────────────────────────
if pipeline_running:
    step = progress.get("step", 0)
    msg = progress.get("msg", "Running…")
    pct = _STEP_PCT[min(step, 8)] / 100

    st.subheader("⏳ Pipeline Running")
    cfg = st.session_state.audit_config_display
    if cfg:
        st.caption(
            f"**Q:** {cfg.get('research_q', '')[:100]}  ·  "
            f"**Max papers:** {cfg.get('max_papers', '?')}  ·  "
            f"**Variants:** {', '.join(cfg.get('variants', [])) or 'all'}"
        )

    st.progress(pct, text=msg)
    _render_step_tracker(step)
    st.info(
        f"**Step {step}/7 —** {msg}  \n"
        "The pipeline is running in the background. "
        "This page auto-refreshes every 2 seconds."
    )
    time.sleep(2)
    st.rerun()


# ── Error display ─────────────────────────────────────────────────────────────
if sid and not pipeline_running and progress.get("error"):
    st.error(f"**Pipeline failed:** {progress['error']}")
    st.markdown(
        "Check that `ANTHROPIC_API_KEY` and `GITHUB_TOKEN` are set correctly. "
        "See `step.md` for setup instructions."
    )
    if st.button("Try Again", type="primary"):
        st.session_state.audit_sid = ""
        st.rerun()


# ── Results display (shown after pipeline completes successfully) ─────────────
if sid and not pipeline_running and progress.get("step") == 8 and not progress.get("error"):
    st.success("**Audit complete!** All agents finished successfully.")
    st.cache_data.clear()

    st.progress(1.0, text=_STEP_MSGS[8])
    _render_step_tracker(8)

    st.markdown("---")
    st.subheader("Results Summary")
    try:
        fresh = get_audit_stats()
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Papers Audited", fresh["papers_total"])
        r2.metric("Claims Extracted", fresh["claims_total"])
        r3.metric("Total Gaps", fresh["gaps_total"])
        r4.metric("Contradictions", fresh["contradictions_total"])
    except Exception:  # noqa: BLE001
        pass

    st.markdown("### Explore Results")
    nav1, nav2, nav3, nav4 = st.columns(4)
    if nav1.button("📋 Claims", use_container_width=True):
        st.switch_page("pages/01_Claims.py")
    if nav2.button("🔬 Gaps", use_container_width=True):
        st.switch_page("pages/02_Gaps.py")
    if nav3.button("⚡ Contradictions", use_container_width=True):
        st.switch_page("pages/03_Contradictions.py")
    if nav4.button("📊 Report & Download", use_container_width=True, type="primary"):
        st.switch_page("pages/04_Report.py")

    # ── Inline report download ─────────────────────────────────────────────
    report_path = progress.get("result")
    if report_path:
        report_dir = Path(report_path).parent
        md_file = report_dir / "audit_report.md"
        html_file = report_dir / "audit_report.html"
        if md_file.exists() or html_file.exists():
            st.markdown("### Download Report")
            dl1, dl2 = st.columns(2)
            if md_file.exists():
                dl1.download_button(
                    label="⬇️ Markdown (.md)",
                    data=md_file.read_text(encoding="utf-8"),
                    file_name="audit_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            if html_file.exists():
                dl2.download_button(
                    label="⬇️ HTML (.html)",
                    data=html_file.read_text(encoding="utf-8"),
                    file_name="audit_report.html",
                    mime="text/html",
                    use_container_width=True,
                )

    st.markdown("---")
    if st.button("🔄 Run Another Audit"):
        st.session_state.audit_sid = ""
        st.rerun()
