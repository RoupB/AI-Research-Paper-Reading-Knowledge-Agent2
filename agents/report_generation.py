# agents/report_generation.py

from __future__ import annotations
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from anthropic import AsyncAnthropic

from agents.base_agent import get_logger
from agents.llm import call_llm
from agents.llm import make_client
from db import queries
from models import ReportGenerationInput, ReportGenerationOutput

log = get_logger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

_SYSTEM = """You are a technical audit report generator for ML reproducibility studies.
Write in precise, neutral scientific language. Do not hedge or soften findings — if a claim cannot be reproduced from the code, state it directly.

### Formatting Rules
- Use **tables** as the primary medium for presenting findings (e.g., paper ID, claim, code evidence, gap classification).
- Keep prose sections short: 2–3 sentences max for summaries, transitions, or context.
- Organize the report into clear sections: Executive Summary, Methodology, Key Findings, Gap Tables, Contradictions, Recommendations.
- Within each section, prefer **structured lists** or tables over long paragraphs.
- Highlight severity levels (Critical, Major, Minor) in a dedicated column, not inline text.
- Ensure flow: start with high-level summary → detailed tabular evidence → synthesis → remediation recommendations.

### Audience
- ML practitioners: need quick visibility into reproducibility gaps.
- Paper authors: need actionable, specific feedback on missing or mismatched code.

# Output Style
- Neutral, scientific tone.
- No narrative filler.
- Tables must be clean, aligned, and scannable.
- Use consistent terminology (e.g., "Gap Type", "Claim", "Evidence", "Status")."""

_METHODOLOGY = """This report was produced by ClaimCheck, an automated agentic pipeline that
discovers LoRA-variant papers from arXiv, resolves their GitHub repositories,
extracts structured benchmark claims from PDFs, audits the released code, and
maps cross-paper contradictions. All findings are automated analysis and should
be treated as hypotheses for human verification, not definitive conclusions.
Static analysis only — code is read, never executed."""


class ReportGenerationAgent:
    """Agent 7 — aggregates all DB records into a Markdown + HTML audit report."""

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        self.client = client or make_client()
        self.log = log

    async def run(
        self,
        output_dir: str | None = None,
        include_raw_claims: bool = False,
        severity_filter: str | None = None,
    ) -> ReportGenerationOutput:
        from config import settings

        inp = ReportGenerationInput(
            output_dir=output_dir or str(settings.report_output_dir),
            include_raw_claims=include_raw_claims,
            severity_filter=severity_filter,
        )

        stats = await queries.get_audit_stats()
        papers = await queries.get_all_papers()
        all_gaps = await queries.get_all_gaps()
        contradictions = await queries.get_all_contradictions()
        all_claims = await queries.get_all_claims()

        if stats["papers_total"] == 0:
            self.log.error("empty_db_report")
            out_dir = self._ensure_dir(inp.output_dir)
            md_path = out_dir / "audit_report.md"
            md_path.write_text(
                "# LoRA Variants Research Audit Report\n\nNo papers were processed.\n",
                encoding="utf-8",
            )
            raise SystemExit(1)

        critical_gaps = [g for g in all_gaps if g.severity == "critical"]
        high_contradictions = [c for c in contradictions if c.severity == "high"]

        exec_summary = await self._exec_summary(stats, all_gaps, contradictions)

        try:
            import markdown as md_lib
            exec_summary_html = md_lib.markdown(exec_summary, extensions=["tables"])
        except Exception:
            exec_summary_html = exec_summary.replace("\n", "<br>")

        # ── Chart data ──────────────────────────────────────────────────────
        gs = stats["gaps_by_severity"]
        cs = stats.get("contradictions_by_severity", {})

        gaps_by_paper_map: dict[str, list] = defaultdict(list)
        for g in all_gaps:
            gaps_by_paper_map[g.paper_id].append(g)

        gaps_per_paper_stats: list[dict] = []
        for p in papers:
            pg = gaps_by_paper_map.get(p.arxiv_id, [])
            short = (p.title[:42] + "…") if len(p.title) > 42 else p.title
            gaps_per_paper_stats.append({
                "paper_id": p.arxiv_id,
                "title":    short,
                "critical": sum(1 for g in pg if g.severity == "critical"),
                "major":    sum(1 for g in pg if g.severity == "major"),
                "minor":    sum(1 for g in pg if g.severity == "minor"),
                "total":    len(pg),
            })
        gaps_per_paper_stats.sort(key=lambda x: x["total"], reverse=True)

        gap_type_counts = Counter(g.gap_type for g in all_gaps)
        status_counts   = Counter(p.status.value for p in papers)

        chart_data = {
            "severity": {
                "labels": ["Critical", "Major", "Minor"],
                "values": [gs.get("critical", 0), gs.get("major", 0), gs.get("minor", 0)],
                "colors": ["#c62828", "#e65100", "#2e7d32"],
            },
            "gaps_per_paper": {
                "papers":   [x["title"]    for x in gaps_per_paper_stats[:15]],
                "critical": [x["critical"] for x in gaps_per_paper_stats[:15]],
                "major":    [x["major"]    for x in gaps_per_paper_stats[:15]],
                "minor":    [x["minor"]    for x in gaps_per_paper_stats[:15]],
            },
            "gap_types": {
                "labels": [k for k, _ in gap_type_counts.most_common(10)],
                "values": [v for _, v in gap_type_counts.most_common(10)],
            },
            "status": {
                "labels": list(status_counts.keys()),
                "values": list(status_counts.values()),
            },
            "contradictions": {
                "labels": ["High", "Medium", "Low"],
                "values": [cs.get("high", 0), cs.get("medium", 0), cs.get("low", 0)],
            },
        }
        # ────────────────────────────────────────────────────────────────────

        context = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stats": stats,
            "papers": papers,
            "exec_summary": exec_summary,
            "exec_summary_html": exec_summary_html,
            "methodology": _METHODOLOGY,
            "all_gaps": all_gaps,
            "critical_gaps": critical_gaps,
            "contradictions": contradictions,
            "high_contradictions": high_contradictions,
            "conditional_claims": [c for c in all_claims if c.is_conditional],
            "all_claims": all_claims if inp.include_raw_claims else [],
            "include_raw_claims": inp.include_raw_claims,
            "chart_data": chart_data,
            "gaps_per_paper_stats": gaps_per_paper_stats,
        }

        out_dir = self._ensure_dir(inp.output_dir)
        md = self._render_markdown(context)
        html = self._render_html(context, md)

        md_path = out_dir / "audit_report.md"
        html_path = out_dir / "audit_report.html"
        try:
            md_path.write_text(md, encoding="utf-8")
            html_path.write_text(html, encoding="utf-8")
        except PermissionError:
            out_dir = self._ensure_dir("/tmp/lora_audit_report")
            md_path = out_dir / "audit_report.md"
            html_path = out_dir / "audit_report.html"
            md_path.write_text(md, encoding="utf-8")
            html_path.write_text(html, encoding="utf-8")
            self.log.warning("report_fallback_path", path=str(out_dir))

        return ReportGenerationOutput(
            report_md_path=str(md_path),
            report_html_path=str(html_path),
            papers_in_report=stats["papers_total"],
            total_gaps=stats["gaps_total"],
            total_contradictions=stats["contradictions_total"],
        )

    @staticmethod
    def _ensure_dir(path: str) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    async def _exec_summary(self, stats, all_gaps, contradictions) -> str:
        gs = stats["gaps_by_severity"]
        top_gaps = [
            {"paper_id": g.paper_id, "gap_type": g.gap_type, "severity": g.severity,
             "description": g.description}
            for g in all_gaps[:5]
        ]
        top_contra = [
            {"paper_a_id": c.paper_a_id, "paper_b_id": c.paper_b_id,
             "type": c.contradiction_type, "description": c.description}
            for c in contradictions[:3]
        ]
        user = (
            "Statistics:\n\n"
            f"Papers audited: {stats['papers_total']}\n"
            f"Total benchmark claims extracted: {stats['claims_total']}\n"
            f"Reproducibility gaps found: {stats['gaps_total']}\n"
            f"  - Critical: {gs.get('critical', 0)}\n"
            f"  - Major: {gs.get('major', 0)}\n"
            f"  - Minor: {gs.get('minor', 0)}\n"
            f"Contradictions across papers: {stats['contradictions_total']}\n"
            f"  - High severity: {stats['contradictions_by_severity'].get('high', 0)}\n\n"
            f"Top 5 reproducibility gaps:\n{json.dumps(top_gaps, indent=2)}\n\n"
            f"Top 3 cross-paper contradictions:\n{json.dumps(top_contra, indent=2)}\n\n"
            "Generate an **Executive Summary** with 3–5 short paragraphs.\n"
            "- Use provided statistics placeholders.\n"
            "- Summarize top reproducibility gaps and contradictions clearly.\n"
            "- Avoid narrative filler; keep sentences concise.\n\n"
            "Then generate a **Key Findings** section:\n"
            "- Bullet list, max 10 items.\n"
            "- Order by severity (Critical → Major → Minor).\n"
            "- Each bullet ≤ 2 lines."
        )
        try:
            return await call_llm(self.client, _SYSTEM, user)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("exec_summary_failed", error=str(exc))
            return (
                f"Audited {stats['papers_total']} papers, extracting "
                f"{stats['claims_total']} benchmark claims. Found {stats['gaps_total']} "
                f"reproducibility gaps and {stats['contradictions_total']} cross-paper "
                "contradictions."
            )

    def _render_markdown(self, ctx: dict) -> str:
        tmpl = _TEMPLATES_DIR / "report.md.j2"
        if tmpl.exists():
            try:
                import jinja2
                env = jinja2.Environment(
                    loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
                    autoescape=False,
                )
                return env.get_template("report.md.j2").render(**ctx)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("md_template_failed", error=str(exc))
        return self._fallback_markdown(ctx)

    def _render_html(self, ctx: dict, md: str) -> str:
        tmpl = _TEMPLATES_DIR / "report.html.j2"
        if tmpl.exists():
            try:
                import jinja2
                env = jinja2.Environment(
                    loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
                    autoescape=True,
                )
                return env.get_template("report.html.j2").render(**ctx)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("html_template_failed", error=str(exc))
        try:
            import markdown as md_lib
            body = md_lib.markdown(md, extensions=["tables"])
        except Exception:  # noqa: BLE001
            body = f"<pre>{md}</pre>"
        return (
            f"<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
            f"<title>LoRA Audit Report</title></head><body>{body}</body></html>"
        )

    @staticmethod
    def _deduplicate_gaps(all_gaps):
        """
        Collapse duplicate gaps for the same paper that share the same
        gap_type and a common description prefix (first 60 chars).
        Returns a list of (gap, count) tuples.
        """
        from collections import defaultdict
        buckets = defaultdict(list)
        for g in all_gaps:
            key = (g.paper_id, g.gap_type, g.description[:60])
            buckets[key].append(g)

        deduped = []
        for (paper_id, gap_type, _), group in buckets.items():
            rep = group[0]
            deduped.append((rep, len(group)))
        sev_order = {"critical": 0, "major": 1, "minor": 2}
        deduped.sort(key=lambda x: (sev_order.get(x[0].severity, 9), x[0].paper_id))
        return deduped

    def _fallback_markdown(self, ctx: dict) -> str:
        from collections import defaultdict

        s = ctx["stats"]
        gs = s["gaps_by_severity"]
        papers_by_id = {p.arxiv_id: p for p in ctx["papers"]}

        lines = [
            "# LoRA Variants Research Audit Report",
            f"*Generated at {ctx['generated_at']}*",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            ctx["exec_summary"],
            "",
            "---",
            "",
            "## Methodology",
            "",
            ctx["methodology"],
            "",
            "---",
            "",
            "## Overview Statistics",
            "",
            "| Metric | Count |",
            "|---|---|",
            f"| Papers audited | {s['papers_total']} |",
            f"| Benchmark claims extracted | {s['claims_total']} |",
            f"| Conditional claims | {s.get('claims_conditional', '—')} |",
            f"| Code facts extracted | {s.get('code_facts_total', '—')} |",
            f"| Reproducibility gaps | {s['gaps_total']} |",
            f"| — Critical | {gs.get('critical', 0)} |",
            f"| — Major | {gs.get('major', 0)} |",
            f"| — Minor | {gs.get('minor', 0)} |",
            f"| Cross-paper contradictions | {s['contradictions_total']} |",
            "",
            "---",
            "",
            "## Papers Audited",
            "",
            "| ArXiv ID | Title | Variant | Repo | Status |",
            "|---|---|---|---|---|",
        ]
        for p in ctx["papers"]:
            repo = f"[repo]({p.repo_url})" if p.repo_url else "—"
            lines.append(
                f"| {p.arxiv_id} | {p.title[:60]}{'…' if len(p.title) > 60 else ''} "
                f"| {p.lora_variant_tag} | {repo} | {p.status.value} |"
            )

        lines += ["", "---", "", "## Reproducibility Gaps", ""]

        gaps_by_paper = defaultdict(list)
        for g in ctx["all_gaps"]:
            gaps_by_paper[g.paper_id].append(g)

        sev_emoji = {"critical": "🔴", "major": "🟠", "minor": "🟡"}

        for paper_id, paper_gaps in sorted(gaps_by_paper.items()):
            paper = papers_by_id.get(paper_id)
            title = paper.title[:70] if paper else paper_id
            deduped = self._deduplicate_gaps(paper_gaps)

            crit  = sum(1 for g, _ in deduped if g.severity == "critical")
            maj   = sum(1 for g, _ in deduped if g.severity == "major")
            minor = sum(1 for g, _ in deduped if g.severity == "minor")

            lines += [
                f"### {paper_id} — {title}",
                "",
                f"**Gaps:** {crit} Critical · {maj} Major · {minor} Minor",
                "",
                "| Severity | Gap Type | Description | Paper Value | Code Value |",
                "|---|---|---|---|---|",
            ]
            for g, count in deduped:
                emoji = sev_emoji.get(g.severity, "⚪")
                desc = g.description[:120].replace("|", "\\|")
                if count > 1:
                    desc += f" *(+{count - 1} similar)*"
                pval = (g.paper_value or "—")[:30].replace("|", "\\|")
                cval = (g.code_value or "—")[:30].replace("|", "\\|")
                lines.append(
                    f"| {emoji} {g.severity.upper()} | `{g.gap_type}` | {desc} | {pval} | {cval} |"
                )
            lines.append("")

        lines += ["---", "", "## Cross-Paper Contradictions", ""]
        if ctx["contradictions"]:
            lines += [
                "| Severity | Paper A | Paper B | Type | Description |",
                "|---|---|---|---|---|",
            ]
            for c in ctx["contradictions"]:
                emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(c.severity, "⚪")
                desc = c.description[:100].replace("|", "\\|")
                lines.append(
                    f"| {emoji} {c.severity.upper()} | {c.paper_a_id} | {c.paper_b_id} "
                    f"| `{c.contradiction_type}` | {desc} |"
                )
        else:
            lines.append(
                "> No cross-paper contradictions detected. This reflects claim "
                "non-overlap across the audited corpus, not verified consistency."
            )

        lines += [
            "", "---", "", "## Conditional Claim Registry", "",
            f"*{len(ctx['conditional_claims'])} claims flagged as conditional "
            "(results only valid under specific hyperparameter/dataset conditions).*",
            "",
        ]

        cond_by_paper = defaultdict(list)
        for c in ctx["conditional_claims"]:
            cond_by_paper[c.paper_id].append(c)

        for paper_id, claims in sorted(cond_by_paper.items()):
            paper = papers_by_id.get(paper_id)
            title = paper.title[:70] if paper else paper_id
            lines += [
                f"### {paper_id} — {title}",
                "",
                "| Metric | Dataset | Model | Value | Key Conditions |",
                "|---|---|---|---|---|",
            ]
            for c in claims:
                cond_str = ", ".join(
                    f"{k}={v}" for k, v in list(c.conditions.items())[:3]
                )
                if len(c.conditions) > 3:
                    cond_str += f" *(+{len(c.conditions)-3} more)*"
                cond_str = cond_str[:80].replace("|", "\\|")
                dataset = (c.dataset or "—")[:40].replace("|", "\\|")
                lines.append(
                    f"| {c.metric} | {dataset} | {c.model_base[:30]} "
                    f"| {c.reported_value} | {cond_str} |"
                )
            lines.append("")

        if ctx["include_raw_claims"] and ctx["all_claims"]:
            lines += [
                "---", "", "## Appendix: All Extracted Claims", "",
                "| Paper | Metric | Dataset | Model | Value | Unit |",
                "|---|---|---|---|---|---|",
            ]
            for c in ctx["all_claims"]:
                lines.append(
                    f"| {c.paper_id} | {c.metric} | {c.dataset[:30]} "
                    f"| {c.model_base[:25]} | {c.reported_value} | {c.unit or '—'} |"
                )

        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = ReportGenerationAgent()
        result = await agent.run()
        print(f"Report MD: {result.report_md_path}")
        print(f"Report HTML: {result.report_html_path}")
        print(f"Papers in report: {result.papers_in_report}")
        print(f"Total gaps: {result.total_gaps}")
        print(f"Total contradictions: {result.total_contradictions}")

    asyncio.run(main())
