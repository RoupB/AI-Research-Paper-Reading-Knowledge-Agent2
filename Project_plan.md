Project Plan: ClaimCheck — An Agentic System for Reproducibility & Consensus Auditing of LoRA-Family Fine-Tuning Methods

1. Problem Statement

Published papers on parameter-efficient fine-tuning (PEFT) methods — LoRA, QLoRA, DoRA, AdaLoRA, LoRA+, VeRA, and similar variants — make numerous claims about performance improvements over baseline LoRA, typically on shared benchmarks (GLUE, commonsense reasoning suites, instruction-tuning evals). However:


Many of these claims are not independently verifiable against the accompanying code (hyperparameters differ, configs are incomplete, or the released code doesn't match the described method).
Claims contradict each other — Method B claims to beat LoRA by X%, Method C claims to beat both, yet Method B and C are rarely compared directly, and when conditions (rank, dataset, model size) differ, "beats LoRA" may not mean the same thing across papers.
Research engineers adopting these methods currently have no systematic way to know which claims hold up, under what conditions, and where the literature actually disagrees — they either trust the abstract or spend days manually digging through code and tables.


ClaimCheck is an agentic pipeline that (1) extracts structured, falsifiable claims from each paper and checks them against the paper's own released code (reproducibility/gap analysis), and (2) aggregates these structured claims across the full LoRA-variant literature to build a "claim graph" showing where methods agree, disagree, or are only comparable under specific conditions (cross-paper consensus mapping).

2. Why This Is a Real and Novel Problem


Real pain point: Research engineers adopting PEFT methods regularly report that benchmark numbers don't reproduce, or that "SOTA" claims only hold under narrow, undisclosed conditions. This is currently solved (if at all) via slow, manual, ad-hoc investigation.
Not addressed by existing tools: Literature search/summarization tools (Elicit, Consensus, SciSpace) summarize what papers say, not whether what they say is true relative to their own code, nor how claims relate across papers in a structured, conditional way.
Publishable contribution shape: The output is not just a tool, but an empirical audit of a specific literature (LoRA variants) — e.g., "X% of claims in this literature are reproducible from released code/configs; Y% of 'beats LoRA' claims hold only under specific rank/dataset conditions." This is the kind of finding that fits reproducibility-focused workshops and "science of science" / meta-research venues.


3. Scope (v1)


Domain: LoRA and its direct variants (target list: LoRA, QLoRA, DoRA, AdaLoRA, LoRA+, VeRA, and ~10-20 closely related papers found during discovery).
In scope:

Automated discovery of relevant papers + their code repos
Structured claim extraction (what is claimed, on what benchmark, under what conditions, vs. what baseline)
Code-vs-paper gap analysis (hyperparameters, method implementation, reproducibility of configs)
Cross-paper claim graph: same-benchmark claims aligned, contradictions/conditions flagged
A validation set of ~20-30 hand-annotated papers to measure agent accuracy



Out of scope (v1):

Actually re-running training/experiments to verify numbers (too compute-heavy) — v1 is a static analysis of paper text + code, not re-execution
The negative-results/internal-lab-notes system (#3) — noted as a future extension
The original four-agent literature-review-to-IEEE-writing pipeline — superseded by this project, but components (paper search, summarization) are reusable





4. System Architecture

Built on LangGraph (Python), as a checkpointed graph with both sequential and agentic-loop sections.

                    ┌─────────────────────────┐
                    │   Paper Discovery Agent   │  (agentic loop, refines
                    │   (arXiv/Semantic Scholar)│   search until enough
                    └───────────┬───────────────┘   relevant papers found)
                                │
                    ┌───────────▼───────────────┐
                    │  Code Repo Resolution Agent │  (finds + clones/fetches
                    │  (GitHub search, link       │   associated code repo
                    │   extraction from paper)    │   for each paper)
                    └───────────┬───────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                                     ▼
   ┌─────────────────────┐                ┌─────────────────────┐
   │  Claim Extraction    │               │  Code Analysis Agent │
   │  Agent (per paper)   │               │  (per paper's repo)  │
   │  - claims, baselines,│               │  - implementation    │
   │    conditions, metric│               │  - configs/hparams   │
   └───────────┬──────────┘               └──────────┬───────────┘
              │                                       │
              └───────────────┬───────────────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │  Gap Analysis Agent       │  (reconciles claims
                    │  (per paper)              │   vs. code → structured
                    └───────────┬───────────────┘   "gap report")
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  Cross-Paper Aggregation  │  (builds claim graph:
                    │  & Contradiction Agent    │   agreements, conflicts,
                    │  (across all papers)      │   conditional claims)
                    └───────────┬───────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  Report Generation        │  (per-paper gap reports +
                    │  (human-readable output)  │   overall consensus map)
                    └─────────────────────────┘

Agent-by-agent description


Paper Discovery Agent (agentic loop — reused/adapted from earlier design): searches arXiv/Semantic Scholar for LoRA-variant papers, refines query if results are too narrow/broad/off-topic, until a target set (e.g., 15-25 papers) is reached.
Code Repo Resolution Agent: for each paper, locates the associated GitHub repo (often linked in the abstract/footer, sometimes requires searching). Flags papers with no usable code — this itself is a data point.
Claim Extraction Agent: reads each paper and extracts a structured list of claims in a fixed schema: {method, baseline, benchmark, metric, claimed_value, baseline_value, conditions (rank, model, dataset size, etc.)}. This structured schema is what makes cross-paper comparison possible later.
Code Analysis Agent: explores the repo's structure, config files, and key implementation files; extracts what hyperparameters/configs are actually specified, and a structural summary of how the method is implemented (e.g., does the code actually implement the described rank-adaptive mechanism, or a simplified version).
Gap Analysis Agent: takes claim extraction + code analysis for a single paper and produces a gap report — does the code support reproducing the claimed numbers (are configs present/complete), does the implementation match the method description, are there discrepancies (e.g., paper claims rank=8 but default config uses rank=16).
Cross-Paper Aggregation & Contradiction Agent: takes all papers' structured claims (from step 3) and builds a claim graph — clusters claims by (benchmark, baseline) pairs, identifies where multiple papers report different numbers for "the same" comparison, and surfaces the conditions under which each claim was made (so "Method B beats LoRA" and "Method C doesn't beat LoRA" might both be true under different ranks/datasets — the agent's job is to surface this nuance rather than flatten it).
Report Generation: compiles per-paper gap reports and the overall claim graph into a structured, human-readable output (and structured JSON/data for analysis).


5. Why Agents (and Not Just an Augmented LLM Call)

This is the key justification, since a single well-prompted LLM call with retrieval could superficially "do" parts of this. Here's why a multi-agent architecture is genuinely needed, not just nice-to-have:


Heterogeneous, multi-step information gathering per paper: For each paper, the system needs to (a) find the paper, (b) find its code repo — which may require a separate web search if not directly linked, (c) read the paper, (d) explore a code repository's file structure to find the relevant files (not just README), (e) cross-reference specific numbers/configs between the two. This is not a single retrieval-then-generate step — it's a sequence of dependent actions where later steps depend on the outcomes of earlier ones (e.g., which files to look at in the repo depends on what the paper's method section describes). This is the core definition of an agentic task: the LLM needs to decide what to look at next based on what it's found so far, not have everything handed to it in one context window.
The agentic refinement loop in discovery is load-bearing, not decorative: A single-pass search for "LoRA variants" will return a mix of relevant, tangential, and noise results. The discovery agent needs to evaluate its own results and decide whether to broaden, narrow, or pivot the search — a feedback loop that a single LLM call cannot perform (it has no "results" to react to within one call).
Separation of concerns improves reliability and debuggability: Claim extraction (reading prose, understanding numbers and conditions) and code analysis (reading file structures, configs, implementation details) are different skills with different failure modes. Combining them into one mega-prompt would mean a single failure mode contaminates everything and makes errors hard to isolate. As separate agents/nodes with structured intermediate outputs (the claim schema, the code analysis schema), each step's output can be validated, logged, and debugged independently — critical for a project whose output needs to be trustworthy enough to publish.
The cross-paper aggregation step genuinely requires all per-paper outputs as input: This is inherently a multi-document reasoning task that depends on the structured outputs of many prior agent runs — it cannot happen "in the same breath" as any single paper's analysis. This is a natural multi-stage pipeline, not something a single augmented LLM call (even with a huge context window) handles well, because the comparison requires normalized, structured data from each paper, not raw text from all papers dumped together.
Context window and cost management: Reading 20+ full papers and their repos in one context would be enormous and expensive, and would force the model to "remember" everything at once. Per-paper agents with structured outputs mean each step operates on a manageable amount of context, and the expensive "read everything" step never has to happen — only the structured summaries get aggregated.
Iterative, conditional control flow: The discovery loop (refine until enough papers found, capped at N attempts), the "no code available → skip code analysis, flag it" branch, and the aggregation step (which only runs once all per-paper analyses are done) are all conditional and stateful — exactly what LangGraph's graph/state model is designed for, and what a single LLM call fundamentally cannot express.


In short: a single augmented LLM call can summarize a paper you hand it. It cannot go find the right paper, find its code, decide what part of the code is relevant to check, compare that against what 19 other papers claimed, and flag the specific conditions under which contradictory claims are each true. That requires a system that takes actions, observes results, and adapts — i.e., agents.

6. Key Benefits


For research engineers: A concrete, queryable map of "what's actually been shown about LoRA variants, under what conditions, and how reproducible it is" — replacing days of manual digging with a structured report.
For the research community: An empirical reproducibility audit of an actively-used literature, surfacing systemic issues (e.g., "60% of 'beats LoRA' claims lack complete configs for reproduction") that individual papers/reviewers don't surface.
As a publishable contribution: The combination of (a) a novel agentic methodology for automated reproducibility auditing, and (b) concrete empirical findings about a specific, relevant literature, fits the shape of accepted papers in reproducibility/meta-research venues and ML workshops.
As a reusable framework: The claim-extraction schema, gap-analysis pattern, and contradiction-mapping approach generalize to other PEFT/method families beyond LoRA — a natural "future work" extension.


7. Validation Plan


Hand-annotate ~20-30 papers (a subset of the full set) with ground-truth claims, conditions, and known gaps (where you, as the human, identify discrepancies between paper and code).
Measure agent precision/recall on: claim extraction accuracy, gap detection accuracy (true gaps found vs. false positives), and contradiction-detection accuracy (does the aggregation agent correctly identify when two papers' claims are/aren't comparable).
This validation set is also a deliverable — a small benchmark for "can an LLM agent audit ML paper reproducibility," which is itself citable.


8. Build Order / Milestones


M1 — Discovery + Repo Resolution: Get the discovery agent finding ~20 LoRA-variant papers + resolving their code repos. Output: a list of (paper, repo, repo-exists Y/N).
M2 — Claim Extraction: Build and test the claim extraction schema on a handful of papers; refine schema based on what's actually extractable.
M3 — Code Analysis + Gap Analysis: For papers with code, build the code analysis agent and gap-analysis reconciliation. Start hand-annotating the validation set in parallel.
M4 — Cross-Paper Aggregation: Once several papers have structured claims, build the contradiction/consensus mapping agent.
M5 — Validation & Report: Run full pipeline on all ~20 papers, compare against hand-annotated validation set, compute accuracy metrics, generate final report (per-paper + aggregate).
M6 — Write-up: Frame findings + methodology as a paper/workshop submission.


9. Open Questions to Resolve as You Go


Exact schema for "claim" and "condition" — will need iteration once you see real extracted claims.
How to handle papers with code in non-Python or unconventional repo structures (some PEFT papers use Jupyter notebooks, others full training frameworks).
Threshold/definition for "contradiction" vs. "different conditions" in the aggregation agent — this is somewhat subjective and may need a human-reviewed rubric.
How much of the validation annotation can be semi-automated (e.g., using one LLM call to propose gaps for human review/correction) vs. fully manual.