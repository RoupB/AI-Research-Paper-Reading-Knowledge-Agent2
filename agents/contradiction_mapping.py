# agents/contradiction_mapping.py

from __future__ import annotations
import json
import re
import uuid
from collections import defaultdict
from itertools import combinations

import networkx as nx
from anthropic import AsyncAnthropic

from agents.base_agent import get_logger
from agents.llm import call_llm_json, make_client
from config import settings
from db import queries
from models import (
    BenchmarkClaim,
    Contradiction,
    ContradictionMappingOutput,
    Paper,
)

log = get_logger(__name__)

_VALID_SEVERITY = {"high", "medium", "low"}

# Heuristic thresholds for direct_numeric detection
_REL_THRESHOLD = 0.01   # 1% relative difference → flag
_ABS_THRESHOLD = 0.5    # 0.5 absolute units (catches near-zero edge cases)

# Synonym maps keep cluster keys stable across capitalisation/abbreviation variants
_METRIC_SYNONYMS: dict[str, str] = {
    "acc": "accuracy",
    "accuracy": "accuracy",
    "bleu": "bleu",
    "bleu-4": "bleu",
    "bleu4": "bleu",
    "rouge": "rouge",
    "rouge-l": "rouge-l",
    "rougel": "rouge-l",
    "f1": "f1",
    "f1 score": "f1",
    "perplexity": "perplexity",
    "ppl": "perplexity",
    "em": "exact_match",
    "exact match": "exact_match",
    "exact_match": "exact_match",
    "mmlu": "mmlu",
}

_DATASET_SYNONYMS: dict[str, str] = {
    "sst2": "sst-2",
    "sst-2": "sst-2",
    "sst 2": "sst-2",
    "glue/sst-2": "sst-2",
    "glue/sst2": "sst-2",
    "mnli": "mnli",
    "glue/mnli": "mnli",
    "qnli": "qnli",
    "glue/qnli": "qnli",
    "qqp": "qqp",
    "glue/qqp": "qqp",
    "mrpc": "mrpc",
    "glue/mrpc": "mrpc",
    "rte": "rte",
    "glue/rte": "rte",
    "cola": "cola",
    "glue/cola": "cola",
    "mt-bench": "mt-bench",
    "mt bench": "mt-bench",
    "mtbench": "mt-bench",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "arc": "arc",
    "arc-e": "arc-easy",
    "arc-easy": "arc-easy",
    "arc-c": "arc-challenge",
    "arc-challenge": "arc-challenge",
    "gsm8k": "gsm8k",
    "humaneval": "humaneval",
    "mbpp": "mbpp",
    "commonsenseqa": "commonsenseqa",
    "csqa": "commonsenseqa",
}

_MODEL_SYNONYMS: dict[str, str] = {
    "llama-7b": "llama-7b",
    "llama 7b": "llama-7b",
    "llama7b": "llama-7b",
    "llama-7b-hf": "llama-7b",
    "llama-13b": "llama-13b",
    "llama 13b": "llama-13b",
    "llama2-7b": "llama2-7b",
    "llama-2-7b": "llama2-7b",
    "llama 2 7b": "llama2-7b",
    "llama2-13b": "llama2-13b",
    "llama-2-13b": "llama2-13b",
    "roberta-large": "roberta-large",
    "roberta large": "roberta-large",
    "bert-base": "bert-base",
    "bert base": "bert-base",
    "bert-large": "bert-large",
    "gpt2": "gpt-2",
    "gpt-2": "gpt-2",
    "t5-base": "t5-base",
    "t5-large": "t5-large",
    "t5-3b": "t5-3b",
    "mistral-7b": "mistral-7b",
    "mistral 7b": "mistral-7b",
    "falcon-7b": "falcon-7b",
    "falcon 7b": "falcon-7b",
}


def _normalize(s: str, synonym_map: dict[str, str]) -> str:
    """Lowercase + collapse whitespace, then apply synonym table."""
    key = re.sub(r"\s+", " ", s.strip().lower())
    return synonym_map.get(key, key)


def _norm_metric(s: str) -> str:
    return _normalize(s, _METRIC_SYNONYMS)


def _norm_dataset(s: str) -> str:
    return _normalize(s, _DATASET_SYNONYMS)


def _norm_model(s: str) -> str:
    return _normalize(s, _MODEL_SYNONYMS)


_SYSTEM = """You are a scientific fact-checker comparing benchmark claims across multiple papers.
You will receive a cluster of claims that all report the same (or closely related) metric on the
same dataset with the same base model, from different papers.

Identify ALL contradictions, including:
- direct_numeric: same experimental setup, different numeric results (flag ANY difference > 1% relative)
- conditional_flip: paper A says method X beats Y; paper B says Y beats X under conditions paper A did not disclose
- dataset_scope: papers use the same dataset name but different splits, versions, or evaluation protocols

Be inclusive rather than exclusive — flag every meaningful discrepancy.
For any numeric difference > 1% relative between claims from different papers, report a contradiction.

For each contradiction return an object:
{
  "paper_a_id": "...",
  "paper_b_id": "...",
  "claim_a_id": "...",
  "claim_b_id": "...",
  "contradiction_type": "direct_numeric" | "conditional_flip" | "dataset_scope",
  "description": "concrete explanation including the actual numeric values",
  "severity": "high" | "medium" | "low"
}

Severity guide: high = >5% relative diff or rank flip; medium = 1–5% diff; low = subtle discrepancy.

Return a JSON array. If claims are truly identical (same value, same conditions, same paper), return [].
For any numeric difference between DIFFERENT papers, always include it."""


class ContradictionMappingAgent:
    """Agent 6 — finds cross-paper contradictions among benchmark claims."""

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        self.client = client or make_client()
        self.log = log

    async def run(self) -> ContradictionMappingOutput:
        claims = await queries.get_all_claims()
        papers = await queries.get_all_papers()
        return await self._process(claims, papers)

    async def _process(
        self, claims: list[BenchmarkClaim], papers: list[Paper]
    ) -> ContradictionMappingOutput:
        if not claims:
            self.log.info("empty_corpus_no_contradictions")
            return ContradictionMappingOutput(
                contradictions_found=0, contradiction_ids=[], papers_involved=[]
            )

        citation_by_paper = {p.arxiv_id: (p.citation_count or 0) for p in papers}
        by_id = {c.claim_id: c for c in claims}

        # Use normalized keys so "Accuracy"/"accuracy"/"acc" all land in the same cluster
        clusters: dict[tuple, list[BenchmarkClaim]] = defaultdict(list)
        for c in claims:
            key = (
                _norm_metric(c.metric),
                _norm_dataset(c.dataset),
                _norm_model(c.model_base),
            )
            clusters[key].append(c)

        graph = nx.DiGraph()
        for c in claims:
            graph.add_node(
                c.claim_id,
                paper_id=c.paper_id,
                metric=c.metric,
                dataset=c.dataset,
                model_base=c.model_base,
                value=c.reported_value,
            )

        contradiction_ids: list[str] = []
        papers_involved: set[str] = set()
        valid_claim_ids = set(by_id.keys())
        seen_pairs: set[frozenset] = set()  # dedup across heuristic + LLM passes

        for (metric, dataset, model_base), cluster in clusters.items():
            distinct_papers = {c.paper_id for c in cluster}
            if len(distinct_papers) < 2:
                continue

            self.log.debug(
                "processing_cluster",
                metric=metric,
                dataset=dataset,
                model_base=model_base,
                n_papers=len(distinct_papers),
                n_claims=len(cluster),
            )

            # Pass 1: heuristic numeric detection — no LLM needed for direct_numeric
            for contra in self._detect_numeric(cluster):
                pair = frozenset({contra.claim_a_id, contra.claim_b_id})
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                await queries.insert_contradiction(contra)
                contradiction_ids.append(contra.contradiction_id)
                papers_involved.update({contra.paper_a_id, contra.paper_b_id})
                graph.add_edge(
                    contra.claim_a_id,
                    contra.claim_b_id,
                    contradiction_type=contra.contradiction_type,
                    severity=contra.severity,
                    description=contra.description,
                )

            # Pass 2: LLM — catches conditional_flip, dataset_scope, and any numeric misses
            for batch in self._batch_cluster(cluster, citation_by_paper):
                raw = await self._assess_cluster(metric, dataset, model_base, batch)
                for rc in raw:
                    contra = self._to_contradiction(rc, valid_claim_ids)
                    if contra is None:
                        continue
                    pair = frozenset({contra.claim_a_id, contra.claim_b_id})
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    await queries.insert_contradiction(contra)
                    contradiction_ids.append(contra.contradiction_id)
                    papers_involved.update({contra.paper_a_id, contra.paper_b_id})
                    graph.add_edge(
                        contra.claim_a_id,
                        contra.claim_b_id,
                        contradiction_type=contra.contradiction_type,
                        severity=contra.severity,
                        description=contra.description,
                    )

        self._serialise_graph(graph)

        self.log.info(
            "contradiction_mapping_done",
            contradictions_found=len(contradiction_ids),
            papers_involved=len(papers_involved),
        )

        return ContradictionMappingOutput(
            contradictions_found=len(contradiction_ids),
            contradiction_ids=contradiction_ids,
            papers_involved=sorted(papers_involved),
        )

    def _detect_numeric(self, cluster: list[BenchmarkClaim]) -> list[Contradiction]:
        """Rule-based direct_numeric detection — no LLM required."""
        results: list[Contradiction] = []
        for ca, cb in combinations(cluster, 2):
            if ca.paper_id == cb.paper_id:
                continue
            va, vb = ca.reported_value, cb.reported_value
            denom = max(abs(va), abs(vb), 1e-9)
            rel_diff = abs(va - vb) / denom
            abs_diff = abs(va - vb)
            if rel_diff < _REL_THRESHOLD and abs_diff < _ABS_THRESHOLD:
                continue
            if rel_diff >= 0.05 or abs_diff >= 2.0:
                severity = "high"
            elif rel_diff >= 0.01 or abs_diff >= 0.5:
                severity = "medium"
            else:
                severity = "low"
            description = (
                f"Paper {ca.paper_id} reports {ca.metric}={va}{' ' + ca.unit if ca.unit else ''} "
                f"on {ca.dataset} ({ca.model_base}), while paper {cb.paper_id} reports {vb}. "
                f"Relative difference: {rel_diff:.1%}, absolute: {abs_diff:.3f}."
            )
            results.append(
                Contradiction(
                    contradiction_id=str(uuid.uuid4()),
                    paper_a_id=ca.paper_id,
                    paper_b_id=cb.paper_id,
                    claim_a_id=ca.claim_id,
                    claim_b_id=cb.claim_id,
                    contradiction_type="direct_numeric",
                    description=description,
                    severity=severity,
                )
            )
        return results

    @staticmethod
    def _batch_cluster(
        cluster: list[BenchmarkClaim], citation_by_paper: dict[str, int]
    ) -> list[list[BenchmarkClaim]]:
        if len(cluster) <= 20:
            return [cluster]
        pivot = max(cluster, key=lambda c: citation_by_paper.get(c.paper_id, 0))
        return [[pivot, c] for c in cluster if c.claim_id != pivot.claim_id]

    async def _assess_cluster(
        self, metric: str, dataset: str, model_base: str, cluster: list[BenchmarkClaim]
    ) -> list[dict]:
        cluster_json = json.dumps([c.model_dump() for c in cluster], default=str)
        user = (
            f"Metric: {metric}\nDataset: {dataset}\nBase model: {model_base}\n\n"
            f"Claims from different papers:\n{cluster_json}\n\n"
            f"Identify ALL contradictions between these claims."
        )
        try:
            data = await call_llm_json(self.client, _SYSTEM, user)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("contradiction_llm_failed", error=str(exc))
            return []
        if isinstance(data, dict):
            data = data.get("contradictions", [])
        return data if isinstance(data, list) else []

    def _to_contradiction(self, rc: dict, valid_claim_ids: set[str]) -> Contradiction | None:
        try:
            claim_a, claim_b = rc["claim_a_id"], rc["claim_b_id"]
            if claim_a not in valid_claim_ids or claim_b not in valid_claim_ids:
                self.log.warning("invalid_claim_ids_discarded")
                return None
            severity = rc.get("severity", "low")
            if severity not in _VALID_SEVERITY:
                severity = "low"
            return Contradiction(
                contradiction_id=str(uuid.uuid4()),
                paper_a_id=str(rc["paper_a_id"]),
                paper_b_id=str(rc["paper_b_id"]),
                claim_a_id=str(claim_a),
                claim_b_id=str(claim_b),
                contradiction_type=str(rc.get("contradiction_type", "direct_numeric")),
                description=str(rc.get("description", "")),
                severity=severity,
            )
        except (KeyError, TypeError) as exc:
            self.log.warning("contradiction_parse_failed", error=str(exc))
            return None

    def _serialise_graph(self, graph: nx.DiGraph) -> None:
        try:
            path = settings.artifacts_dir / "claim_graph.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = nx.node_link_data(graph)
            path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.log.warning("graph_serialise_failed", error=str(exc))
