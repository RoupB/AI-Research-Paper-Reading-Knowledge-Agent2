# agents/contradiction_mapping.py

from __future__ import annotations
import json
import uuid
from collections import defaultdict

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

_SYSTEM = """You are a scientific fact-checker comparing benchmark claims across multiple papers.
You will receive a cluster of claims that all report the same metric on the same
dataset with the same base model, from different papers.

Identify contradictions:
- direct_numeric: same setup, significantly different numeric values (>2% relative)
- conditional_flip: paper A says method X beats Y; paper B says Y beats X under
  conditions paper A did not disclose
- dataset_scope: papers use the same dataset name but different splits or versions

For each contradiction provide:
- paper_a_id, paper_b_id, claim_a_id, claim_b_id
- contradiction_type
- description: concrete explanation of what conflicts
- severity: "high" | "medium" | "low"

If no contradiction exists in this cluster, return an empty array."""


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

        clusters: dict[tuple, list[BenchmarkClaim]] = defaultdict(list)
        for c in claims:
            clusters[(c.metric, c.dataset, c.model_base)].append(c)

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

        for (metric, dataset, model_base), cluster in clusters.items():
            distinct_papers = {c.paper_id for c in cluster}
            if len(distinct_papers) < 2:
                continue

            batches = self._batch_cluster(cluster, citation_by_paper)
            for batch in batches:
                raw = await self._assess_cluster(metric, dataset, model_base, batch)
                for rc in raw:
                    contra = self._to_contradiction(rc, valid_claim_ids)
                    if contra is None:
                        continue
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

        return ContradictionMappingOutput(
            contradictions_found=len(contradiction_ids),
            contradiction_ids=contradiction_ids,
            papers_involved=sorted(papers_involved),
        )

    @staticmethod
    def _batch_cluster(
        cluster: list[BenchmarkClaim], citation_by_paper: dict[str, int]
    ) -> list[list[BenchmarkClaim]]:
        if len(cluster) <= 20:
            return [cluster]
        # Pivot on the most-cited paper; pair each other claim against the pivot.
        pivot = max(cluster, key=lambda c: citation_by_paper.get(c.paper_id, 0))
        return [[pivot, c] for c in cluster if c.claim_id != pivot.claim_id]

    async def _assess_cluster(
        self, metric: str, dataset: str, model_base: str, cluster: list[BenchmarkClaim]
    ) -> list[dict]:
        cluster_json = json.dumps([c.model_dump() for c in cluster], default=str)
        user = (
            f"Metric: {metric}\nDataset: {dataset}\nBase model: {model_base}\n\n"
            f"Claims from different papers:\n{cluster_json}\n\nFind contradictions."
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
