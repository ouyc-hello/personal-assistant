from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from personal_assistant.rag.router import classify_route
from personal_assistant.schemas import Route, SearchHit


class SearchBackend(Protocol):
    """Common query contract for memory, document and graph retrieval."""

    def search(self, query: str, *, user_id: str, limit: int = 8) -> Sequence[SearchHit]: ...


PermissionCheck = Callable[[SearchHit, str], bool]


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieval output plus an explainable, backend-level execution trace."""

    route: Route
    hits: tuple[SearchHit, ...]
    trace: tuple[str, ...]
    backend_hits: Mapping[str, tuple[SearchHit, ...]] = field(default_factory=dict)


class UnifiedRetriever:
    """Route, authorize, fuse and rerank the three retrieval backends.

    The retriever never changes the source-system state. It only combines already
    authorized candidates; PostgreSQL remains the memory source of truth, Milvus
    the document index, and Neo4j the graph index.
    """

    def __init__(
        self,
        backends: Mapping[str, SearchBackend],
        *,
        backend_weights: Mapping[str, float] | None = None,
        permission_check: PermissionCheck | None = None,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.backends = dict(backends)
        self.backend_weights = {name: 1.0 for name in self.backends}
        self.backend_weights.update(backend_weights or {})
        if any(weight <= 0 for weight in self.backend_weights.values()):
            raise ValueError("backend weights must be positive")
        self.permission_check = permission_check or default_permission_check
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        user_id: str,
        route: Route | None = None,
        limit: int = 8,
        per_backend_limit: int | None = None,
    ) -> RetrievalResult:
        if limit <= 0:
            return RetrievalResult(route=route or classify_route(query), hits=(), trace=("retrieval:skip limit<=0",))
        selected_route = route or classify_route(query)
        backend_names = route_backends(selected_route)
        candidate_limit = per_backend_limit or max(limit, 8)
        if candidate_limit <= 0:
            raise ValueError("per_backend_limit must be positive")

        trace: list[str] = [f"route:{selected_route.value}", f"retrieval:backends={','.join(backend_names) or 'none'}"]
        authorized: dict[str, tuple[SearchHit, ...]] = {}
        for backend_name in backend_names:
            backend = self.backends.get(backend_name)
            if backend is None:
                trace.append(f"retrieval:{backend_name}:unconfigured")
                continue
            try:
                raw_hits = list(backend.search(query, user_id=user_id, limit=candidate_limit))
            except Exception as exc:  # one unavailable backend must not leak or abort the turn
                trace.append(f"retrieval:{backend_name}:error={type(exc).__name__}")
                continue
            filtered = tuple(hit for hit in raw_hits if self.permission_check(hit, user_id))
            denied = len(raw_hits) - len(filtered)
            authorized[backend_name] = filtered
            trace.append(f"retrieval:{backend_name}:raw={len(raw_hits)}:allowed={len(filtered)}:denied={denied}")

        fused = reciprocal_rank_fusion(authorized, weights=self.backend_weights, rrf_k=self.rrf_k)
        hits = tuple(item.hit for item in fused[:limit])
        trace.append(f"retrieval:fused={len(hits)}")
        return RetrievalResult(
            route=selected_route,
            hits=hits,
            trace=tuple(trace),
            backend_hits=authorized,
        )


def route_backends(route: Route) -> tuple[str, ...]:
    return {
        Route.CHAT: (),
        Route.MEMORY: ("postgresql+pgvector",),
        Route.DOCUMENT: ("milvus",),
        Route.GRAPH: ("neo4j",),
        Route.HYBRID: ("postgresql+pgvector", "milvus", "neo4j"),
    }[route]


def default_permission_check(hit: SearchHit, user_id: str) -> bool:
    """Apply a second authorization boundary to backend output.

    Adapters should already scope queries by ``user_id``. This defense-in-depth
    check rejects a hit carrying an explicit owner or allow-list for another user.
    Hits without those fields remain valid because some adapters enforce scope
    in the query itself.
    """

    metadata = hit.metadata
    owner = metadata.get("user_id") or metadata.get("owner_user_id")
    if owner is not None and str(owner) != user_id:
        return False
    allowed = metadata.get("allowed_user_ids")
    if allowed is not None and user_id not in {str(value) for value in allowed}:
        return False
    if metadata.get("visibility") == "private" and owner is None:
        return False
    return True


@dataclass(frozen=True)
class _FusedHit:
    hit: SearchHit
    fusion_score: float
    first_rank: int

    @property
    def id(self) -> str:
        return _dedupe_key(self.hit)


def reciprocal_rank_fusion(
    backend_hits: Mapping[str, Sequence[SearchHit]],
    *,
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
) -> list[_FusedHit]:
    """Fuse ranked lists deterministically while retaining backend provenance."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    weights = weights or {}
    merged: dict[str, _FusedHit] = {}
    for backend_name, hits in backend_hits.items():
        weight = float(weights.get(backend_name, 1.0))
        if weight <= 0:
            raise ValueError("backend weights must be positive")
        for rank, hit in enumerate(hits, start=1):
            key = _dedupe_key(hit)
            contribution = weight / (rrf_k + rank)
            previous = merged.get(key)
            if previous is None:
                metadata = dict(hit.metadata)
                metadata.update({
                    "retrieved_from": [backend_name],
                    "retrieval_rank": rank,
                    "fusion_contributions": {backend_name: contribution},
                })
                enriched = hit.model_copy(update={"metadata": metadata})
                merged[key] = _FusedHit(enriched, contribution, rank)
            else:
                sources = list(previous.hit.metadata.get("retrieved_from", []))
                if backend_name not in sources:
                    sources.append(backend_name)
                metadata = dict(previous.hit.metadata)
                metadata["retrieved_from"] = sources
                metadata["fusion_contributions"] = {
                    **dict(metadata.get("fusion_contributions", {})),
                    backend_name: contribution,
                }
                merged[key] = _FusedHit(
                    previous.hit.model_copy(update={"metadata": metadata}),
                    previous.fusion_score + contribution,
                    min(previous.first_rank, rank),
                )

    values = list(merged.values())
    values.sort(key=lambda item: (-item.fusion_score, item.first_rank, item.hit.backend, item.hit.id))
    return [
        _FusedHit(
            item.hit.model_copy(update={"score": item.fusion_score}),
            item.fusion_score,
            item.first_rank,
        )
        for item in values
    ]


def _dedupe_key(hit: SearchHit) -> str:
    canonical = hit.metadata.get("canonical_id")
    if canonical:
        return f"canonical:{canonical}"
    return f"{hit.backend}:{hit.id}"
