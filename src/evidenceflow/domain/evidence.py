"""Domain Models for read-only evidence navigation around one indexed Chunk."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidenceflow.domain.chunk import DocumentChunk


class EvidenceIndexIdentity(BaseModel):
    """Identity of the one index a navigation result was read from.

    ``index_signature`` is the lifecycle pointer's signature, so two indexes that share a collection
    name and Chunk count are still distinguishable (PROJECT_SPEC §13.1). It is ``None`` only in the
    unmanaged startup path, where no lifecycle state file exists and therefore no signature does
    either.
    """

    model_config = ConfigDict(frozen=True)

    collection_name: str = Field(min_length=1, description="被读取的 Chroma Collection 名称。")
    index_signature: str | None = Field(
        default=None,
        min_length=1,
        description="lifecycle active 指针记录的 index_signature；无状态文件时为 None。",
    )
    chunk_count: int = Field(ge=0, description="该索引声明的 Chunk 数量。")
    governance: Literal["lifecycle_active", "lifecycle_staging", "settings_fallback"] = Field(
        description=(
            "lifecycle_active = 由 active 指针治理；lifecycle_staging = 只读评测时由 staging 指针治理；"
            "settings_fallback = 无状态文件时的回落。"
        )
    )

    @property
    def is_managed(self) -> bool:
        """Whether a lifecycle pointer (and therefore a signature) governs this index."""

        return self.governance in {"lifecycle_active", "lifecycle_staging"}


class IndexedDocument(BaseModel):
    """可浏览的索引文档，页码范围仅覆盖已索引片段。"""

    document_id: str
    file_name: str
    chunk_count: int
    page_start: int | None
    page_end: int | None


class EvidenceNeighborhood(BaseModel):
    """One anchor Chunk plus the existing Chunks at adjacent ``chunk_index`` values in its document.

    ``previous`` is ordered from the farthest to the nearest neighbour (ascending ``chunk_index``),
    and ``following`` from the nearest to the farthest, so ``[ *previous, anchor, *following ]``
    reads in document order. Only the anchor's own ``document_id`` may appear here.
    """

    model_config = ConfigDict(frozen=True)

    anchor: DocumentChunk = Field(description="请求的 chunk_id 对应的完整 Chunk。")
    previous: tuple[DocumentChunk, ...] = Field(
        default=(),
        description="同一文档中 chunk_index 较小的相邻 Chunk，按文档顺序排列。",
    )
    following: tuple[DocumentChunk, ...] = Field(
        default=(),
        description="同一文档中 chunk_index 较大的相邻 Chunk，按文档顺序排列。",
    )
    identity: EvidenceIndexIdentity = Field(description="读取该结果的索引身份。")
    page_start: int | None = Field(
        default=None,
        ge=1,
        description="整个邻域覆盖的首个真实 1-based PDF 页码；无页码来源为 None。",
    )
    page_end: int | None = Field(
        default=None,
        ge=1,
        description="整个邻域覆盖的末个真实 1-based PDF 页码；无页码来源为 None。",
    )

    @model_validator(mode="after")
    def validate_neighborhood(self) -> "EvidenceNeighborhood":
        """Reject cross-document, unordered or out-of-range neighbours at the model boundary."""

        document_id = self.anchor.document_id
        previous_indices = [chunk.chunk_index for chunk in self.previous]
        following_indices = [chunk.chunk_index for chunk in self.following]
        if any(chunk.document_id != document_id for chunk in (*self.previous, *self.following)):
            raise ValueError("evidence neighbourhood must stay inside the anchor's document")
        if previous_indices != sorted(previous_indices) or len(set(previous_indices)) != len(
            previous_indices
        ):
            raise ValueError("previous chunks must be strictly increasing by chunk_index")
        if following_indices != sorted(following_indices) or len(set(following_indices)) != len(
            following_indices
        ):
            raise ValueError("following chunks must be strictly increasing by chunk_index")
        if previous_indices and previous_indices[-1] >= self.anchor.chunk_index:
            raise ValueError("previous chunks must precede the anchor")
        if following_indices and following_indices[0] <= self.anchor.chunk_index:
            raise ValueError("following chunks must follow the anchor")
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must both be set or both be None")
        if self.page_start is not None and self.page_start > self.page_end:
            raise ValueError("page_start must be less than or equal to page_end")
        return self

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        """Return the anchor and its neighbours in document order."""

        return (*self.previous, self.anchor, *self.following)
