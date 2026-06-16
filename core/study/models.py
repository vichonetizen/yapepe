"""
models.py — Contratos de datos de EstudIA (pista integrada).

Auto-contenido a propósito: la pista integrada vive dentro de pentamodal-app y NO
depende del esqueleto externo (lectura/estudia). Así las dos pistas quedan
"por separado". La regla dura del proyecto vive aquí como tipos: toda Question y
todo GradeResult llevan `evidence` (chunk_ids fuente). La validación efectiva
(que esos chunk_ids existan y resuelvan a una cita) la hace grounding.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --------------------------------------------------------------------------- #
# Enumeraciones
# --------------------------------------------------------------------------- #
class DocFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    IMAGE = "image"


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    FIGURE_CAPTION = "figure_caption"
    FORMULA = "formula"


class Relation(str, Enum):
    DEFINES = "defines"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    IS_A = "is-a"
    PART_OF = "part-of"
    CITES = "cites"


class Focus(str, Enum):
    ALL = "all"
    TOPIC = "topic"
    COMPARE = "compare"
    HIERARCHY = "hierarchy"
    MAP = "map"


class QuestionType(str, Enum):
    OPEN = "open"
    MCQ = "mcq"
    TF = "tf"
    CLOZE = "cloze"


# --------------------------------------------------------------------------- #
# Cita — el "sello" que produce grounding.resolve_evidence()
# --------------------------------------------------------------------------- #
@dataclass
class Citation:
    """Resolución de un chunk_id a una referencia trazable a la fuente."""
    chunk_id: str
    doc_id: str
    filename: str
    page: Optional[int] = None
    section_path: list[str] = field(default_factory=list)
    quote: str = ""

    def label(self) -> str:
        pg = f" pág.{self.page}" if self.page is not None else ""
        return f"[📄 {self.filename}{pg}]"


# --------------------------------------------------------------------------- #
# Capa 1 — Ingesta
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    block_id: str
    type: BlockType
    text: str
    level: Optional[int] = None           # solo headings (1..6)
    page: Optional[int] = None
    confidence: Optional[float] = None     # baja confianza -> revisión humana
    bbox: Optional[list[float]] = None     # [x, y, w, h]


@dataclass
class Document:
    doc_id: str
    source_path: str
    format: DocFormat
    is_scanned: bool
    blocks: list[Block] = field(default_factory=list)
    title: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    ocr_confidence: Optional[float] = None

    def low_confidence_blocks(self, threshold: float = 0.6) -> list[Block]:
        return [b for b in self.blocks
                if b.confidence is not None and b.confidence < threshold]


# --------------------------------------------------------------------------- #
# Capa 2 — Comprensión
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section_path: list[str] = field(default_factory=list)
    page: Optional[int] = None
    embedding: Optional[list[float]] = None
    concepts: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)


@dataclass
class Concept:
    concept_id: str
    label: str
    definition: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)
    related: list[tuple[str, Relation]] = field(default_factory=list)

    def is_grounded(self) -> bool:
        """Regla dura del contrato: todo concepto se ancla a evidencia."""
        return bool(self.source_chunks)


@dataclass
class Author:
    author_id: str
    name: str
    theses: list[str] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)


@dataclass
class Edge:
    src: str
    dst: str
    type: Relation
    evidence: str           # chunk_id — obligatorio

    def is_grounded(self) -> bool:
        return bool(self.evidence)


# --------------------------------------------------------------------------- #
# Capa 3 — Motor de estudio
# --------------------------------------------------------------------------- #
@dataclass
class StudyRequest:
    corpus_ids: list[str]
    focus: Focus = Focus.ALL
    topic: Optional[str] = None
    key_concepts: list[str] = field(default_factory=list)
    key_authors: list[str] = field(default_factory=list)
    difficulty: int = 3     # 1..5


@dataclass
class Question:
    q_id: str
    type: QuestionType
    prompt: str
    answer_key: str
    rationale: str
    evidence: list[str]                  # [chunk_id] — OBLIGATORIO
    concept_ids: list[str] = field(default_factory=list)
    options: Optional[list[str]] = None  # solo MCQ
    difficulty: int = 3

    def is_grounded(self) -> bool:
        """Regla dura: sin evidencia, la pregunta no es válida."""
        return bool(self.evidence)


@dataclass
class GradeResult:
    q_id: str
    user_answer: str
    score: float            # 0..1
    verdict: str
    feedback: str
    evidence: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)   # [concept_id] a reforzar


# --------------------------------------------------------------------------- #
# Capa 4 — Estructuras pedagógicas
# --------------------------------------------------------------------------- #
@dataclass
class ConceptMap:
    nodes: list[str]                         # [concept_id]
    edges: list[Edge] = field(default_factory=list)
    render: str = "mermaid"                  # mermaid | graphviz


@dataclass
class Hierarchy:
    root_concept: str
    levels: list[list[str]] = field(default_factory=list)   # BFS por niveles
    by_author: Optional[dict] = None         # {author_id: [concept_id]}


@dataclass
class Comparison:
    concept_ids: list[str]                   # típicamente 2
    dimensions: list[dict] = field(default_factory=list)
    # cada dimension: {dimension, a_value, b_value, evidence:[chunk_id]}  evidence NO vacía
    summary: str = ""

    def is_grounded(self) -> bool:
        return bool(self.dimensions) and all(d.get("evidence") for d in self.dimensions)


@dataclass
class Debate:
    topic: str
    positions: list[dict] = field(default_factory=list)
    # cada position: {author, stance, evidence:[chunk_id]}  evidence NO vacía
    synthesis: str = ""

    def is_grounded(self) -> bool:
        return bool(self.positions) and all(p.get("evidence") for p in self.positions)


# --------------------------------------------------------------------------- #
# Capa 5 — Memoria / FSRS
# --------------------------------------------------------------------------- #
@dataclass
class Card:
    card_id: str
    concept_id: str
    front: str
    back: str
    due: Optional[str] = None        # ISO 8601; None = nunca repasada
    interval: int = 0                # días
    ease: float = 2.5                # factor de facilidad (FSRS/SM-2-lite)
    lapses: int = 0                  # veces fallada
    reps: int = 0                    # repasos correctos consecutivos
    last_review: Optional[str] = None
    difficulty: int = 3              # dificultad de la card (1..5)


@dataclass
class Exam:
    exam_id: str
    corpus_ids: list[str]
    questions: list[Question] = field(default_factory=list)
    time_limit: Optional[int] = None     # segundos; None = sin límite
    coverage_target: float = 0.85        # ≥85% de conceptos clave
    is_submitted: bool = False


# --------------------------------------------------------------------------- #
# Capa 6 — Aprendizaje adaptativo
# --------------------------------------------------------------------------- #
@dataclass
class MasteryUpdate:
    concept_id: str
    prev_level: float
    new_level: float
    trigger: str
    action: str


@dataclass
class ErrorCluster:
    cluster_id: str
    concept_ids: list[str]
    error_type: str
    frequency: int = 0
    recommended_action: str = ""
