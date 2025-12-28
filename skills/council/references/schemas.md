# JSON Schemas

Pydantic v2 strict mode models.

## Core Models

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from dataclasses import dataclass

class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True)

class Opinion(StrictModel):
    answer: str = Field(max_length=2000)
    key_points: list[str] = Field(max_length=5)
    assumptions: list[str]
    uncertainties: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    sources_if_known: list[str] = Field(default_factory=list)

class CriteriaScores(StrictModel):
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    reasoning: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)

class PeerReview(StrictModel):
    scores: dict[str, CriteriaScores]
    ranking: list[str]
    key_conflicts: list[str]
    uncertainties: list[str]
    notes: str = Field(max_length=500)

class Synthesis(StrictModel):
    final_answer: str
    contradiction_resolutions: list[dict]
    remaining_uncertainties: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    dissenting_view: str | None = None

@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int
    success: bool
    error: str | None = None
```

## NDJSON Event Types

| Type | Stage | Description |
|------|-------|-------------|
| `status` | * | Progress message |
| `routing` | 0 | Classification result |
| `opinion_start` | 1 | Model query started |
| `opinion_complete` | 1 | Model response received |
| `opinion_error` | 1 | Model failed (ABSTENTION) |
| `score` | 2 | Reviewer score |
| `contradiction` | 2.5 | Conflict detected |
| `token` | 3 | Chairman streaming |
| `final` | 3 | Complete response |
| `meta` | - | Session metrics |
