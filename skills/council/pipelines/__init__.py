"""
Pipeline implementations for Council deliberation.

Pipelines define the high-level orchestration strategy:
- classic: Original gather → peer-review → synthesis flow
- storm: STORM-inspired Moderator-led flow with KnowledgeBase and evidence grounding
"""

from .base import Pipeline, PipelineResult
from .classic import ClassicPipeline
from .storm import StormPipeline

# Pipeline registry
PIPELINES = {
    'classic': ClassicPipeline,
    'storm': StormPipeline,
}


def get_pipeline(name: str) -> type:
    """Get pipeline class by name."""
    if name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {name}. Available: {list(PIPELINES.keys())}")
    return PIPELINES[name]


__all__ = ['Pipeline', 'PipelineResult', 'ClassicPipeline', 'StormPipeline', 'PIPELINES', 'get_pipeline']
