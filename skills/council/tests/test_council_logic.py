import asyncio
import json
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import council
from scripts.council import (
    LLMResponse,
    Persona,
    SessionConfig,
    check_convergence,
    run_adaptive_cascade,
    gather_opinions,
    generate_personas_with_llm,
)


def test_check_convergence_combines_confidence_and_signal():
    """Convergence should trigger when confidence and signals are high."""
    round_history = [
        {
            "claude": json.dumps({"confidence": 0.6, "convergence_signal": False}),
            "gemini": json.dumps({"confidence": 0.55, "convergence_signal": False}),
        },
        {
            "claude": json.dumps({"confidence": 0.9, "convergence_signal": True}),
            "gemini": json.dumps({"confidence": 0.8, "convergence_signal": True}),
        },
    ]

    converged, score = check_convergence(round_history)

    assert converged is True
    # Score blends average confidence (0.85) and full convergence signal (1.0)
    assert pytest.approx(score, rel=1e-3) == 0.91


def test_generate_personas_uses_persona_library_cache(monkeypatch):
    """Invalid LLM persona output should fall back to cached PersonaManager personas."""

    async def _run():
        async def fake_adapter(prompt: str, timeout: int):
            return LLMResponse(
                content=json.dumps({"raw": "not json array"}),
                model="claude",
                latency_ms=1,
                success=True,
            )

        fallback_calls = []

        def fake_assign_personas(query: str, num_models: int):
            fallback_calls.append((query, num_models))
            return [
                Persona(
                    title="Cached Persona",
                    role="Fallback",
                    prompt_prefix="prefix",
                    specializations=[],
                )
                for _ in range(num_models)
            ]

        monkeypatch.setattr(council, "ADAPTERS", {"claude": fake_adapter})
        monkeypatch.setattr(
            council.PERSONA_MANAGER, "assign_personas", fake_assign_personas
        )

        personas = await generate_personas_with_llm(
            "Test query", num_models=2, chairman="claude", mode="consensus", timeout=5
        )

        return personas, fallback_calls

    personas, fallback_calls = asyncio.run(_run())

    assert fallback_calls == [("Test query", 2)]
    assert [p.title for p in personas] == ["Cached Persona", "Cached Persona"]


def test_gather_opinions_uses_adaptive_timeout(monkeypatch):
    """Gathering opinions should request timeouts from the adaptive manager per model."""

    async def _run():
        timeouts_used = []

        async def fake_adapter(prompt: str, timeout: int):
            timeouts_used.append(timeout)
            return LLMResponse(
                content=json.dumps(
                    {
                        "answer": "ok",
                        "key_points": [],
                        "confidence": 0.9,
                        "convergence_signal": True,
                    }
                ),
                model="claude",
                latency_ms=10,
                success=True,
            )

        async def fake_generate_personas_with_llm(*args, **kwargs):
            num_models = args[1]
            return [
                Persona(
                    title=f"P{i}",
                    role="role",
                    prompt_prefix="prefix",
                    specializations=[],
                )
                for i in range(num_models)
            ]

        class FakeAdaptiveTimeout:
            def __init__(self):
                self.requested = []

            def get_timeout(self, model: str) -> int:
                self.requested.append(model)
                return 7 if model == "claude" else 9

            def record_latency(self, model: str, latency_ms: int, success: bool = True):
                pass

            def get_stats(self):
                return {}

        fake_timeout = FakeAdaptiveTimeout()

        monkeypatch.setattr(
            council, "ADAPTERS", {"claude": fake_adapter, "gemini": fake_adapter}
        )
        monkeypatch.setattr(council, "check_cli_available", lambda model: True)
        monkeypatch.setattr(
            council, "generate_personas_with_llm", fake_generate_personas_with_llm
        )
        council.ADAPTIVE_TIMEOUT = fake_timeout
        council.DEGRADATION_STATE = council.DegradationState(["claude", "gemini"])

        config = SessionConfig(
            query="Q?",
            mode="consensus",
            models=["claude", "gemini"],
            chairman="claude",
            timeout=30,
            anonymize=False,
            council_budget="balanced",
            output_level="standard",
            max_rounds=1,
        )

        responses = await gather_opinions(
            config, round_num=1, previous_round_opinions=None
        )

        return timeouts_used, responses

    timeouts_used, responses = asyncio.run(_run())

    assert timeouts_used == [7, 9]
    assert all(response.success for response in responses.values())


def test_adaptive_cascade_escalates_to_devils_advocate(monkeypatch):
    """Adaptive cascade should escalate to debate then devil's advocate when convergence is low."""

    async def _run():
        calls = []

        async def fake_run_council(config: SessionConfig):
            calls.append((config.mode, config.timeout))
            if config.mode == "consensus":
                return {"convergence_score": 0.2, "synthesis": {"confidence": 0.4}}
            if config.mode == "debate":
                return {"convergence_score": 0.3, "synthesis": {"confidence": 0.5}}
            return {"convergence_score": 0.95, "synthesis": {"confidence": 0.96}}

        async def fake_meta_synthesize(*args, **kwargs):
            return {"final_answer": "meta", "confidence": 0.9}

        monkeypatch.setattr(council, "run_council", fake_run_council)
        monkeypatch.setattr(council, "meta_synthesize", fake_meta_synthesize)

        config = SessionConfig(
            query="Need consensus",
            mode="adaptive",
            models=["claude", "gemini", "codex"],
            chairman="claude",
            timeout=25,
            anonymize=True,
            council_budget="balanced",
            output_level="standard",
            max_rounds=2,
        )

        result = await run_adaptive_cascade(config)

        return calls, result

    calls, result = asyncio.run(_run())

    assert [mode for mode, _ in calls] == ["consensus", "debate", "devil_advocate"]
    assert all(timeout == 25 for _, timeout in calls)
    assert result["modes"] == ["consensus", "debate", "devil's advocate"]
    assert result["final_answer"] == "meta"
