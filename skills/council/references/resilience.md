# Council Resilience Features

## Graceful Degradation

Council continues operating when models fail, adjusting confidence automatically.

### Degradation Levels

| Level | Condition | Confidence Penalty | Continue? |
|-------|-----------|-------------------|-----------|
| **FULL** | All models available | 0% | Yes |
| **DEGRADED** | Some failed, >=2 remaining | 10% | Yes |
| **MINIMAL** | <2 models (below quorum) | 25% | No |

### Confidence Adjustment

```
Raw confidence: 0.90

FULL:     0.90 x 1.00 = 0.90
DEGRADED: 0.90 x 0.90 = 0.81
MINIMAL:  0.90 x 0.75 = 0.675
```

## Adaptive Timeout

System learns from response times:

- **Initial**: Configured timeout (default 60s)
- **After 3+ samples**: p95 latency x 1.5 safety margin
- **Bounds**: 50%-200% of base timeout

```
Model: claude
Observed: [2000ms, 2500ms, 3000ms, 2200ms]
P95: ~3000ms
Adaptive: 3000ms x 1.5 = 4.5s -> clamped to 30s
```

## Circuit Breaker

Prevents cascading failures by excluding repeatedly failing models.

| State | Behavior |
|-------|----------|
| **CLOSED** | Normal operation |
| **OPEN** | Model skipped (3+ failures) |
| **HALF_OPEN** | Testing recovery (after 60s) |

Configuration:
- Failure threshold: 3
- Recovery timeout: 60s
- Success threshold: 2 (to close)

## Observable Events

```json
{"type": "model_degraded", "model": "gemini", "reason": "TIMEOUT", "level": "degraded"}
{"type": "model_recovered", "model": "gemini", "level": "full"}
{"type": "degradation_status", "level": "degraded", "available_models": ["claude", "codex"]}
```

## Output Fields

```json
{
  "confidence": 0.81,
  "raw_confidence": 0.90,
  "degradation": {
    "level": "degraded",
    "expected_models": ["claude", "gemini", "codex"],
    "available_models": ["claude", "codex"],
    "failed_models": {"gemini": "TIMEOUT"},
    "availability_ratio": 0.667
  },
  "circuit_breaker": {
    "claude": {"state": "closed", "failures": 0},
    "gemini": {"state": "open", "failures": 3}
  }
}
```
