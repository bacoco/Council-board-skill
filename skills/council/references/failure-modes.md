# Council Failure Modes

Explicit documentation of what happens when things go wrong.

## CLI Failures

### Timeout (>60s default)

**Trigger**: Model doesn't respond within configured timeout.

**Behavior**:
1. Request marked as `TIMEOUT`
2. Circuit breaker records failure
3. Model excluded from current round
4. Session continues with remaining models
5. Confidence adjusted for degradation

**Output**:
```json
{"type": "opinion_error", "model": "gemini", "error": "TIMEOUT", "round": 1}
{"type": "model_degraded", "model": "gemini", "reason": "TIMEOUT", "level": "degraded"}
```

**User sees**: "Gemini unavailable (timeout). Results based on 2/3 perspectives."

---

### Rate Limiting (429)

**Trigger**: Provider returns HTTP 429 or rate limit message.

**Behavior**:
1. Classified as transient error → retry with exponential backoff
2. Base delay: 2s, doubles each retry (2s → 4s → 8s)
3. Max 3 retries before marking as failed
4. If all retries fail, model excluded

**Output**:
```json
{"type": "query_retry_backoff", "model": "claude", "attempt": 2, "delay_ms": 4000}
```

---

### Authentication Error (401/403)

**Trigger**: CLI not authenticated or token expired.

**Behavior**:
1. Classified as **permanent error** → NO retry
2. Immediate failure, circuit breaker opens
3. Model excluded for session duration

**Output**:
```json
{"type": "query_permanent_error", "model": "codex", "error": "auth"}
{"type": "circuit_breaker", "model": "codex", "state": "open"}
```

**User action required**: Re-authenticate CLI (`claude auth login`, etc.)

---

### Network Error

**Trigger**: DNS failure, connection refused, network unreachable.

**Behavior**:
1. Classified as transient → retry with backoff
2. After retries exhausted, model excluded
3. May recover if network restored mid-session

**Output**:
```json
{"type": "opinion_error", "model": "gemini", "error": "connection refused"}
```

---

### CLI Not Found

**Trigger**: CLI binary not installed or not in PATH.

**Behavior**:
1. Detected at startup via `--check`
2. Model permanently unavailable for session
3. No retry attempted

**Output** (from `--check`):
```
║  ✗ gemini   │ NOT INSTALLED                                ║
║             │ → CLI not found. Install: npm install -g @google/gemini-cli
```

---

## Quorum Failures

### Below Minimum (< 2 models)

**Trigger**: Fewer than `min_quorum` models respond in a round.

**Behavior**:
1. Round marked as failed
2. Session cannot continue
3. Partial results returned with warning
4. Confidence severely penalized (25%)

**Output**:
```json
{"type": "quorum_failure", "required": 2, "available": 1, "round": 1}
{"type": "degradation_status", "level": "minimal", "msg": "Below quorum"}
```

**User sees**: "Council cannot reach quorum. Only 1/3 models available. Results unreliable."

---

## Parsing Failures

### Invalid JSON Response

**Trigger**: Model returns malformed JSON or plain text when JSON expected.

**Behavior**:
1. Attempt to extract JSON from response using regex
2. If extraction fails, use raw text as content
3. Response scored lower in peer review
4. Session continues

**Output**:
```json
{"type": "json_parse_warning", "model": "codex", "fallback": "raw_text"}
```

---

### Missing Required Fields

**Trigger**: JSON valid but missing expected fields (confidence, opinion, etc.)

**Behavior**:
1. Use sensible defaults (confidence: 0.5)
2. Log warning
3. Session continues with partial data

---

## Circuit Breaker States

| State | Meaning | Behavior |
|-------|---------|----------|
| **CLOSED** | Normal operation | All requests allowed |
| **OPEN** | 3+ consecutive failures | All requests blocked for 60s |
| **HALF_OPEN** | Recovery testing | Allow 1 request to test recovery |

### Recovery Flow

```
CLOSED --[3 failures]--> OPEN --[60s wait]--> HALF_OPEN --[success]--> CLOSED
                                                      --[failure]--> OPEN
```

---

## Degradation Levels

| Level | Condition | Confidence Penalty | Can Continue? |
|-------|-----------|-------------------|---------------|
| **FULL** | All 3 models available | 0% | Yes |
| **DEGRADED** | 2 models available | 10% | Yes |
| **MINIMAL** | 1 model available | 25% | No (quorum fail) |

---

## Recovery Actions

### Immediate (User)

1. **Check setup**: `python3 council.py --check`
2. **Re-authenticate**: `claude auth login`, `gemini auth login`, `codex auth`
3. **Check network**: Verify connectivity to API endpoints

### Automatic (Council)

1. **Retry transient errors** with exponential backoff
2. **Failover chairman** to alternate model
3. **Exclude failing models** via circuit breaker
4. **Adjust confidence** for degradation level

### Manual Recovery

If circuit breaker is stuck open:
1. Wait 60 seconds for automatic recovery
2. Or restart Council session (resets circuit breaker state)
