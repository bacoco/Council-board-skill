#!/usr/bin/env python3
"""
Council API Server - HTTP/WebSocket interface for multi-model deliberation.

Provides REST endpoints and WebSocket streaming for the Council skill.
Designed to be imported by Claude Code skills or run standalone.

Usage:
    # Start server
    python3 api_server.py --port 8080

    # Or import and run programmatically
    from api_server import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

Endpoints:
    POST /council          - Run council deliberation (blocking)
    POST /council/stream   - Run council with SSE streaming
    WS   /council/ws       - WebSocket for real-time events
    GET  /health           - Health check
    GET  /models           - List available models
"""

import argparse
import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List, AsyncGenerator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# FastAPI imports (with graceful fallback)
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

# Import council components
from council import (
    SessionConfig,
    run_council,
    run_vote_council,
    run_adaptive_cascade,
    expand_models_with_fallback,
    get_available_models,
    check_cli_available,
    CIRCUIT_BREAKER,
    DEFAULT_TIMEOUT,
    MIN_QUORUM,
)
from security.input_validator import validate_and_sanitize

# ============================================================================
# Pydantic Models for API
# ============================================================================

class CouncilRequest(BaseModel):
    """Request body for council deliberation."""
    query: str = Field(..., description="Question or topic to deliberate")
    context: Optional[str] = Field(None, description="Code or additional context")
    mode: str = Field("adaptive", description="Deliberation mode: adaptive, consensus, debate, vote, devil_advocate")
    models: List[str] = Field(["claude", "gemini", "codex"], description="Models to use")
    chairman: str = Field("claude", description="Chairman model for synthesis")
    timeout: int = Field(DEFAULT_TIMEOUT, description="Timeout per model in seconds")
    max_rounds: int = Field(3, description="Maximum deliberation rounds")
    anonymize: bool = Field(True, description="Anonymize responses in peer review")


class CouncilResponse(BaseModel):
    """Response from council deliberation."""
    session_id: str
    mode: str
    answer: str
    confidence: float
    raw_confidence: Optional[float] = None
    convergence_score: Optional[float] = None
    rounds_completed: Optional[int] = None
    degradation_level: str = "full"
    duration_ms: int
    models_responded: List[str]
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models: dict
    circuit_breaker: dict
    version: str = "1.0.0"


class ModelsResponse(BaseModel):
    """Available models response."""
    available: List[str]
    requested: List[str]
    circuit_breaker_status: dict


# ============================================================================
# Event Capture for Streaming
# ============================================================================

class EventCapture:
    """Captures council events for streaming."""

    def __init__(self):
        self.events: List[dict] = []
        self.subscribers: List[asyncio.Queue] = []

    def add_subscriber(self) -> asyncio.Queue:
        """Add a new subscriber queue."""
        queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    def remove_subscriber(self, queue: asyncio.Queue):
        """Remove a subscriber queue."""
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    async def emit(self, event: dict):
        """Emit event to all subscribers."""
        self.events.append(event)
        for queue in self.subscribers:
            await queue.put(event)

    async def emit_done(self):
        """Signal completion to all subscribers."""
        for queue in self.subscribers:
            await queue.put(None)


# Global event capture (replaced per request)
_event_capture: Optional[EventCapture] = None


def _patched_emit(event: dict):
    """Patched emit function that captures events."""
    import council
    # Call original emit
    event['ts'] = int(time.time())
    redacted = council.INPUT_VALIDATOR.redact_output(event)
    print(json.dumps(redacted), flush=True)

    # Also send to capture if active
    if _event_capture:
        asyncio.create_task(_event_capture.emit(redacted))


# ============================================================================
# API Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Council API Server starting...")
    yield
    # Shutdown
    print("Council API Server shutting down...")


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Council API",
        description="Multi-model deliberation API for Claude, Gemini, and Codex",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS middleware for browser access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app = None


# ============================================================================
# Helper Functions
# ============================================================================

def build_config(request: CouncilRequest) -> SessionConfig:
    """Build SessionConfig from API request."""
    # Validate and sanitize input
    validation = validate_and_sanitize(
        query=request.query,
        context=request.context,
        max_rounds=request.max_rounds,
        timeout=request.timeout,
        strict=False
    )

    if not validation['is_valid']:
        raise HTTPException(status_code=400, detail=f"Validation failed: {validation['violations']}")

    # Expand models with fallback
    expanded_models = expand_models_with_fallback(request.models, min_models=3)

    return SessionConfig(
        query=validation['query'],
        mode=request.mode,
        models=expanded_models,
        chairman=request.chairman,
        timeout=validation['timeout'],
        anonymize=request.anonymize,
        council_budget='balanced',
        output_level='standard',
        max_rounds=validation['max_rounds'],
        context=validation['context']
    )


async def run_deliberation(config: SessionConfig) -> dict:
    """Run the appropriate deliberation mode."""
    if config.mode == 'adaptive':
        return await run_adaptive_cascade(config)
    elif config.mode == 'vote':
        return await run_vote_council(config)
    else:
        return await run_council(config)


def extract_response(result: dict, mode: str) -> CouncilResponse:
    """Extract standardized response from result."""
    # Handle different result structures
    if 'error' in result:
        return CouncilResponse(
            session_id=result.get('session_id', 'unknown'),
            mode=mode,
            answer="",
            confidence=0.0,
            duration_ms=result.get('duration_ms', 0),
            models_responded=[],
            error=result['error'],
            degradation_level=result.get('degradation', {}).get('level', 'unknown')
        )

    # Adaptive cascade has nested structure
    if 'meta_synthesis' in result:
        synthesis = result.get('meta_synthesis', {})
        answer = synthesis.get('final_answer', result.get('final_answer', ''))
        confidence = synthesis.get('confidence', result.get('confidence', 0.0))
    else:
        synthesis = result.get('synthesis', {})
        answer = synthesis.get('final_answer', '')
        confidence = result.get('confidence', synthesis.get('confidence', 0.0))

    # Extract models responded
    if 'vote_result' in result:
        models = [b['model'] for b in result['vote_result'].get('ballots', [])]
    else:
        models = list(result.get('final_opinions', {}).keys())

    return CouncilResponse(
        session_id=result.get('session_id', 'unknown'),
        mode=mode,
        answer=answer,
        confidence=confidence,
        raw_confidence=result.get('raw_confidence'),
        convergence_score=result.get('convergence_score'),
        rounds_completed=result.get('rounds_completed'),
        degradation_level=result.get('degradation', {}).get('level', 'full'),
        duration_ms=result.get('duration_ms', 0),
        models_responded=models
    )


# ============================================================================
# API Endpoints
# ============================================================================

if FASTAPI_AVAILABLE:

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        models = {
            'claude': check_cli_available('claude'),
            'gemini': check_cli_available('gemini'),
            'codex': check_cli_available('codex'),
        }
        return HealthResponse(
            status="healthy" if any(models.values()) else "degraded",
            models=models,
            circuit_breaker=CIRCUIT_BREAKER.get_status()
        )


    @app.get("/models", response_model=ModelsResponse)
    async def list_models():
        """List available models."""
        requested = ['claude', 'gemini', 'codex']
        available = get_available_models(requested)
        return ModelsResponse(
            available=available,
            requested=requested,
            circuit_breaker_status=CIRCUIT_BREAKER.get_status()
        )


    @app.post("/council", response_model=CouncilResponse)
    async def run_council_endpoint(request: CouncilRequest):
        """
        Run council deliberation (blocking).

        Returns the final result after all rounds complete.
        """
        try:
            config = build_config(request)
            result = await run_deliberation(config)
            return extract_response(result, request.mode)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    @app.post("/council/stream")
    async def run_council_stream(request: CouncilRequest):
        """
        Run council deliberation with Server-Sent Events streaming.

        Streams events as they occur, then sends final result.
        """
        global _event_capture

        async def event_generator() -> AsyncGenerator[str, None]:
            global _event_capture

            try:
                config = build_config(request)

                # Set up event capture
                _event_capture = EventCapture()
                queue = _event_capture.add_subscriber()

                # Start deliberation in background
                task = asyncio.create_task(run_deliberation(config))

                # Stream events as they arrive
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=1.0)
                        if event is None:
                            break
                        yield f"data: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        # Check if task completed
                        if task.done():
                            break
                        # Send keepalive
                        yield f": keepalive\n\n"

                # Get final result
                result = await task
                response = extract_response(result, request.mode)
                yield f"data: {json.dumps({'type': 'result', 'data': response.model_dump()})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            finally:
                _event_capture = None

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )


    @app.websocket("/council/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """
        WebSocket endpoint for real-time council deliberation.

        Send JSON request, receive events as they occur.
        """
        global _event_capture

        await websocket.accept()

        try:
            # Receive request
            data = await websocket.receive_json()
            request = CouncilRequest(**data)
            config = build_config(request)

            # Set up event capture
            _event_capture = EventCapture()
            queue = _event_capture.add_subscriber()

            # Start deliberation
            task = asyncio.create_task(run_deliberation(config))

            # Stream events
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    if event is None:
                        break
                    await websocket.send_json(event)
                except asyncio.TimeoutError:
                    if task.done():
                        break

            # Send final result
            result = await task
            response = extract_response(result, request.mode)
            await websocket.send_json({'type': 'result', 'data': response.model_dump()})
            await websocket.send_json({'type': 'done'})

        except WebSocketDisconnect:
            pass
        except Exception as e:
            await websocket.send_json({'type': 'error', 'error': str(e)})
        finally:
            _event_capture = None
            await websocket.close()


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Run the API server."""
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI not installed.")
        print("Run: pip install fastapi uvicorn")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Council API Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')

    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == '__main__':
    main()
