#!/usr/bin/env python3
"""
LLM Council - Multi-model deliberation orchestrator.
CLI entry point for council deliberations.
"""

import argparse
import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

# Import from parent package
sys.path.insert(0, str(Path(__file__).parent.parent))
from providers import CouncilConfig
from security.input_validator import validate_and_sanitize

# Import from core modules
from core.models import SessionConfig
from core.emit import emit, set_output_mode
from core.adapters import expand_models_with_fallback, ADAPTERS, check_cli_available

# Import mode implementations (used by classic pipeline, kept for direct access)
from modes.consensus import run_council
from modes.vote import run_vote_council
from modes.adaptive import run_adaptive_cascade

# Import pipeline implementations
from pipelines import get_pipeline, PIPELINES


# ============================================================================
# Configuration Loading
# ============================================================================

def load_council_config_defaults() -> CouncilConfig:
    """Load defaults from council.config.yaml if available."""
    try:
        return CouncilConfig.from_file(CouncilConfig.default_path())
    except Exception:
        # Fall back to in-code defaults if config is unreadable
        return CouncilConfig()


# ============================================================================
# Setup Validation
# ============================================================================

def check_setup() -> dict:
    """
    Validate that all required CLIs are installed and working.
    Returns a dict with status for each CLI.
    """
    results = {
        'claude': {'installed': False, 'version': None, 'error': None},
        'gemini': {'installed': False, 'version': None, 'error': None},
        'codex': {'installed': False, 'version': None, 'error': None},
    }

    # Check Claude CLI
    try:
        result = subprocess.run(
            ['claude', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            results['claude']['installed'] = True
            results['claude']['version'] = result.stdout.strip().split('\n')[0]
        else:
            results['claude']['error'] = result.stderr.strip() or 'Unknown error'
    except FileNotFoundError:
        results['claude']['error'] = 'CLI not found. Install: npm install -g @anthropic-ai/claude-code'
    except subprocess.TimeoutExpired:
        results['claude']['error'] = 'Timeout checking CLI'
    except Exception as e:
        results['claude']['error'] = str(e)

    # Check Gemini CLI
    try:
        result = subprocess.run(
            ['gemini', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            results['gemini']['installed'] = True
            results['gemini']['version'] = result.stdout.strip().split('\n')[0]
        else:
            results['gemini']['error'] = result.stderr.strip() or 'Unknown error'
    except FileNotFoundError:
        results['gemini']['error'] = 'CLI not found. Install: npm install -g @google/gemini-cli'
    except subprocess.TimeoutExpired:
        results['gemini']['error'] = 'Timeout checking CLI'
    except Exception as e:
        results['gemini']['error'] = str(e)

    # Check Codex CLI
    try:
        result = subprocess.run(
            ['codex', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            results['codex']['installed'] = True
            results['codex']['version'] = result.stdout.strip().split('\n')[0]
        else:
            results['codex']['error'] = result.stderr.strip() or 'Unknown error'
    except FileNotFoundError:
        results['codex']['error'] = 'CLI not found. Install: npm install -g @openai/codex'
    except subprocess.TimeoutExpired:
        results['codex']['error'] = 'Timeout checking CLI'
    except Exception as e:
        results['codex']['error'] = str(e)

    return results


def print_setup_status(results: dict):
    """Print setup validation results in a user-friendly format."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║              COUNCIL SETUP VALIDATION                        ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    all_ok = True
    for cli, status in results.items():
        if status['installed']:
            icon = "✓"
            version = status['version'] or 'unknown'
            print(f"║  {icon} {cli:8} │ {version[:45]:<45} ║")
        else:
            icon = "✗"
            all_ok = False
            error = status['error'] or 'Unknown error'
            print(f"║  {icon} {cli:8} │ NOT INSTALLED                                ║")
            # Print error on next line if it exists
            if error:
                # Truncate error to fit
                error_short = error[:55] if len(error) <= 55 else error[:52] + '...'
                print(f"║             │ → {error_short:<43} ║")

    print("╠══════════════════════════════════════════════════════════════╣")

    installed_count = sum(1 for s in results.values() if s['installed'])

    if all_ok:
        print("║  STATUS: All CLIs ready ✓                                    ║")
        print("║  Council can use all 3 models for deliberation.              ║")
    elif installed_count >= 2:
        print(f"║  STATUS: Degraded mode ({installed_count}/3 CLIs available)                      ║")
        print("║  Council will work with reduced confidence.                  ║")
    else:
        print(f"║  STATUS: Cannot run ({installed_count}/3 CLIs available)                         ║")
        print("║  Council requires at least 2 CLIs. Install missing ones.     ║")

    print("╚══════════════════════════════════════════════════════════════╝\n")

    return all_ok


# ============================================================================
# Context Loading
# ============================================================================

def load_context_from_file(context_path: Path, existing_context: str = '') -> str:
    """
    Load context from a manifest file, including any referenced files.

    Args:
        context_path: Path to manifest file
        existing_context: Existing inline context to prepend

    Returns:
        Combined context string
    """
    if not context_path.exists():
        emit({'type': 'context_error', 'file': str(context_path), 'error': 'Manifest file not found'})
        return existing_context

    try:
        manifest_content = context_path.read_text(encoding='utf-8')

        # Parse manifest for file paths (lines starting with ### followed by a path)
        file_pattern = re.compile(r'^###\s+(\S+\.(?:py|ts|js|tsx|jsx|go|rs|java|md|json|yaml|yml|toml))\s*$', re.MULTILINE)
        file_paths = file_pattern.findall(manifest_content)

        # Build context from manifest + loaded files
        context_parts = [manifest_content]

        if file_paths:
            context_parts.append("\n\n# === LOADED FILES ===\n")
            files_loaded = []
            files_blocked = []

            # SECURITY: Scope file paths to manifest directory (prevent arbitrary file reads)
            allowed_base = context_path.parent.resolve()
            max_file_size = 100000  # 100KB per file limit

            for file_path in file_paths:
                fp = Path(file_path)

                # Handle relative paths (resolve relative to manifest directory)
                if not fp.is_absolute():
                    fp = (allowed_base / fp).resolve()
                else:
                    fp = fp.resolve()

                # SECURITY: Path scoping - only allow files within manifest directory tree
                try:
                    fp.relative_to(allowed_base)
                except ValueError:
                    # Path is outside allowed directory
                    files_blocked.append(file_path)
                    context_parts.append(f"\n## File: {file_path}\n[BLOCKED: Path outside allowed scope]\n")
                    continue

                # SECURITY: Prevent symlink attacks
                if fp.is_symlink():
                    real_path = fp.resolve()
                    try:
                        real_path.relative_to(allowed_base)
                    except ValueError:
                        files_blocked.append(file_path)
                        context_parts.append(f"\n## File: {file_path}\n[BLOCKED: Symlink points outside allowed scope]\n")
                        continue

                if fp.exists():
                    try:
                        # SECURITY: Check file size before reading
                        file_size = fp.stat().st_size
                        if file_size > max_file_size:
                            context_parts.append(f"\n## File: {file_path}\n[BLOCKED: File too large ({file_size} > {max_file_size} bytes)]\n")
                            continue

                        content = fp.read_text(encoding='utf-8')
                        context_parts.append(f"\n## File: {file_path}\n```\n{content}\n```\n")
                        files_loaded.append(file_path)
                    except Exception as e:
                        context_parts.append(f"\n## File: {file_path}\n[Error reading: {e}]\n")
                else:
                    context_parts.append(f"\n## File: {file_path}\n[File not found]\n")

            emit({
                'type': 'context_loaded',
                'manifest': str(context_path),
                'files_loaded': files_loaded,
                'files_blocked': files_blocked
            })
        else:
            # No file paths found, just use manifest as context
            emit({'type': 'context_loaded', 'manifest': str(context_path), 'files_loaded': []})

        # Combine all parts
        loaded_context = ''.join(context_parts)
        if existing_context:
            return f"{existing_context}\n\n{loaded_context}"
        return loaded_context

    except Exception as e:
        emit({'type': 'context_error', 'file': str(context_path), 'error': str(e)})
        return existing_context


# ============================================================================
# Direct Mode
# ============================================================================

async def run_direct(models: list, query: str, timeout: int, human: bool):
    """
    Call models directly without deliberation, show raw responses.

    Sequential execution for ordered output. No personas, no synthesis,
    no peer review - just raw CLI responses.
    """
    for model in models:
        model = model.strip()

        # Check availability
        if not check_cli_available(model):
            if human:
                print(f"\n❌ {model.upper()}: CLI not available")
            else:
                print(json.dumps({
                    "model": model,
                    "success": False,
                    "error": "CLI not available"
                }))
            continue

        # Query the model
        try:
            response = await ADAPTERS[model](query, timeout)

            if human:
                print(f"\n{'='*60}")
                print(f"🤖 {model.upper()} ({response.latency_ms}ms)")
                print(f"{'='*60}")
                if response.success:
                    print(response.content)
                else:
                    print(f"Error: {response.error}")
            else:
                print(json.dumps({
                    "model": model,
                    "success": response.success,
                    "latency_ms": response.latency_ms,
                    "content": response.content if response.success else None,
                    "error": response.error
                }))
        except Exception as e:
            if human:
                print(f"\n❌ {model.upper()}: {str(e)}")
            else:
                print(json.dumps({
                    "model": model,
                    "success": False,
                    "error": str(e)
                }))


# ============================================================================
# CLI
# ============================================================================

def main():
    config_defaults = load_council_config_defaults()

    parser = argparse.ArgumentParser(description='LLM Council - Multi-model deliberation')
    parser.add_argument('--check', action='store_true', help='Validate setup (test all CLIs)')
    parser.add_argument('--query', '-q', help='Question to deliberate')
    parser.add_argument('--context', '-c', help='Code or additional context for analysis (optional)')
    parser.add_argument('--context-file', '-f', help='Path to file containing context (code, docs, etc.)')
    parser.add_argument('--mode', '-m', default='adaptive',
                       choices=['adaptive', 'consensus', 'debate', 'vote', 'devil_advocate',
                                'storm_decision', 'storm_research', 'storm_review'])
    parser.add_argument('--pipeline', '-p', default=None,
                       choices=['classic', 'storm'],
                       help='Pipeline to use: classic (original) or storm (STORM-inspired). '
                            'Auto-detected from mode if not specified. Default from config.')
    parser.add_argument('--models', default='claude,gemini,codex', help='Comma-separated model list')
    parser.add_argument('--chairman', default='claude', help='Synthesizer model')
    parser.add_argument('--timeout', type=int, default=config_defaults.timeout, help='Per-model timeout (seconds)')
    parser.add_argument('--anonymize', action=argparse.BooleanOptionalAction, default=True,
                        help='Anonymize responses (use --no-anonymize to disable)')
    parser.add_argument('--budget', default='balanced', choices=['fast', 'balanced', 'thorough'])
    parser.add_argument('--output', default='standard', choices=['minimal', 'standard', 'audit'])
    parser.add_argument('--max-rounds', type=int, default=config_defaults.max_rounds, help='Max rounds for deliberation')
    parser.add_argument(
        '--enable-perf-metrics',
        action=argparse.BooleanOptionalAction,
        default=config_defaults.enable_perf_metrics,
        help='Emit performance metrics events (latencies by stage). Default comes from council.config.yaml (enable_perf_metrics).'
    )
    parser.add_argument(
        '--trail',
        action=argparse.BooleanOptionalAction,
        default=config_defaults.enable_trail,
        help='Include deliberation trail in output (who said what). Default comes from council.config.yaml (enable_trail). Use --no-trail to disable.'
    )
    parser.add_argument(
        '--human',
        action='store_true',
        default=False,
        help='Human-readable output instead of JSON (recommended for interactive use)'
    )
    parser.add_argument(
        '--direct',
        action='store_true',
        default=False,
        help='Call models directly without deliberation, show raw responses'
    )

    args = parser.parse_args()

    # Apply output mode settings globally
    set_output_mode(human=args.human, perf=args.enable_perf_metrics)

    # ============================================================================
    # Setup Check Mode
    # ============================================================================

    if args.check:
        results = check_setup()
        all_ok = print_setup_status(results)
        sys.exit(0 if all_ok else 1)

    # ============================================================================
    # Direct Mode: Call models directly without deliberation
    # ============================================================================

    if args.direct:
        if not args.query:
            parser.error("--query is required for --direct mode")

        # SECURITY: Validate input before direct mode (fixes validation bypass)
        validation = validate_and_sanitize(
            query=args.query,
            context=None,
            max_rounds=1,
            timeout=args.timeout,
            strict=True
        )
        if not validation['is_valid']:
            emit({
                'type': 'error',
                'msg': 'Input validation failed - request rejected',
                'violations': validation['violations']
            })
            sys.exit(1)

        models = [m.strip() for m in args.models.split(',')]
        asyncio.run(run_direct(models, validation['query'], args.timeout, args.human))
        sys.exit(0)

    # Validate --query is provided when not in check mode
    if not args.query:
        parser.error("--query is required (use --check to validate setup instead)")

    # ============================================================================
    # Context Loading: File or inline
    # ============================================================================

    context = args.context or ''

    # Load context from manifest file if specified
    if args.context_file:
        context_path = Path(args.context_file).resolve()
        context = load_context_from_file(context_path, context)

    # ============================================================================
    # SECURITY: Input Validation and Sanitization
    # ============================================================================

    validation = validate_and_sanitize(
        query=args.query,
        context=context,  # Use loaded context
        max_rounds=args.max_rounds,
        timeout=args.timeout,
        strict=True  # SECURITY: Block injection attempts, don't just warn
    )

    # Fail if query contains injection patterns
    if not validation['is_valid']:
        emit({
            'type': 'error',
            'msg': 'Input validation failed - request rejected',
            'violations': validation['violations']
        })
        sys.exit(1)

    # Emit any redacted secrets as info
    if validation['redacted_secrets']:
        emit({
            'type': 'info',
            'msg': f"Redacted {len(validation['redacted_secrets'])} secrets from context"
        })

    # Apply fallback for unavailable models
    requested_models = args.models.split(',')
    expanded_models = expand_models_with_fallback(requested_models, min_models=3)

    config = SessionConfig(
        query=validation['query'],  # SANITIZED QUERY
        mode=args.mode,
        models=expanded_models,
        chairman=args.chairman,
        timeout=validation['timeout'],  # VALIDATED TIMEOUT
        anonymize=args.anonymize,
        council_budget=args.budget,
        output_level=args.output,
        max_rounds=validation['max_rounds'],  # VALIDATED MAX_ROUNDS
        min_quorum=config_defaults.min_quorum,  # From council.config.yaml
        convergence_threshold=config_defaults.convergence_threshold,  # From council.config.yaml
        enable_perf_metrics=args.enable_perf_metrics,
        enable_trail=args.trail,
        context=validation['context']  # REDACTED CONTEXT
    )

    # Pipeline selection logic:
    # 1. Explicit --pipeline flag takes priority
    # 2. STORM modes (storm_*) auto-select storm pipeline
    # 3. Fall back to config default (usually 'classic')
    STORM_MODES = {'storm_decision', 'storm_research', 'storm_review'}

    if args.pipeline:
        pipeline_name = args.pipeline
    elif args.mode in STORM_MODES:
        pipeline_name = 'storm'
    else:
        pipeline_name = config_defaults.pipeline

    # Get pipeline class and instantiate
    PipelineClass = get_pipeline(pipeline_name)
    pipeline = PipelineClass(config)

    # Execute pipeline
    pipeline_result = asyncio.run(pipeline.run())

    # Convert PipelineResult to dict for output
    result = pipeline_result.raw_result or {
        'answer': pipeline_result.answer,
        'confidence': pipeline_result.confidence,
        'mode': pipeline_result.mode_used,
        'rounds': pipeline_result.rounds,
        'trail_file': pipeline_result.trail_file,
        'pipeline': pipeline_result.pipeline,
    }

    # Add STORM-specific fields if present
    if pipeline_result.knowledge_base:
        result['knowledge_base'] = pipeline_result.knowledge_base
    if pipeline_result.evidence_coverage is not None:
        result['evidence_coverage'] = pipeline_result.evidence_coverage
    if pipeline_result.unresolved_objections:
        result['unresolved_objections'] = pipeline_result.unresolved_objections

    if config.output_level == 'audit':
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
