"""Phase 0 exit-criteria gates. If any of these fail, Phase 0 is not done."""


def test_voussoir_imports_cleanly():
    import voussoir

    assert voussoir.__version__


def test_container_resolves_default_llm_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from voussoir.container.defaults import default_container
    from voussoir.protocols import ILLMProvider

    c = default_container()
    binding_keys = [p for (p, _) in c.list_bindings()]
    assert ILLMProvider in binding_keys


def test_all_voussoir_modules_importable():
    """Every Phase-0 module imports without side-effects."""
    import voussoir.auth  # noqa: F401
    import voussoir.container  # noqa: F401
    import voussoir.executors  # noqa: F401
    import voussoir.guardrails  # noqa: F401
    import voussoir.memory  # noqa: F401
    import voussoir.middleware  # noqa: F401
    import voussoir.observability  # noqa: F401
    import voussoir.observability.logging_setup  # noqa: F401
    import voussoir.protocols  # noqa: F401
    import voussoir.tools  # noqa: F401


def test_audit_report_exists():
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    assert (repo / "docs/audits/ctxforge-2026-05.md").exists()
