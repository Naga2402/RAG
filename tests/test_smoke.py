"""Smoke tests that run without GPU/DB — validate config, language detection,
chunking, and graph wiring so the scaffold is verifiably sound on day one."""
from src.config import CFG
from src.ingestion.chunker import chunk_text
from src.ingestion.language import detect_language


def test_config_loads():
    assert CFG["project"]["name"] == "JISR"
    assert set(CFG["project"]["languages"]) == {"en", "ar"}


def test_language_detection():
    assert detect_language("ما هي شروط الدفع في العقد؟") == "ar"
    assert detect_language("What is the payment term in the contract?") == "en"


def test_chunker_tags_language():
    text = "Payment is due within 30 days.\n\nالدفع مستحق خلال ثلاثين يومًا."
    chunks = chunk_text(text, source="t.pdf")
    assert chunks
    assert {c.lang for c in chunks} <= {"en", "ar"}


def test_graph_builds():
    # Import-only: ensures node wiring/edges compile without external services.
    from src.agents.graph import build_graph
    assert build_graph() is not None
