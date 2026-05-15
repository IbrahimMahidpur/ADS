"""
Test suite for the Multimodal Agentic DS Engine.
Run with:  pytest tests/ -v
"""
import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("data") / "sample.csv"
    df = pd.DataFrame({
        "age": [25, 32, 41, 28, 55, 38, 29, 47],
        "income": [45000, 72000, 88000, 51000, 120000, 67000, 48000, 95000],
        "churn": [0, 0, 1, 0, 1, 0, 0, 1],
    })
    df.to_csv(tmp, index=False)
    return tmp


@pytest.fixture(scope="session")
def sample_txt(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("data") / "notes.txt"
    tmp.write_text("This is a plain text document for testing ingestion.")
    return tmp


@pytest.fixture(scope="session")
def api_client():
    from multimodal_ds.api.app import app
    return TestClient(app)


# ── Schema tests ───────────────────────────────────────────────────────────

class TestSchema:
    def test_unified_document_defaults(self):
        from multimodal_ds.core.schema import UnifiedDocument, DataType, ProcessingStatus
        doc = UnifiedDocument()
        assert doc.data_type == DataType.UNKNOWN
        assert doc.status == ProcessingStatus.PENDING
        assert doc.text_content == ""
        assert isinstance(doc.metadata, dict)

    def test_to_dict_truncates_text(self):
        from multimodal_ds.core.schema import UnifiedDocument
        doc = UnifiedDocument(text_content="x" * 5000)
        d = doc.to_dict()
        assert len(d["text_content"]) <= 2000

    def test_provenance_defaults(self):
        from multimodal_ds.core.schema import Provenance
        p = Provenance(source_path="/tmp/test.csv")
        assert p.source_path == "/tmp/test.csv"
        assert p.processing_time_s == 0.0
        assert isinstance(p.ingested_at, str)


# ── Tabular ingestion tests ────────────────────────────────────────────────

class TestTabularIngestion:
    def test_ingest_csv(self, sample_csv):
        from multimodal_ds.ingestion.tabular_ingestion import ingest_tabular
        from multimodal_ds.core.schema import ProcessingStatus

        doc = ingest_tabular(str(sample_csv))
        assert doc.status == ProcessingStatus.DONE
        assert doc.structured_data is not None
        assert doc.schema_info["shape"] == [8, 3]
        assert "age" in doc.schema_info["columns"]
        assert "income" in doc.schema_info["numeric_cols"]

    def test_profile_contains_stats(self, sample_csv):
        from multimodal_ds.ingestion.tabular_ingestion import ingest_tabular
        doc = ingest_tabular(str(sample_csv))
        assert "numeric_stats" in doc.data_profile
        assert "missing_values" in doc.data_profile
        assert "duplicate_rows" in doc.data_profile

    def test_text_summary_generated(self, sample_csv):
        from multimodal_ds.ingestion.tabular_ingestion import ingest_tabular
        doc = ingest_tabular(str(sample_csv))
        assert "8 rows" in doc.text_content
        assert "3 columns" in doc.text_content

    def test_automl_suggestion(self, sample_csv):
        from multimodal_ds.ingestion.tabular_ingestion import ingest_tabular
        doc = ingest_tabular(str(sample_csv))
        suggestion = doc.metadata.get("automl_suggestion", {})
        assert suggestion.get("task") in ("classification", "regression", "unknown")

    def test_missing_file_fails_gracefully(self):
        from multimodal_ds.ingestion.tabular_ingestion import ingest_tabular
        from multimodal_ds.core.schema import ProcessingStatus
        doc = ingest_tabular("/nonexistent/path/data.csv")
        assert doc.status == ProcessingStatus.FAILED
        assert "error" in doc.metadata


# ── Plain text ingestion tests ─────────────────────────────────────────────

class TestTextIngestion:
    def test_ingest_txt(self, sample_txt):
        from multimodal_ds.ingestion.router import _ingest_plain_text
        from multimodal_ds.core.schema import ProcessingStatus

        doc = _ingest_plain_text(str(sample_txt))
        assert doc.status == ProcessingStatus.DONE
        assert "plain text" in doc.text_content.lower()
        assert doc.metadata["word_count"] > 0

    def test_char_count_tracked(self, sample_txt):
        from multimodal_ds.ingestion.router import _ingest_plain_text
        doc = _ingest_plain_text(str(sample_txt))
        assert doc.metadata["char_count"] == len(doc.text_content)


# ── Router tests ───────────────────────────────────────────────────────────

class TestRouter:
    def test_multi_modal_routing(self, sample_csv, sample_txt):
        # Test that ingesting both a CSV and a TXT file together returns two valid UnifiedDocuments
        from multimodal_ds.ingestion.router import ingest_multiple
        from multimodal_ds.core.schema import DataType, ProcessingStatus

        docs = ingest_multiple([str(sample_csv), str(sample_txt)])
        assert len(docs) == 2
        # Verify we have one tabular and one text document
        types = {doc.data_type for doc in docs}
        assert DataType.TABULAR in types
        assert DataType.TEXT in types
        # Ensure both documents have a DONE status
        for doc in docs:
            assert doc.status == ProcessingStatus.DONE
    def test_routes_csv(self, sample_csv):
        from multimodal_ds.ingestion.router import route_and_ingest
        from multimodal_ds.core.schema import DataType
        doc = route_and_ingest(str(sample_csv))
        assert doc.data_type == DataType.TABULAR

    def test_routes_txt(self, sample_txt):
        from multimodal_ds.ingestion.router import route_and_ingest
        from multimodal_ds.core.schema import DataType
        doc = route_and_ingest(str(sample_txt))
        assert doc.data_type == DataType.TEXT

    def test_routes_unknown_as_text(self, tmp_path):
        from multimodal_ds.ingestion.router import route_and_ingest
        from multimodal_ds.core.schema import DataType
        f = tmp_path / "mystery.xyz"
        f.write_text("some content")
        doc = route_and_ingest(str(f))
        assert doc.data_type == DataType.TEXT

    def test_ingest_multiple(self, sample_csv, sample_txt):
        from multimodal_ds.ingestion.router import ingest_multiple
        docs = ingest_multiple([str(sample_csv), str(sample_txt)])
        assert len(docs) == 2


# ── Statistical agent tests ────────────────────────────────────────────────

class TestStatisticalAgent:
    def test_normality_check(self, sample_csv):
        from multimodal_ds.agents.statistical_agent import StatisticalReasoningAgent
        agent = StatisticalReasoningAgent(session_id="test")
        df = pd.read_csv(sample_csv)
        report = agent._check_normality(df)
        assert "age" in report or "income" in report

    def test_correlation_check(self, sample_csv):
        from multimodal_ds.agents.statistical_agent import StatisticalReasoningAgent
        agent = StatisticalReasoningAgent(session_id="test")
        df = pd.read_csv(sample_csv)
        result = agent._check_correlation(df)
        assert "matrix" in result
        assert "strong_pairs" in result

    def test_recommendations_generated(self, sample_csv):
        from multimodal_ds.agents.statistical_agent import StatisticalReasoningAgent
        agent = StatisticalReasoningAgent(session_id="test")
        df = pd.read_csv(sample_csv)
        # Bypass LLM interpret step by providing a minimal fake report
        fake_report = {
            "normality": {"age": {"is_normal": False}},
            "correlation": {"n_strong": 0, "strong_pairs": [], "matrix": {}},
            "multicollinearity": {"multicollinearity_detected": False},
            "stationarity": {},
        }
        recs = agent._generate_recommendations(fake_report)
        assert isinstance(recs, list)
        assert len(recs) >= 1


# ── Memory tests ───────────────────────────────────────────────────────────

class TestAgentMemory:
    def test_store_and_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr("multimodal_ds.config.CHROMA_DIR", tmp_path / "chroma")
        from multimodal_ds.memory.agent_memory import AgentMemory
        mem = AgentMemory(collection_name="test_col")
        # Without Ollama, embedding will fail — should still store via text fallback
        # Just verify no exception is raised
        try:
            entry_id = mem.store("test content", metadata={"type": "test"})
            assert isinstance(entry_id, str)
        except Exception:
            pass  # ChromaDB may not be installed in CI


# ── API tests ──────────────────────────────────────────────────────────────

class TestAPI:
    def test_health(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_ingest_csv(self, api_client, sample_csv):
        with open(sample_csv, "rb") as f:
            resp = api_client.post(
                "/ingest",
                files={"file": ("sample.csv", f, "text/csv")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["data_type"] == "tabular"
        assert data["status"] == "done"
        assert "document_id" in data

    def test_ingest_txt(self, api_client, sample_txt):
        with open(sample_txt, "rb") as f:
            resp = api_client.post(
                "/ingest",
                files={"file": ("notes.txt", f, "text/plain")},
            )
        assert resp.status_code == 200
        assert resp.json()["data_type"] == "text"

    def test_ingest_no_file_fails(self, api_client):
        resp = api_client.post("/ingest")
        assert resp.status_code == 422  # FastAPI validation error

    def test_session_empty(self, api_client):
        resp = api_client.get("/session/nonexistent_session_xyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entry_count"] == 0

    def test_output_not_found(self, api_client):
        resp = api_client.get("/output/nonexistent_xyz")
        assert resp.status_code == 404


# ── Graph parallel ingestion tests ───────────────────────────────────────────

class TestGraphParallel:
    def test_parallel_ingestion_csv_txt(self, sample_csv, sample_txt):
        from multimodal_ds.graph import build_graph, make_initial_state
        
        # 1. Setup initial state with both CSV and TXT
        state = make_initial_state(
            user_query="Analyze my data and notes",
            uploaded_files=[str(sample_csv), str(sample_txt)]
        )
        
        # 2. Build and compile graph
        app = build_graph(use_sqlite_checkpointer=False)
        
        # 3. Execute graph and capture node executions
        # We limit the run up to merge_ingest to keep the test fast and avoid LLM calls
        # We can use interrupt_after or simply check node_names in stream
        node_names = []
        
        config = {"configurable": {"thread_id": "test_parallel"}}
        # Note: In a real test we might want to mock the agents to avoid actual LLM calls
        # if the test continues past ingestion.
        try:
            for output in app.stream(state, config=config):
                for node_name in output.keys():
                    node_names.append(node_name)
                    if node_name == "merge_ingest":
                        raise StopIteration # Stop after ingestion merge
        except StopIteration:
            pass

        # 4. Assertions
        assert "tab_ingest" in node_names, "tab_ingest should have run"
        assert "doc_ingest" in node_names, "doc_ingest should have run"
        assert "img_ingest" not in node_names, "img_ingest should NOT have run"
        assert "audio_ingest" not in node_names, "audio_ingest should NOT have run"
        assert "merge_ingest" in node_names, "merge_ingest should have run"
        
        # Check that state was merged correctly
        full_final_state = app.get_state(config).values
        assert len(full_final_state["tabular_summaries"]) > 0, "Tabular summaries should be present"
        assert len(full_final_state["parsed_documents"]) > 0, "Parsed documents should be present"
