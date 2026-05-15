import pytest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

import httpx
from fastapi.testclient import TestClient

from multimodal_ds.graph import build_graph, make_initial_state, session_logger
from multimodal_ds.core.schema import DataType, UnifiedDocument
from multimodal_ds.api.app import app


class DummyPiiResult:
    def __init__(self, entity_type):
        self.entity_type = entity_type

def mock_presidio_analyze(self, text, language="en", **kwargs):
    if "PII" in text:
        return [DummyPiiResult("CREDIT_CARD")]
    return []

def mock_httpx_post(url, *args, **kwargs):
    """
    Mock Ollama / LLM endpoint responses to test maximum amount of real code.
    Returns targeted responses based on the payload (system prompt or prompt).
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    
    req_json = kwargs.get("json", {})
    prompt = req_json.get("prompt", "")
    system = req_json.get("system", "")
    model = req_json.get("model", "")
    
    response_text = ""
    
    lower_system = system.lower()
    lower_prompt = prompt.lower()
    lower_model = model.lower()

    if "hypothesis" in lower_system or "hypotheses" in lower_prompt:
        response_text = '{"hypotheses": ["H1: Mocked Hypothesis"]}'
    elif "planner" in lower_system or "tasks" in lower_prompt or "planner" in lower_model:
        # Generate 2 tasks to test accumulating fields
        response_text = '{"tasks": [{"name": "Task 1", "description": "First task"}, {"name": "Task 2", "description": "Second task"}]}'
    elif "reviewer" in lower_model or "evaluate" in lower_system or "evaluator" in lower_system:
        # Evaluation
        if "report" in lower_prompt or "report" in lower_system:
            response_text = "# Executive Report\n\nAll tasks completed successfully."
        else:
            # Task evaluation verdict
            response_text = '{"session_verdict": "success", "score": 95, "feedback": "Good job", "output_preview": "Looks ok"}'
    elif "coder" in lower_model or "python" in lower_system:
        # Code execution agent
        response_text = '```python\nprint("Mocked code executed")\n# Create a dummy artifact\nopen("dummy_artifact.csv", "w").write("a,b\\n1,2")\n```'
    elif "visualization" in lower_system or "plot" in lower_prompt:
        response_text = '{"chart_type": "scatter", "x": "a", "y": "b"}'
    else:
        # Generic fallback
        response_text = "Generic response from mock LLM."
        
    mock_resp.json.return_value = {"response": response_text}
    return mock_resp

@pytest.fixture
def temp_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "data.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("id,value\n1,10\n2,20\n3,30\n")
            
        txt_path = os.path.join(tmpdir, "notes.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("This document has PII: 4532-1234-5678-9012")
            
        yield csv_path, txt_path


def test_health_endpoint_works():
    """
    6. Specifically tests that the count() crash is gone (health endpoint works).
    """
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    data = resp.json()
    assert data["status"] == "ok"
    assert "memory_entries" in data


@patch("httpx.post", side_effect=mock_httpx_post)
@patch("httpx.Client.post", side_effect=mock_httpx_post)
@patch("httpx.AsyncClient.post", side_effect=mock_httpx_post)
@patch("presidio_analyzer.AnalyzerEngine.analyze", mock_presidio_analyze)
def test_graph_integration_end_to_end(mock_async_post, mock_client_post, mock_post, temp_files):
    """
    Full integration test that satisfies all constraints.
    """
    csv_path, txt_path = temp_files
    
    # Ensure handlers are clean before the run to test duplication
    session_logger.handlers.clear()
    
    # 4. Calls build_graph() + make_initial_state() + graph.invoke()
    graph = build_graph()
    initial_state = make_initial_state(
        user_query="Analyze this data for insights",
        uploaded_files=[csv_path, txt_path],
        session_id="test_integration_session_001"
    )
    
    final_state = graph.invoke(initial_state, config={"configurable": {"thread_id": "test_integration_session_001"}})
    
    # 5. Asserts final state contents
    assert "final_report" in final_state
    assert isinstance(final_state["final_report"], str)
    assert len(final_state["final_report"]) > 0, "Final report should be a non-empty string"
    
    # Check that all tasks are completed
    assert final_state.get("current_step", 0) > 0
    assert final_state["current_step"] == final_state["steps_total"], "Not all tasks completed"
    
    # Check eval report
    eval_report = final_state.get("eval_report")
    assert eval_report is not None, "Evaluation report should exist"
    
    if hasattr(eval_report, "to_dict"):
        eval_dict = eval_report.to_dict()
    else:
        eval_dict = eval_report
    assert "session_verdict" in eval_dict or "task_results" in eval_dict
    
    # 7. Tests that the duplicate logger handler is not present
    assert len(session_logger.handlers) == 1, f"Expected 1 logger handler, found {len(session_logger.handlers)}"
    
    # 8. Tests that ACCUMULATING fields (code_outputs, errors) contain entries from ALL steps
    # We mocked the planner to return 2 tasks. The state should accumulate the code outputs for both tasks.
    code_outputs = final_state.get("code_outputs", [])
    full_code_outputs = final_state.get("full_code_outputs", [])
    
    # Both tasks should have produced an output if they didn't crash
    assert len(full_code_outputs) == 2, f"Expected 2 code_output entries, got {len(full_code_outputs)}"
    
    # 9. Tests that a BLOCKED PII document does not abort the pipeline — clean docs continue to planning
    parsed_docs = final_state.get("parsed_documents", [])
    blocked_docs = [
        d for d in parsed_docs 
        if (hasattr(d, "status") and getattr(d.status, "value", d.status) == "blocked") or 
           (isinstance(d, dict) and d.get("status") == "blocked")
    ]
    assert len(blocked_docs) == 1, "The TXT document containing PII should be blocked"
    
    # 10. Tests fan-out routing: CSV + TXT upload -> both tabular and text ingest nodes produce output in final state
    assert len(parsed_docs) == 2, "Both CSV and TXT should be in parsed documents"
    
    tabular_docs = [
        d for d in parsed_docs 
        if (hasattr(d, "data_type") and getattr(d.data_type, "value", d.data_type) == "tabular") or
           (isinstance(d, dict) and d.get("data_type") == "tabular")
    ]
    text_docs = [
        d for d in parsed_docs 
        if (hasattr(d, "data_type") and getattr(d.data_type, "value", d.data_type) == "text") or
           (isinstance(d, dict) and d.get("data_type") == "text")
    ]
    
    assert len(tabular_docs) == 1, "Expected 1 tabular document to be parsed"
    assert len(text_docs) == 1, "Expected 1 text document to be parsed"
