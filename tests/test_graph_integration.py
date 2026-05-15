"""Integration test for the full LangGraph pipeline.

The test builds the graph, creates an initial state and invokes the graph while
monkey‑patching every LLM‑backed agent to a lightweight stub. This exercise
catches previously reported crashes such as the ``AgentMemory.count()`` bug,
duplicate logger handlers, and mismatched state keys.
"""

import pytest
from unittest.mock import MagicMock
import httpx

def mock_httpx_post(url, **kwargs):
    json_body = kwargs.get("json", {})
    mock = MagicMock()
    mock.status_code = 200
    # Ollama chat endpoint
    if "/api/chat" in str(url):
        messages = json_body.get("messages", [])
        last_msg = messages[-1].get("content", "") if messages else ""
        # Code generation requests
        if any(keyword in last_msg.lower() for keyword in ["write", "code", "python", "task"]):
            mock.json.return_value = {"message": {"content": "```python\nprint('hello')\n```"}}
        # Evaluation/review requests
        elif any(keyword in last_msg.lower() for keyword in ["evaluat", "score"]):
            import json as _json
            mock.json.return_value = {"message": {"content": _json.dumps({
                "statistical_validity": {"score": 8, "reasoning": "Valid."},
                "hallucination_risk": {"score": 8, "reasoning": "Clean."},
                "data_leakage": {"score": 9, "reasoning": "No leakage."},
                "output_completeness": {"score": 7, "reasoning": "Complete."},
                "recommendation": "Proceed."
            })}}}
        # Planner/hypothesis requests
        elif any(keyword in last_msg.lower() for keyword in ["hypothes", "plan", "task"]):
            mock.json.return_value = {"message": {"content": '[{"step":1,"name":"EDA","type":"eda","description":"Load and explore the dataset","tools":["pandas"],"expected_output":"Summary","depends_on":[]}]' }}}
        else:
            mock.json.return_value = {"message": {"content": "Analysis complete."}}
    elif "/api/embeddings" in str(url):
        mock.json.return_value = {"embedding": [0.1] * 768}
    else:
        mock.json.return_value = {"response": "OK", "message": {"content": "OK"}}
    return mock


@pytest.fixture
def graph_config():
    return {"configurable": {"thread_id": "test-integration-001"}}


def test_graph_integration(monkeypatch, graph_config):
    """Run the complete graph with all external agents stubbed.

    The stub agents return deterministic structures and never perform network I/O.
    The test verifies that the final state contains an ``eval_report`` entry and
    that no exceptions are raised during graph execution.
    """
    # ---------------------------------------------------------------------
    # Stub out all agents that would otherwise hit LLM services.
    # ---------------------------------------------------------------------
    # CodeExecutionAgent – return a successful execution result.
    class DummyCodeExecutionAgent:
        def __init__(self, *_, **__):
            pass

        def execute(self, task_description, data_context="", file_paths=None, max_retries=2):
            return {
                "success": True,
                "output": "dummy output",
                "files_created": [],
                "code": "print('hello')",
                "error": "",
                "retries_used": 0,
            }

    monkeypatch.setattr(
        "multimodal_ds.agents.code_execution_agent.CodeExecutionAgent",
        DummyCodeExecutionAgent,
    )

    # EvaluationAgent – return a simple report.
    class DummyEvalReport:
        def to_dict(self):
            return {
                "session_id": "test",
                "task_count": 0,
                "flagged_count": 0,
                "pass_count": 0,
                "overall_session_score": 10.0,
                "session_verdict": "PASS",
                "evaluations": [],
            }

    class DummyEvaluationAgent:
        def __init__(self, session_id="default"):
            pass

        def evaluate_task_results(self, task_results, data_context="", stat_report=None):
            return DummyEvalReport()

    monkeypatch.setattr(
        "multimodal_ds.agents.evaluation_agent.EvaluationAgent",
        DummyEvaluationAgent,
    )

    # Planner – produce an empty plan and no analysis tasks.
    def dummy_run_planner(user_objective, documents, session_id):
        return {"analysis_plan": "", "final_plan": "", "analysis_tasks": []}

    monkeypatch.setattr(
        "multimodal_ds.agents.planner_agent.run_planner",
        dummy_run_planner,
    )

    # ModelSelectionAgent – no model selection or tuning.
    class DummyModelSelectionAgent:
        def __init__(self, session_id="default"):
            pass

        def select_models(self, df, target_col, stat_report, automl_suggestion):
            return {}

        def tune_all_models(self, X, y, result):
            return {}

        def generate_tuned_ensemble_code(self, result, tuning_results, df_name, target_col):
            return ""

    monkeypatch.setattr(
        "multimodal_ds.agents.model_selection_agent.ModelSelectionAgent",
        DummyModelSelectionAgent,
    )

    # VisualizationAgent – return an empty manifest.
    class DummyVisManifest:
        def __init__(self):
            self.charts = []

    class DummyVisualizationAgent:
        def __init__(self, session_id="default"):
            pass

        def generate(self, df, target_col):
            return DummyVisManifest()

        def set_tuning_results(self, tuning):
            pass

    monkeypatch.setattr(
        "multimodal_ds.agents.visualization_agent.VisualizationAgent",
        DummyVisualizationAgent,
    )

    # ProblemUnderstandingAgent – return a minimal spec and no errors.
    class DummyProblemUnderstandingAgent:
        def __init__(self, session_id="default"):
            pass

        def understand(self, user_query, uploaded_files):
            class Spec:
                def to_dict(self):
                    return {}
            return Spec()

    monkeypatch.setattr(
        "multimodal_ds.agents.problem_understanding_agent.ProblemUnderstandingAgent",
        DummyProblemUnderstandingAgent,
    )

    # ReflectionAgent – stub with minimal reflect method.
    class DummyReflectionReport:
        def __init__(self):
            self.improved_instructions = ""
            self.target_agent = "executor"
        def to_dict(self):
            return {"dummy": True}

    class DummyReflectionAgent:
        def __init__(self, session_id="default", max_retries=2, *args, **kwargs):
            # Accept extra args for compatibility
            pass
        def reflect(self, eval_report=None, state=None):
            return DummyReflectionReport()

    monkeypatch.setattr(
        "multimodal_ds.agents.reflection_agent.ReflectionAgent",
        DummyReflectionAgent,
    )
    # Patch httpx globally to avoid real calls
    monkeypatch.setattr("httpx.post", mock_httpx_post)
    monkeypatch.setattr("httpx.Client.post", mock_httpx_post)
    monkeypatch.setattr("httpx.AsyncClient.post", mock_httpx_post)
    # ---------------------------------------------------------------------
    # Build and run the graph.
    # ---------------------------------------------------------------------
    from multimodal_ds.graph import build_graph, make_initial_state

    graph = build_graph()
    # Provide a trivial user query and no uploaded files.
    state = make_initial_state(user_query="test query", uploaded_files=[])

    # ``invoke`` runs the full graph and returns the final state.
    final_state = graph.invoke(state, config=graph_config)

    # The pipeline should finish without raising and should contain an eval_report.
    assert "eval_report" in final_state
    eval_report = final_state["eval_report"]
    assert eval_report["session_verdict"] == "PASS"
    assert eval_report["overall_session_score"] == 10.0


def test_graph_imports_without_presidio(monkeypatch):
    """graph.py must import cleanly even if presidio is missing."""
    import sys
    monkeypatch.setitem(sys.modules, "presidio_analyzer", None)
    monkeypatch.setitem(sys.modules, "presidio_anonymizer", None)
    import importlib
    import multimodal_ds.graph as g
    importlib.reload(g)   # should not raise
