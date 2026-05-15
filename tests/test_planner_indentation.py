import pytest
from unittest.mock import MagicMock, patch
from multimodal_ds.graph import _planner_node
from multimodal_ds.core.schema import DataType, ProcessingStatus

def test_planner_runs_without_tabular_data():
    """Verify that _planner_node calls run_planner even if statistical_report is missing/empty."""
    
    # Mock run_planner to return a dummy plan
    mock_plan = {
        "analysis_plan": [{"name": "Step 1", "description": "Do something"}],
        "final_plan": "Step 1: Do something",
        "hypotheses": ["H1"]
    }
    
    # Patch the run_planner in the agents module since it's imported locally in _planner_node
    with patch("multimodal_ds.agents.planner_agent.run_planner", return_value=mock_plan) as mock_run:
        # State with NO statistical_report and NO tabular_summaries
        state = {
            "user_query": "Analyze this data",
            "session_id": "test_session",
            "uploaded_files": ["doc.pdf"],
            "parsed_documents": [{"text_content": "PDF text content"}],
            "tabular_summaries": [],
            "statistical_report": {}  # Empty dict
        }
        
        result = _planner_node(state)
        
        # Ensure run_planner was called
        mock_run.assert_called_once()
        
        # Verify result contains the plan tasks
        assert result["analysis_tasks"] == mock_plan["analysis_plan"]
        assert result["steps_total"] == 1
        assert result["analysis_plan"] == "Step 1: Do something"

def test_planner_runs_with_tabular_data():
    """Verify that _planner_node still works correctly with tabular data and statistical_report."""
    
    mock_plan = {
        "analysis_plan": [{"name": "Step 1", "description": "Do something"}],
        "final_plan": "Step 1: Do something",
        "hypotheses": ["H1"]
    }
    
    with patch("multimodal_ds.agents.planner_agent.run_planner", return_value=mock_plan) as mock_run:
        state = {
            "user_query": "Analyze this data",
            "session_id": "test_session",
            "uploaded_files": ["data.csv"],
            "tabular_summaries": [{
                "source": "data.csv",
                "columns": ["A", "B"],
                "shape": [10, 2],
                "dtypes": {"A": "int", "B": "float"},
                "data_profile": {"numeric_stats": {"A": {"mean": 1, "std": 0, "min": 1, "max": 1}}}
            }],
            "statistical_report": {
                "normality": {"A": {"is_normal": False}},
                "correlation": {"n_strong": 1},
                "multicollinearity": {"multicollinearity_detected": True}
            }
        }
        
        result = _planner_node(state)
        
        # Ensure run_planner was called
        mock_run.assert_called_once()
        
        # Check that the documents passed to run_planner include the statistical report
        args, kwargs = mock_run.call_args
        documents = kwargs.get("documents")
        
        assert documents is not None
        # One for tabular summary, one for statistical validation report
        assert len(documents) >= 2
        
        stats_doc = next((d for d in documents if "Statistical Validation Report" in d.text_content), None)
        assert stats_doc is not None
        assert "Statistical findings: non-normal columns=['A'], strong_correlations=1, multicollinearity=True" in stats_doc.text_content
