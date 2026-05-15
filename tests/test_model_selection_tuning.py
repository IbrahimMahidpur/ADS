import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from multimodal_ds.agents.model_selection_agent import ModelSelectionAgent
from multimodal_ds.core.message_bus import MessageType

class TestModelSelectionTuning(unittest.TestCase):
    def setUp(self):
        self.session_id = "test_session_tuning"
        self.agent = ModelSelectionAgent(session_id=self.session_id)
        
        # Create dummy data
        self.X = pd.DataFrame(np.random.rand(100, 5), columns=[f"col_{i}" for i in range(5)])
        self.y = pd.Series(np.random.randint(0, 2, size=100))
        self.selection = {
            "primary_model": "RandomForestClassifier",
            "ensemble_models": ["LogisticRegression"],
            "cv_strategy": "stratified_kfold",
            "scoring_metric": "roc_auc"
        }

    @patch("multimodal_ds.agents.model_selection_agent.optuna")
    def test_tune_all_models_returns_dict_and_persists(self, mock_optuna):
        # Mock optuna study and optimization
        mock_study = MagicMock()
        mock_study.best_params = {"n_estimators": 100}
        mock_study.best_value = 0.85
        mock_optuna.create_study.return_value = mock_study
        
        # Mock MessageBus to capture published messages
        with patch("multimodal_ds.core.message_bus.get_bus") as mock_get_bus:
            mock_bus = MagicMock()
            mock_get_bus.return_value = mock_bus
            
            # Run tuning with very few trials for speed (though we mocked optuna)
            results = self.agent.tune_all_models(self.X, self.y, self.selection, n_trials=1)
            
            # 1. Verify return type and content
            self.assertIsInstance(results, dict)
            self.assertIn("RandomForestClassifier", results)
            self.assertIn("LogisticRegression", results)
            self.assertEqual(results["best_overall_model"], "RandomForestClassifier")
            self.assertEqual(results["best_overall_score"], 0.85)
            
            # 2. Verify persistence in SharedContextPool
            pool_results = self.agent.pool.get("tuning_results")
            self.assertEqual(pool_results, results)
            best_model = self.agent.pool.get("best_model")
            self.assertEqual(best_model, "RandomForestClassifier")
            
            # 3. Verify MessageBus publication
            mock_bus.publish.assert_called_once()
            published_msg = mock_bus.publish.call_args[0][0]
            self.assertEqual(published_msg.msg_type, MessageType.STATS_COMPLETE)
            self.assertEqual(published_msg.payload["best_model"], "RandomForestClassifier")
            self.assertEqual(published_msg.payload["best_score"], 0.85)
            self.assertEqual(published_msg.payload["models_tuned"], 2)

if __name__ == "__main__":
    unittest.main()
