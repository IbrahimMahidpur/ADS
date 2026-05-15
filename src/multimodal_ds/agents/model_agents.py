from __future__ import annotations
import logging
from typing import Optional
import pandas as pd
from multimodal_ds.config import OLLAMA_BASE_URL, CODER_MODEL, LLM_TIMEOUT

logger = logging.getLogger(__name__)

class BaseModelAgent:
    """Base class for all model-specific agents."""
    MODEL_NAME = "base"
    IMPORTS = ["import pandas as pd", "import numpy as np"]
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
    
    def generate_training_code(
        self,
        target_col: str,
        feature_cols: list[str],
        task_type: str,  # "classification" | "regression"
        best_params: dict,
    ) -> str:
        raise NotImplementedError

class XGBoostAgent(BaseModelAgent):
    MODEL_NAME = "xgboost"
    def generate_training_code(self, target_col, feature_cols, task_type, best_params):
        estimator = "XGBClassifier" if task_type == "classification" else "XGBRegressor"
        return f"""
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import joblib, pandas as pd, numpy as np

df = pd.read_csv('{self.session_id}_data.csv')
X = df[{feature_cols}]
y = df['{target_col}']

model = xgb.{estimator}(**{best_params if best_params else {}})
scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
model.fit(X, y)
joblib.dump(model, 'model_xgboost.pkl')
pd.DataFrame({{'feature': X.columns, 'importance': model.feature_importances_}}).to_csv('feature_importance.csv', index=False)
print(f'XGBoost CV AUC: {{scores.mean():.4f}} ± {{scores.std():.4f}}')
"""

class LightGBMAgent(BaseModelAgent):
    MODEL_NAME = "lightgbm"
    def generate_training_code(self, target_col, feature_cols, task_type, best_params):
        estimator = "LGBMClassifier" if task_type == "classification" else "LGBMRegressor"
        return f"""
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import joblib, pandas as pd, numpy as np

df = pd.read_csv('{self.session_id}_data.csv')
X = df[{feature_cols}]
y = df['{target_col}']

model = lgb.{estimator}(verbose=-1, **{best_params if best_params else {}})
scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
model.fit(X, y)
joblib.dump(model, 'model_lightgbm.pkl')
pd.DataFrame({{'feature': X.columns, 'importance': model.feature_importances_}}).to_csv('feature_importance.csv', index=False)
print(f'LightGBM CV AUC: {{scores.mean():.4f}} ± {{scores.std():.4f}}')
"""

class RandomForestAgent(BaseModelAgent):
    MODEL_NAME = "random_forest"
    def generate_training_code(self, target_col, feature_cols, task_type, best_params):
        estimator = "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor"
        return f"""
from sklearn.ensemble import {estimator}
from sklearn.model_selection import cross_val_score
import joblib, pandas as pd, numpy as np

df = pd.read_csv('{self.session_id}_data.csv')
X = df[{feature_cols}]
y = df['{target_col}']

model = {estimator}(n_jobs=-1, **{best_params if best_params else {}})
scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
model.fit(X, y)
joblib.dump(model, 'model_random_forest.pkl')
pd.DataFrame({{'feature': X.columns, 'importance': model.feature_importances_}}).to_csv('feature_importance.csv', index=False)
print(f'RandomForest CV AUC: {{scores.mean():.4f}} ± {{scores.std():.4f}}')
"""

class LogisticRegressionAgent(BaseModelAgent):
    MODEL_NAME = "logistic_regression"
    def generate_training_code(self, target_col, feature_cols, task_type, best_params):
        return f"""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib, pandas as pd, numpy as np

df = pd.read_csv('{self.session_id}_data.csv')
X = df[{feature_cols}]
y = df['{target_col}']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, **{best_params if best_params else {}}))
])
scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
pipeline.fit(X, y)
joblib.dump(pipeline, 'model_logistic_regression.pkl')
print(f'LogisticRegression CV AUC: {{scores.mean():.4f}} ± {{scores.std():.4f}}')
"""

class NeuralNetAgent(BaseModelAgent):
    MODEL_NAME = "neural_net"
    def generate_training_code(self, target_col, feature_cols, task_type, best_params):
        return f"""
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib, pandas as pd, numpy as np

df = pd.read_csv('{self.session_id}_data.csv')
X = df[{feature_cols}]
y = df['{target_col}']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MLPClassifier(max_iter=500, early_stopping=True,
                            **{best_params if best_params else {}}))
])
scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
pipeline.fit(X, y)
joblib.dump(pipeline, 'model_neural_net.pkl')
print(f'NeuralNet CV AUC: {{scores.mean():.4f}} ± {{scores.std():.4f}}')
"""

# Registry — maps model name strings to agent classes
MODEL_AGENT_REGISTRY: dict[str, type[BaseModelAgent]] = {
    "xgboost":            XGBoostAgent,
    "lightgbm":           LightGBMAgent,
    "random_forest":      RandomForestAgent,
    "logistic_regression": LogisticRegressionAgent,
    "neural_net":         NeuralNetAgent,
}

def get_model_agent(model_name: str, session_id: str = "default") -> BaseModelAgent:
    """Factory — returns the correct model agent for a given model name string."""
    cls = MODEL_AGENT_REGISTRY.get(model_name.lower().replace(" ", "_"))
    if cls is None:
        logger.warning(f"[ModelAgents] No dedicated agent for '{model_name}' — using RandomForest fallback")
        cls = RandomForestAgent
    return cls(session_id=session_id)
