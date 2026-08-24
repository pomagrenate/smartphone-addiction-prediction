import os
import unittest
import numpy as np
import pandas as pd

from src.data.loader import load_raw_data, TRAIN_PATH, TEST_PATH
from src.features.build_features import preprocess_datasets, create_engineered_features
from src.models.tabm import HAS_TORCH, TabMModel, TabMLoss
from src.utils.metrics import evaluate_predictions, rank_transform

if HAS_TORCH:
    import torch


class TestHyTabAddictPipeline(unittest.TestCase):
    
    def test_01_data_loader(self):
        """Test dataset loading and fallback generation."""
        train_df, test_df, sample_sub = load_raw_data()
        self.assertTrue(os.path.exists(TRAIN_PATH))
        self.assertTrue(os.path.exists(TEST_PATH))
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)
        self.assertIn("addicted_label", train_df.columns)
        print("✓ Test 01 Passed: Data Loader & Synthetic Generation")

    def test_02_feature_engineering(self):
        """Test domain feature calculations."""
        train_df, test_df, _ = load_raw_data()
        train_eng, test_eng, feature_cols, scaled_cols, cat_cols = preprocess_datasets(train_df, test_df)
        
        expected_features = [
            "unproductive_usage_ratio",
            "sleep_disruption_index",
            "notification_density",
            "session_duration_est",
            "weekend_excess_time",
            "distress_score",
            "screen_to_work_ratio"
        ]
        
        for feat in expected_features:
            self.assertIn(feat, train_eng.columns)
            self.assertFalse(train_eng[feat].isnull().any())
            
        print("✓ Test 02 Passed: Domain Feature Engineering")

    def test_03_tabm_architecture(self):
        """Test PyTorch TabM forward pass & multi-head loss if torch installed."""
        if not HAS_TORCH:
            print("⚠ Test 03 Skipped: PyTorch not installed in environment")
            return
            
        model = TabMModel(num_numerical=10, cat_cardinalities=[3], k_models=8, d_hidden=32)
        x_num = torch.randn(16, 10)
        x_cat = torch.zeros(16, 1, dtype=torch.long)
        y_true = torch.randint(0, 2, (16,), dtype=torch.float32)
        
        logits = model(x_num, x_cat)
        self.assertEqual(logits.shape, (16, 8))
        
        criterion = TabMLoss()
        loss = criterion(logits, y_true)
        self.assertFalse(torch.isnan(loss))
        print("✓ Test 03 Passed: PyTorch TabM Neural Backbone")

    def test_04_metrics_and_ranking(self):
        """Test ROC-AUC and rank transformation."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.75, 0.3])
        
        res = evaluate_predictions(y_true, y_pred, prefix="TestMetric")
        self.assertGreater(res["roc_auc"], 0.9)
        
        ranks = rank_transform(y_pred)
        self.assertEqual(len(ranks), len(y_pred))
        self.assertTrue(np.all((ranks >= 0.0) & (ranks <= 1.0)))
        print("✓ Test 04 Passed: Metrics & Rank Transformation")


if __name__ == "__main__":
    unittest.main()
