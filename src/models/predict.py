import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd

from src.features.build_features import create_engineered_features
from src.utils.metrics import rank_transform
from src.models.tabm import HAS_TORCH, TabMModel

if HAS_TORCH:
    import torch


def run_inference(
    test_path: str = None,
    data_dir: str = None,
    model_dir: str = "models",
    output_dir: str = "submissions"
):
    print("=" * 60)
    print(f"[Inference] HyTab-Addict Standalone Model Predictor")
    print("=" * 60)
    
    # 1. Resolve test data file path
    if test_path and os.path.exists(test_path):
        target_test_path = test_path
    elif data_dir and os.path.exists(os.path.join(data_dir, "test.csv")):
        target_test_path = os.path.join(data_dir, "test.csv")
    else:
        candidate_paths = [
            "/kaggle/input/playground-series-s6e8/test.csv",
            "/kaggle/input/smartphone-addiction-prediction/test.csv",
            os.path.join("data", "raw", "test.csv")
        ]
        target_test_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                target_test_path = p
                break
                
    if not target_test_path or not os.path.exists(target_test_path):
        raise FileNotFoundError("[Error] Could not find test dataset file for inference.")
        
    print(f"[Inference] Loading test dataset from: '{target_test_path}'")
    raw_test = pd.read_csv(target_test_path)
    
    # 2. Load preprocessors & saved models
    preprocessor_path = os.path.join(model_dir, "preprocessors.pkl")
    meta_path = os.path.join(model_dir, "meta_learner.pkl")
    
    if not (os.path.exists(preprocessor_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"[Error] Saved models not found in '{model_dir}/'. Run 'python src/models/train.py' first.")
        
    with open(preprocessor_path, "rb") as f:
        prep = pickle.load(f)
        
    scaler = prep["scaler"]
    encoders = prep["encoders"]
    all_num_cols = prep["all_num_cols"]
    cat_cols = prep["cat_cols"]
    feature_cols = prep["feature_cols"]
    
    with open(meta_path, "rb") as f:
        meta_learner = pickle.load(f)
        
    # 3. Engineer features & preprocess test data
    test_df = create_engineered_features(raw_test)
    
    for col in cat_cols:
        le = encoders[col]
        test_df[col] = test_df[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)
        
    test_num_scaled = scaler.transform(test_df[all_num_cols])
    scaled_num_cols = [f"{col}_scaled" for col in all_num_cols]
    for i, col_name in enumerate(scaled_num_cols):
        test_df[col_name] = test_num_scaled[:, i]
        
    X_test_num = test_df[scaled_num_cols].values
    X_test_cat = test_df[cat_cols].values
    X_test_tree = test_df[feature_cols].values
    
    # 4. Load & Predict GBDT Models
    lgb_files = sorted(glob.glob(os.path.join(model_dir, "lgb_fold_*.pkl")))
    xgb_files = sorted(glob.glob(os.path.join(model_dir, "xgb_fold_*.pkl")))
    cat_files = sorted(glob.glob(os.path.join(model_dir, "cat_fold_*.pkl")))
    
    n_folds = max(len(lgb_files), 1)
    
    test_preds_lgb = np.zeros(len(test_df))
    for f_path in lgb_files:
        with open(f_path, "rb") as f:
            model = pickle.load(f)
            test_preds_lgb += model.predict_proba(X_test_tree)[:, 1] / len(lgb_files)
            
    test_preds_xgb = np.zeros(len(test_df))
    for f_path in xgb_files:
        with open(f_path, "rb") as f:
            model = pickle.load(f)
            test_preds_xgb += model.predict_proba(X_test_tree)[:, 1] / len(xgb_files)
            
    test_preds_cat = np.zeros(len(test_df))
    for f_path in cat_files:
        with open(f_path, "rb") as f:
            model = pickle.load(f)
            test_preds_cat += model.predict_proba(X_test_tree)[:, 1] / len(cat_files)
            
    # 5. Load & Predict Neural Models
    test_preds_nn = np.zeros(len(test_df))
    if HAS_TORCH:
        tabm_files = sorted(glob.glob(os.path.join(model_dir, "tabm_fold_*.pth")))
        if len(tabm_files) > 0:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cat_cardinalities = [len(encoders[c].classes_) + 1 for c in cat_cols]
            
            for f_path in tabm_files:
                model = TabMModel(
                    num_numerical=len(all_num_cols),
                    cat_cardinalities=cat_cardinalities,
                    k_models=16,
                    d_embedding=16,
                    d_hidden=128,
                    num_layers=3
                ).to(device)
                model.load_state_dict(torch.load(f_path, map_location=device))
                model.eval()
                
                with torch.no_grad():
                    t_num = torch.tensor(X_test_num, dtype=torch.float32).to(device)
                    t_cat = torch.tensor(X_test_cat, dtype=torch.long).to(device)
                    t_logits = model(t_num, t_cat)
                    test_preds_nn += torch.sigmoid(t_logits).mean(dim=1).cpu().numpy() / len(tabm_files)
    else:
        mlp_files = sorted(glob.glob(os.path.join(model_dir, "mlp_fold_*.pkl")))
        if len(mlp_files) > 0:
            X_te_nn = np.column_stack([X_test_num, X_test_cat])
            for f_path in mlp_files:
                with open(f_path, "rb") as f:
                    model = pickle.load(f)
                    test_preds_nn += model.predict_proba(X_te_nn)[:, 1] / len(mlp_files)
                    
    # 6. Apply Stacking & Meta-Learner
    test_matrix = np.column_stack([
        rank_transform(test_preds_lgb),
        rank_transform(test_preds_xgb),
        rank_transform(test_preds_cat),
        rank_transform(test_preds_nn)
    ])
    
    test_final = meta_learner.predict(test_matrix)
    test_final = np.clip(test_final, 0.0001, 0.9999)
    
    # 7. Output Kaggle Submission File
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df = pd.DataFrame({
        "id": raw_test["id"],
        "addicted_label": np.round(test_final, 6)
    })
    submission_df.to_csv(submission_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"✓ Inference Complete! Submission saved to '{submission_path}' ({len(submission_df)} rows)")
    print("=" * 60)
    print("Sample Output:")
    print(submission_df.head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Standalone Inference on Test Set")
    parser.add_argument("--test-path", type=str, default=None, help="Path to target test.csv file")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing test.csv")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing saved model artifacts")
    parser.add_argument("--output-dir", type=str, default="submissions", help="Directory where submission.csv will be saved")
    args = parser.parse_args()
    
    run_inference(
        test_path=args.test_path,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
