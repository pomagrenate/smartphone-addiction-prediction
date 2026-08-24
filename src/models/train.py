import os
import sys

# Ensure repository root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from src.data.loader import load_raw_data
from src.features.build_features import create_engineered_features, NUMERICAL_COLS, ENGINEERED_COLS, CATEGORICAL_COLS
from src.utils.metrics import evaluate_predictions, rank_transform
from src.models.tabm import HAS_TORCH, TabMModel, TabMLoss

if HAS_TORCH:
    import torch
    from torch.utils.data import TensorDataset, DataLoader

MODEL_DIR = "models"
SUBMISSION_DIR = "submissions"


def get_safe_torch_device() -> str:
    """Return 'cuda' only if PyTorch CUDA execution actually succeeds on the current GPU architecture."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return "cpu"
    try:
        cap = torch.cuda.get_device_capability()
        if cap[0] < 7:
            print(f"[Device Warning] GPU sm_{cap[0]}{cap[1]} (Tesla P100) lacks kernel support in this PyTorch build. Using CPU for TabM.")
            return "cpu"
        test_t = torch.zeros(1, device="cuda")
        _ = test_t + 1
        return "cuda"
    except Exception as e:
        print(f"[Device Warning] CUDA execution failed ({e}). Using CPU for TabM.")
        return "cpu"


def train_tabm_fold(
    train_num: np.ndarray,
    train_cat: np.ndarray,
    train_y: np.ndarray,
    val_num: np.ndarray,
    val_cat: np.ndarray,
    cat_cardinalities: list,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu"
):
    num_features = train_num.shape[1]
    model = TabMModel(
        num_numerical=num_features,
        cat_cardinalities=cat_cardinalities,
        k_models=16,
        d_embedding=16,
        d_hidden=128,
        num_layers=3,
        dropout=0.15
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = TabMLoss()
    
    t_num = torch.tensor(train_num, dtype=torch.float32)
    t_cat = torch.tensor(train_cat, dtype=torch.long)
    t_y = torch.tensor(train_y, dtype=torch.float32)
    
    dataset = TensorDataset(t_num, t_cat, t_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    v_num = torch.tensor(val_num, dtype=torch.float32).to(device)
    v_cat = torch.tensor(val_cat, dtype=torch.long).to(device)
    
    model.train()
    for epoch in range(epochs):
        for b_num, b_cat, b_y in loader:
            b_num, b_cat, b_y = b_num.to(device), b_cat.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = model(b_num, b_cat)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        val_logits = model(v_num, v_cat)
        val_probs = torch.sigmoid(val_logits).mean(dim=1).cpu().numpy()
        
    return model, val_probs


def run_pipeline(
    n_splits: int = 5,
    seed: int = 42,
    data_dir: str = None,
    train_path: str = None,
    test_path: str = None,
    output_dir: str = "submissions",
    model_dir: str = "models"
):
    print("=" * 60)
    print(f"[Train] Initializing HyTab-Addict ML Pipeline ({n_splits}-Fold CV)")
    print("=" * 60)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    raw_train, raw_test, sample_sub = load_raw_data(
        data_dir=data_dir,
        train_path=train_path,
        test_path=test_path
    )
    print(f"[Loader] Loaded Train Shape: {raw_train.shape}, Test Shape: {raw_test.shape}")
    
    # 2. Preprocess & Feature Engineering with Disk Caching
    cache_dir = os.path.join(model_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train_engineered.pkl")
    test_cache = os.path.join(cache_dir, "test_engineered.pkl")
    
    if os.path.exists(train_cache) and os.path.exists(test_cache):
        print(f"[Loader] Loading fast cached engineered features from '{cache_dir}'...")
        train_df = pd.read_pickle(train_cache)
        test_df = pd.read_pickle(test_cache)
    else:
        print("[Loader] Engineering features from raw dataset...")
        train_df = create_engineered_features(raw_train)
        test_df = create_engineered_features(raw_test)
        train_df.to_pickle(train_cache)
        test_df.to_pickle(test_cache)
        print(f"[Saver] Fast cached engineered features saved to '{train_cache}'.")
    
    all_num_cols = NUMERICAL_COLS + ENGINEERED_COLS
    
    # Fit & Save Encoders & Scalers
    encoders = {}
    for col in CATEGORICAL_COLS:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = test_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)
        encoders[col] = le
        
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_num_scaled = scaler.fit_transform(train_df[all_num_cols])
    test_num_scaled = scaler.transform(test_df[all_num_cols])
    
    scaled_num_cols = [f"{col}_scaled" for col in all_num_cols]
    for i, col_name in enumerate(scaled_num_cols):
        train_df[col_name] = train_num_scaled[:, i]
        test_df[col_name] = test_num_scaled[:, i]
        
    # Save preprocessing metadata
    with open(os.path.join(model_dir, "preprocessors.pkl"), "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "encoders": encoders,
            "all_num_cols": all_num_cols,
            "scaled_num_cols": scaled_num_cols,
            "cat_cols": CATEGORICAL_COLS,
            "feature_cols": all_num_cols + CATEGORICAL_COLS
        }, f)
    print(f"[Saver] Preprocessing metadata saved to '{model_dir}/preprocessors.pkl'.")
    
    target_col = "addicted_label"
    y = train_df[target_col].values
    y_binary = (y >= 0.5).astype(int)
    
    feature_cols = all_num_cols + CATEGORICAL_COLS
    cat_cardinalities = [train_df[col].nunique() + 1 for col in CATEGORICAL_COLS]
    
    X_num = train_df[[f"{c}_scaled" for c in all_num_cols]].values
    X_cat = train_df[CATEGORICAL_COLS].values
    X_tree = train_df[feature_cols].values
    
    X_test_num = test_df[[f"{c}_scaled" for c in all_num_cols]].values
    X_test_cat = test_df[CATEGORICAL_COLS].values
    X_test_tree = test_df[feature_cols].values
    
    # Out of fold arrays
    oof_lgb = np.zeros(len(train_df))
    oof_xgb = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))
    oof_nn = np.zeros(len(train_df))
    
    test_preds_lgb = np.zeros(len(test_df))
    test_preds_xgb = np.zeros(len(test_df))
    test_preds_cat = np.zeros(len(test_df))
    test_preds_nn = np.zeros(len(test_df))
    
    # Detect GPU availability for XGBoost, CatBoost, and PyTorch TabM
    use_gpu = False
    if HAS_TORCH:
        use_gpu = torch.cuda.is_available()
    print(f"[Pipeline] PyTorch Available: {HAS_TORCH} | GPU Acceleration: {use_gpu} | Folds: {n_splits}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tree, y_binary)):
        print(f"\n--- Training & Checkpointing Fold {fold + 1}/{n_splits} ---")
        
        X_tr_tree, y_tr = X_tree[train_idx], y_binary[train_idx]
        X_va_tree, y_va = X_tree[val_idx], y_binary[val_idx]
        
        # 1. LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            random_state=seed + fold,
            verbosity=-1
        )
        lgb_model.fit(
            X_tr_tree, y_tr,
            eval_set=[(X_va_tree, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        oof_lgb[val_idx] = lgb_model.predict_proba(X_va_tree)[:, 1]
        test_preds_lgb += lgb_model.predict_proba(X_test_tree)[:, 1] / n_splits
        with open(os.path.join(model_dir, f"lgb_fold_{fold}.pkl"), "wb") as f:
            pickle.dump(lgb_model, f)
            
        # 2. XGBoost (GPU Accelerated if available)
        xgb_kwargs = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 6,
            "random_state": seed + fold,
            "eval_metric": "logloss",
            "early_stopping_rounds": 50,
            "tree_method": "hist"
        }
        if use_gpu:
            xgb_kwargs["device"] = "cuda"
            
        xgb_model = xgb.XGBClassifier(**xgb_kwargs)
        xgb_model.fit(
            X_tr_tree, y_tr,
            eval_set=[(X_va_tree, y_va)],
            verbose=False
        )
        oof_xgb[val_idx] = xgb_model.predict_proba(X_va_tree)[:, 1]
        test_preds_xgb += xgb_model.predict_proba(X_test_tree)[:, 1] / n_splits
        with open(os.path.join(model_dir, f"xgb_fold_{fold}.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)
            
        # 3. CatBoost (GPU Accelerated if available)
        cat_kwargs = {
            "iterations": 300,
            "learning_rate": 0.03,
            "depth": 6,
            "random_seed": seed + fold,
            "verbose": False
        }
        if use_gpu:
            cat_kwargs["task_type"] = "GPU"
            
        cat_model = CatBoostClassifier(**cat_kwargs)
        cat_model.fit(X_tr_tree, y_tr, eval_set=(X_va_tree, y_va), early_stopping_rounds=50)
        oof_cat[val_idx] = cat_model.predict_proba(X_va_tree)[:, 1]
        test_preds_cat += cat_model.predict_proba(X_test_tree)[:, 1] / n_splits
        with open(os.path.join(model_dir, f"cat_fold_{fold}.pkl"), "wb") as f:
            pickle.dump(cat_model, f)
            
        # 4. Neural Model Checkpoint
        if HAS_TORCH:
            device = get_safe_torch_device()
            tabm_m, tabm_val_preds = train_tabm_fold(
                train_num=X_num[train_idx],
                train_cat=X_cat[train_idx],
                train_y=y_tr,
                val_num=X_num[val_idx],
                val_cat=X_cat[val_idx],
                cat_cardinalities=cat_cardinalities,
                epochs=20,
                device=device
            )
            oof_nn[val_idx] = tabm_val_preds
            
            tabm_m.eval()
            with torch.no_grad():
                t_num = torch.tensor(X_test_num, dtype=torch.float32).to(device)
                t_cat = torch.tensor(X_test_cat, dtype=torch.long).to(device)
                t_logits = tabm_m(t_num, t_cat)
                test_preds_nn += torch.sigmoid(t_logits).mean(dim=1).cpu().numpy() / n_splits
                
            torch.save(tabm_m.state_dict(), os.path.join(model_dir, f"tabm_fold_{fold}.pth"))
        else:
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=200,
                random_state=seed + fold
            )
            X_tr_nn = np.column_stack([X_num[train_idx], X_cat[train_idx]])
            X_va_nn = np.column_stack([X_num[val_idx], X_cat[val_idx]])
            X_te_nn = np.column_stack([X_test_num, X_test_cat])
            
            mlp.fit(X_tr_nn, y_tr)
            oof_nn[val_idx] = mlp.predict_proba(X_va_nn)[:, 1]
            test_preds_nn += mlp.predict_proba(X_te_nn)[:, 1] / n_splits
            with open(os.path.join(model_dir, f"mlp_fold_{fold}.pkl"), "wb") as f:
                pickle.dump(mlp, f)
                
    print("\n" + "=" * 60)
    print("Out-Of-Fold Evaluation Summary")
    print("=" * 60)
    evaluate_predictions(y, oof_lgb, prefix="LightGBM")
    evaluate_predictions(y, oof_xgb, prefix="XGBoost")
    evaluate_predictions(y, oof_cat, prefix="CatBoost")
    nn_name = "TabM PyTorch Ensemble" if HAS_TORCH else "MLP Neural Net Fallback"
    evaluate_predictions(y, oof_nn, prefix=nn_name)
    
    # 5. Stacking & Percentile Rank Transformation
    print("\n[Stacking] Training Ridge Meta-Learner on Rank-Transformed Logits...")
    oof_matrix = np.column_stack([
        rank_transform(oof_lgb),
        rank_transform(oof_xgb),
        rank_transform(oof_cat),
        rank_transform(oof_nn)
    ])
    
    test_matrix = np.column_stack([
        rank_transform(test_preds_lgb),
        rank_transform(test_preds_xgb),
        rank_transform(test_preds_cat),
        rank_transform(test_preds_nn)
    ])
    
    meta_learner = Ridge(alpha=1.0, random_state=seed)
    meta_learner.fit(oof_matrix, y)
    
    with open(os.path.join(model_dir, "meta_learner.pkl"), "wb") as f:
        pickle.dump(meta_learner, f)
        
    oof_final = meta_learner.predict(oof_matrix)
    oof_final = np.clip(oof_final, 0.0, 1.0)
    
    evaluate_predictions(y, oof_final, prefix="HyTab-Addict Final Stacked Model")
    print(f"[Saver] Best fold models & Stacking Meta-Learner saved to '{model_dir}/'.")
    
    # 6. Save Submission File
    test_final = meta_learner.predict(test_matrix)
    test_final = np.clip(test_final, 0.0001, 0.9999)
    
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df = pd.DataFrame({
        "id": raw_test["id"],
        "addicted_label": np.round(test_final, 6)
    })
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n[Submission] Successfully generated '{submission_path}' ({len(submission_df)} rows).")
    print("Sample Output:")
    print(submission_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & Save Best HyTab-Addict Models")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory path containing train.csv and test.csv")
    parser.add_argument("--train-path", type=str, default=None, help="Path to train.csv file")
    parser.add_argument("--test-path", type=str, default=None, help="Path to test.csv file")
    parser.add_argument("--output-dir", type=str, default="submissions", help="Directory where submission.csv will be saved")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory where best models will be saved")
    args = parser.parse_args()
    
    run_pipeline(
        n_splits=args.folds,
        data_dir=args.data_dir,
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        model_dir=args.model_dir
    )
