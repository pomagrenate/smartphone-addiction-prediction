import os
import pandas as pd
import numpy as np

RAW_DATA_DIR = os.path.join("data", "raw")
TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(RAW_DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(RAW_DATA_DIR, "sample_submission.csv")


def generate_synthetic_data(num_train: int = 5000, num_test: int = 1000, random_state: int = 42) -> None:
    """Generate realistic synthetic tabular datasets for testing the pipeline if raw data is missing."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    np.random.seed(random_state)
    
    def generate_features(n_samples: int, is_train: bool = True):
        ids = np.arange(100000, 100000 + n_samples) if is_train else np.arange(200000, 200000 + n_samples)
        age = np.random.randint(18, 65, size=n_samples)
        daily_screen_time = np.random.exponential(scale=3.5, size=n_samples) + 0.5
        daily_screen_time = np.clip(daily_screen_time, 0.5, 16.0)
        
        social_media = daily_screen_time * np.random.uniform(0.1, 0.6, size=n_samples)
        gaming = daily_screen_time * np.random.uniform(0.0, 0.4, size=n_samples)
        work_study = daily_screen_time * np.random.uniform(0.1, 0.5, size=n_samples)
        
        sleep = np.clip(8.5 - 0.25 * daily_screen_time + np.random.normal(0, 1, n_samples), 3.0, 10.0)
        notifications = np.random.poisson(lam=120, size=n_samples) * (daily_screen_time / 4.0)
        app_opens = np.random.poisson(lam=60, size=n_samples) * (daily_screen_time / 4.0)
        weekend_screen_time = daily_screen_time * np.random.uniform(0.9, 1.4, size=n_samples)
        
        genders = np.random.choice(["Male", "Female", "Other"], size=n_samples, p=[0.48, 0.48, 0.04])
        stress_level = np.clip(np.round(1 + 0.5 * daily_screen_time + np.random.normal(0, 1.5, n_samples)), 1, 10)
        academic_impact = np.clip(np.round(1 + 0.4 * daily_screen_time + np.random.normal(0, 1.5, n_samples)), 1, 10)
        
        df = pd.DataFrame({
            "id": ids,
            "age": age,
            "daily_screen_time_hours": np.round(daily_screen_time, 2),
            "social_media_hours": np.round(social_media, 2),
            "gaming_hours": np.round(gaming, 2),
            "work_study_hours": np.round(work_study, 2),
            "sleep_hours": np.round(sleep, 2),
            "notifications_per_day": np.round(notifications, 0),
            "app_opens_per_day": np.round(app_opens, 0),
            "weekend_screen_time": np.round(weekend_screen_time, 2),
            "gender": genders,
            "stress_level": stress_level,
            "academic_work_impact": academic_impact
        })
        
        if is_train:
            logit = (
                0.35 * daily_screen_time +
                0.25 * social_media +
                0.20 * gaming -
                0.15 * sleep +
                0.15 * stress_level +
                0.15 * academic_impact - 3.5
            )
            prob = 1.0 / (1.0 + np.exp(-logit))
            df["addicted_label"] = np.round(prob, 6)
            
        return df

    print("[Loader] Raw datasets not found. Generating synthetic competition datasets...")
    train_df = generate_features(num_train, is_train=True)
    test_df = generate_features(num_test, is_train=False)
    
    sample_sub = pd.DataFrame({
        "id": test_df["id"],
        "addicted_label": 0.5
    })
    
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    sample_sub.to_csv(SAMPLE_SUB_PATH, index=False)
    print(f"[Loader] Saved synthetic datasets to '{RAW_DATA_DIR}/'.")


def load_raw_data(data_dir: str = None, train_path: str = None, test_path: str = None, sample_sub_path: str = None):
    """Load train, test, and sample submission dataframes from custom or auto-detected Kaggle paths."""
    tr_path, te_path, sub_path = None, None, None
    
    if train_path and test_path:
        tr_path = train_path
        te_path = test_path
        sub_path = sample_sub_path
    elif data_dir:
        tr_path = os.path.join(data_dir, "train.csv")
        te_path = os.path.join(data_dir, "test.csv")
        sub_path = os.path.join(data_dir, "sample_submission.csv")
    else:
        # Check standard Kaggle paths first
        candidate_paths = [
            "/kaggle/input/playground-series-s6e8",
            "/kaggle/input/smartphone-addiction-prediction",
            os.path.join("data", "raw")
        ]
        for p in candidate_paths:
            candidate_tr = os.path.join(p, "train.csv")
            candidate_te = os.path.join(p, "test.csv")
            if os.path.exists(candidate_tr) and os.path.exists(candidate_te):
                tr_path = candidate_tr
                te_path = candidate_te
                sub_path = os.path.join(p, "sample_submission.csv")
                print(f"[Loader] Auto-detected dataset directory: '{p}'")
                break
                
        if not tr_path:
            tr_path = TRAIN_PATH
            te_path = TEST_PATH
            sub_path = SAMPLE_SUB_PATH
            if not (os.path.exists(tr_path) and os.path.exists(te_path)):
                generate_synthetic_data()

    print(f"[Loader] Loading train dataset from: {tr_path}")
    print(f"[Loader] Loading test dataset from:  {te_path}")
    
    train_df = pd.read_csv(tr_path)
    test_df = pd.read_csv(te_path)
    sample_sub = pd.read_csv(sub_path) if (sub_path and os.path.exists(sub_path)) else None
    
    return train_df, test_df, sample_sub
