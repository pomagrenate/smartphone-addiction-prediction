import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

NUMERICAL_COLS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "stress_level",
    "academic_work_impact",
]

CATEGORICAL_COLS = ["gender"]

ENGINEERED_COLS = [
    "unproductive_usage_ratio",
    "sleep_disruption_index",
    "notification_density",
    "session_duration_est",
    "weekend_excess_time",
    "distress_score",
    "screen_to_work_ratio",
    "leisure_intensity",
    "app_engagement_rate",
    "passive_vs_active_ratio",
    "sleep_deficit",
    "addiction_risk",
    "weekend_to_weekday_ratio",
    "stress_screen_interaction",
    "productivity_fraction",
]


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific behavioral metrics from raw telemetry columns."""
    df = df.copy()
    
    # Coerce all numerical columns to float, handling potential string/object dtypes or ordinal text
    text_map = {'low': 1, 'medium': 2, 'high': 3, 'very high': 4, 'extreme': 5}
    for col in NUMERICAL_COLS:
        if col in df.columns:
            if df[col].dtype == 'object':
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.isna().any():
                    mapped = df[col].astype(str).str.lower().map(text_map)
                    converted = converted.fillna(mapped)
                df[col] = converted.fillna(0).astype(float)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    eps = 1e-5
    screen_time = df["daily_screen_time_hours"].clip(lower=0.1)
    
    # 1. Unproductive Ratio
    unproductive_hours = df["social_media_hours"] + df["gaming_hours"]
    df["unproductive_usage_ratio"] = unproductive_hours / (screen_time + eps)
    
    # 2. Sleep Disruption Index
    df["sleep_disruption_index"] = screen_time / (df["sleep_hours"].clip(lower=0.5) + eps)
    
    # 3. Notification Density
    df["notification_density"] = df["notifications_per_day"] / (screen_time + eps)
    
    # 4. Session Duration Estimate (Minutes per open)
    df["session_duration_est"] = (screen_time * 60.0) / (df["app_opens_per_day"].clip(lower=1) + 1.0)
    
    # 5. Weekend Excess Screen Time
    df["weekend_excess_time"] = df["weekend_screen_time"] - screen_time
    
    # 6. Psychological Distress Score
    df["distress_score"] = df["stress_level"] * df["academic_work_impact"]
    
    # 7. Screen time relative to productive work/study
    df["screen_to_work_ratio"] = screen_time / (df["work_study_hours"].clip(lower=0.1) + eps)
    
    # 8. Digital leisure intensity (social + gaming per hour of screen)
    df["leisure_intensity"] = (df["social_media_hours"] + df["gaming_hours"]) / (screen_time + eps) * df["stress_level"]
    
    # 9. App engagement rate (opens per hour)
    df["app_engagement_rate"] = df["app_opens_per_day"] / (screen_time + eps)
    
    # 10. Passive usage estimate (notifications vs app opens ratio)
    df["passive_vs_active_ratio"] = df["notifications_per_day"] / (df["app_opens_per_day"].clip(lower=1) + eps)
    
    # 11. Sleep efficiency (screen time cuts into 8h baseline)
    df["sleep_deficit"] = (8.0 - df["sleep_hours"]).clip(lower=0)
    
    # 12. Combined addiction risk proxy
    df["addiction_risk"] = (
        df["social_media_hours"] * 0.4 +
        df["gaming_hours"] * 0.3 +
        df["stress_level"] * 0.2 -
        df["sleep_hours"] * 0.1
    )
    
    # 13. Weekend vs weekday usage balance
    df["weekend_to_weekday_ratio"] = df["weekend_screen_time"] / (screen_time + eps)
    
    # 14. Stress-amplified screen time
    df["stress_screen_interaction"] = screen_time * df["stress_level"]
    
    # 15. Productivity time fraction
    df["productivity_fraction"] = df["work_study_hours"] / (screen_time + eps)
    
    return df


def preprocess_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Preprocess train and test dataframes, creating engineered features and encoding categoricals."""
    train_df = create_engineered_features(train_df)
    test_df = create_engineered_features(test_df)
    
    all_num_cols = NUMERICAL_COLS + ENGINEERED_COLS
    
    # Label encode categorical variables
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = test_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)
        encoders[col] = le
        
    # Scale numerical features for neural models
    scaler = StandardScaler()
    train_num_scaled = scaler.fit_transform(train_df[all_num_cols])
    test_num_scaled = scaler.transform(test_df[all_num_cols])
    
    scaled_num_cols = [f"{col}_scaled" for col in all_num_cols]
    
    for i, col_name in enumerate(scaled_num_cols):
        train_df[col_name] = train_num_scaled[:, i]
        test_df[col_name] = test_num_scaled[:, i]
        
    feature_cols = all_num_cols + CATEGORICAL_COLS
    
    return train_df, test_df, feature_cols, scaled_num_cols, CATEGORICAL_COLS
