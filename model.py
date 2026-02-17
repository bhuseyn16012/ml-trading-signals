"""
model.py — XGBoost Model Training with Walk-Forward Validation

Trains an XGBoost classifier to predict next-day price direction
using expanding-window walk-forward validation to prevent look-ahead bias.

Walk-Forward Process:
    Fold 1: Train [Year 1]         → Test [Year 2]
    Fold 2: Train [Years 1-2]      → Test [Year 3]
    Fold 3: Train [Years 1-2-3]    → Test [Year 4]
    Fold 4: Train [Years 1-2-3-4]  → Test [Year 5]

Usage:
    from model import train_walk_forward
    results = train_walk_forward(df, feature_cols)
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_walk_forward(
    df: pd.DataFrame,
    feature_cols: list,
    n_splits: int = 5,
    confidence_threshold: float = 0.5,
) -> dict:
    """
    Train XGBoost with expanding-window walk-forward validation.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame from features.build_features().
    feature_cols : list
        Column names to use as model inputs (from features.get_feature_columns()).
    n_splits : int
        Number of walk-forward folds (default: 5).
    confidence_threshold : float
        Minimum predicted probability to generate a BUY signal (default: 0.5).
        Higher values = fewer but more confident trades.

    Returns
    -------
    dict with keys:
        - "predictions": DataFrame with Date, Actual, Predicted, Probability
        - "feature_importance": DataFrame ranking features by importance
        - "fold_metrics": list of per-fold accuracy scores
        - "model": the final trained XGBoost model
    """
    print(f"[model] Starting walk-forward validation with {n_splits} folds...")
    print(f"[model] Using {len(feature_cols)} features, "
          f"confidence threshold: {confidence_threshold}")

    # --- Split data into folds ---
    fold_size = len(df) // n_splits
    all_predictions = []
    fold_metrics = []
    feature_importance_sum = np.zeros(len(feature_cols))

    for fold in range(1, n_splits):
        # Expanding window: train on all data up to this fold
        train_end = fold * fold_size
        test_end = min((fold + 1) * fold_size, len(df))

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train = train_df[feature_cols]
        y_train = train_df["Target"]
        X_test = test_df[feature_cols]
        y_test = test_df["Target"]

        # --- Train XGBoost ---
        # Conservative hyperparameters to avoid overfitting
        # We intentionally do NOT tune these on test data (that would be leakage)
        model = XGBClassifier(
            n_estimators=200,       # Number of boosting rounds
            max_depth=4,            # Shallow trees → less overfitting
            learning_rate=0.05,     # Small steps → better generalization
            subsample=0.8,          # Use 80% of data per tree (randomness helps)
            colsample_bytree=0.8,   # Use 80% of features per tree
            reg_alpha=0.1,          # L1 regularization (sparsity)
            reg_lambda=1.0,         # L2 regularization (smoothness)
            random_state=42,        # Reproducibility
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

        model.fit(X_train, y_train)

        # --- Predict probabilities ---
        probabilities = model.predict_proba(X_test)[:, 1]  # P(up)
        predictions = (probabilities >= confidence_threshold).astype(int)

        # --- Store results ---
        fold_pred = pd.DataFrame({
            "Date": test_df["Date"].values,
            "Close": test_df["Close"].values,
            "Actual": y_test.values,
            "Predicted": predictions,
            "Probability": probabilities,
        })
        all_predictions.append(fold_pred)

        # Track accuracy per fold
        acc = accuracy_score(y_test, predictions)
        fold_metrics.append({
            "fold": fold,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "accuracy": acc,
            "up_pct_actual": y_test.mean(),
        })

        # Accumulate feature importance
        feature_importance_sum += model.feature_importances_

        print(f"  Fold {fold}: Train={len(train_df):,} rows, "
              f"Test={len(test_df):,} rows, Accuracy={acc:.1%}")

    # --- Combine all predictions ---
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # --- Average feature importance across folds ---
    feature_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": feature_importance_sum / (n_splits - 1),
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # --- Summary ---
    overall_acc = accuracy_score(predictions_df["Actual"], predictions_df["Predicted"])
    print(f"\n[model] Walk-forward complete.")
    print(f"  Overall accuracy: {overall_acc:.1%}")
    print(f"  Total predictions: {len(predictions_df):,}")
    print(f"  Top 5 features:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"    - {row['Feature']}: {row['Importance']:.4f}")

    return {
        "predictions": predictions_df,
        "feature_importance": feature_importance,
        "fold_metrics": fold_metrics,
        "model": model,  # Final fold's model (trained on most data)
    }


# ---- Quick test when run directly ----
if __name__ == "__main__":
    from data_loader import load_data
    from features import build_features, get_feature_columns

    df = load_data("META", period="5y")
    df = build_features(df)
    feature_cols = get_feature_columns(df)

    results = train_walk_forward(df, feature_cols)

    print("\nFold-by-fold metrics:")
    for m in results["fold_metrics"]:
        print(f"  Fold {m['fold']}: Acc={m['accuracy']:.1%}, "
              f"Actual up%={m['up_pct_actual']:.1%}")

    print("\nAll features ranked by importance:")
    print(results["feature_importance"].to_string(index=False))
