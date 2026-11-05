# """
# NETWORK TROUBLESHOOTING ML TRAINING PIPELINE
# =============================================
# Complete training pipeline with:
# - Data preprocessing
# - Decision Tree training
# - Model evaluation
# - Feature importance analysis
# - Model persistence

# Author: AI Assistant
# Date: 2026
# """

# import pandas as pd
# import numpy as np
# import pickle
# import json
# from datetime import datetime
# from pathlib import Path

# # ML Libraries
# from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
# from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     confusion_matrix, classification_report, ConfusionMatrixDisplay
# )

# import warnings
# warnings.filterwarnings('ignore')

# # Optional: For visualization (comment out if not available)
# try:
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     PLOTTING_AVAILABLE = True
# except ImportError:
#     PLOTTING_AVAILABLE = False
#     print("[Warning] Matplotlib/Seaborn not available - skipping plots")


# # ============================================================================
# # STEP 1: LOAD DATASET
# # ============================================================================

# def load_dataset(csv_path='network_dataset_expanded.csv'):
#     """
#     Load network dataset from CSV
    
#     Parameters:
#     -----------
#     csv_path : str
#         Path to CSV file
    
#     Returns:
#     --------
#     pd.DataFrame : Loaded dataset
#     """
    
#     # print("\n" + "=" * 70)
#     # print("STEP 1: LOADING DATASET")
#     # print("=" * 70)
    
#     # try:
#     #     df = pd.read_csv(csv_path)
#     #     print(f"\n✓ Dataset loaded successfully")
#     #     print(f"  Shape: {df.shape}")
#     #     print(f"  Columns: {df.columns.tolist()}")
#     #     return df
#     # except FileNotFoundError:
#     #     print(f"\n✗ Dataset not found at {csv_path}")
#     #     print(f"  Generating dataset from scratch...")
        
#     #     # Generate dataset if not found
#     #     from network_troubleshooting_dataset import create_expanded_dataset
#     #     df = create_expanded_dataset(num_samples=1000, save_to_csv=True)
#     #     return df
#     print("\n" + "=" * 70)
#     print("STEP 1: LOADING DATASET")
#     print("=" * 70)
    
#     if not Path(csv_path).exists():
#         raise FileNotFoundError(
#             f"\n❌ Dataset NOT found at: {csv_path}\n"
#             f"👉 Please make sure your dataset file exists.\n"
#             f"👉 No automatic generation is allowed."
#         )
    
#     df = pd.read_csv(csv_path)
#     print(f"\n✓ Dataset loaded successfully")
#     print(f"  Shape: {df.shape}")
#     print(f"  Columns: {df.columns.tolist()}")
    
#     return df



# # ============================================================================
# # STEP 2: DATA PREPROCESSING
# # ============================================================================

# def preprocess_data(df):
#     """
#     Preprocess dataset for ML training
    
#     Steps:
#     ------
#     1. Drop non-feature columns
#     2. Handle categorical features
#     3. Separate features and target
#     4. Encode categorical variables
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Raw dataset
    
#     Returns:
#     --------
#     tuple : (X_processed, y, encoders)
#     """
    
#     print("\n" + "=" * 70)
#     print("STEP 2: DATA PREPROCESSING")
#     print("=" * 70)
    
#     print("\n[2.1] Dropping non-feature columns...")
    
#     # Columns to drop (metadata, text descriptions, solutions)
#     drop_cols = ['symptom_text', 'solutions', 'timestamp', 'dataset_version', 'severity']
#     df_processed = df.drop(columns=drop_cols, errors='ignore')
    
#     print(f"  Dropped: {drop_cols}")
#     print(f"  Remaining columns: {df_processed.columns.tolist()}")
    
#     # ====================================================================
#     print("\n[2.2] Separating features and target...")
    
#     X = df_processed.drop(columns=['diagnosis'])
#     y = df_processed['diagnosis']
    
#     print(f"  Features (X) shape: {X.shape}")
#     print(f"  Target (y) shape: {y.shape}")
#     print(f"  Feature columns: {X.columns.tolist()}")
    
#     # ====================================================================
#     print("\n[2.3] Encoding categorical features...")
    
#     encoders = {}
#     X_encoded = X.copy()
    
#     # Identify categorical columns
#     categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
#     print(f"  Categorical columns: {categorical_cols}")
    
#     # Encode categorical features
#     for col in categorical_cols:
#         le = LabelEncoder()
#         X_encoded[col] = le.fit_transform(X[col])
#         encoders[col] = le
        
#         print(f"    {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
#     # Verify all features are numeric
#     print(f"\n  Final feature dtypes:")
#     print(f"    {X_encoded.dtypes.to_dict()}")
    
#     # ====================================================================
#     print("\n[2.4] Feature Summary Statistics")
#     print("-" * 70)
#     print(X_encoded.describe().to_string())
    
#     return X_encoded, y, encoders


# # ============================================================================
# # STEP 3: TRAIN-TEST SPLIT
# # ============================================================================

# def split_data(X, y, test_size=0.2, random_state=42):
#     """
#     Split data into training and testing sets
    
#     Parameters:
#     -----------
#     X : pd.DataFrame
#         Features
#     y : pd.Series
#         Target
#     test_size : float
#         Proportion of test data (0.2 = 20%)
#     random_state : int
#         Random seed
    
#     Returns:
#     --------
#     tuple : (X_train, X_test, y_train, y_test)
#     """
    
#     print("\n" + "=" * 70)
#     print("STEP 3: TRAIN-TEST SPLIT")
#     print("=" * 70)
    
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y
#     )
    
#     print(f"\n✓ Data split completed")
#     print(f"  Training set size: {X_train.shape[0]} ({100*(1-test_size):.0f}%)")
#     print(f"  Testing set size: {X_test.shape[0]} ({100*test_size:.0f}%)")
    
#     print(f"\n  Training set class distribution:")
#     for diagnosis, count in y_train.value_counts().items():
#         pct = 100 * count / len(y_train)
#         print(f"    {diagnosis:25s}: {count:3d} ({pct:5.1f}%)")
    
#     print(f"\n  Testing set class distribution:")
#     for diagnosis, count in y_test.value_counts().items():
#         pct = 100 * count / len(y_test)
#         print(f"    {diagnosis:25s}: {count:3d} ({pct:5.1f}%)")
    
#     return X_train, X_test, y_train, y_test


# # ============================================================================
# # STEP 4: MODEL TRAINING
# # ============================================================================

# def train_decision_tree(X_train, y_train, max_depth=5):
#     """
#     Train Decision Tree classifier
    
#     Parameters:
#     -----------
#     X_train : pd.DataFrame
#         Training features
#     y_train : pd.Series
#         Training target
#     max_depth : int
#         Maximum tree depth (controls complexity)
    
#     Returns:
#     --------
#     DecisionTreeClassifier : Trained model
#     """
    
#     print("\n" + "=" * 70)
#     print("STEP 4: MODEL TRAINING")
#     print("=" * 70)
    
#     print(f"\n[4.1] Training Decision Tree Classifier")
#     print(f"  Parameters:")
#     print(f"    max_depth = {max_depth}")
#     print(f"    criterion = gini (Gini impurity)")
#     print(f"    random_state = 42")
    
#     # Initialize and train model
#     dt_model = DecisionTreeClassifier(
#         max_depth=max_depth,
#         criterion='gini',
#         random_state=42,
#         min_samples_split=5,
#         min_samples_leaf=2
#     )
    
#     dt_model.fit(X_train, y_train)
    
#     print(f"\n✓ Model training completed")
#     print(f"  Tree depth: {dt_model.get_depth()}")
#     print(f"  Number of leaves: {dt_model.get_n_leaves()}")
#     print(f"  Number of features used: {dt_model.n_features_in_}")
    
#     # Training accuracy
#     train_accuracy = dt_model.score(X_train, y_train)
#     print(f"  Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
#     return dt_model


# # ============================================================================
# # STEP 5: MODEL EVALUATION
# # ============================================================================

# def evaluate_model(model, X_train, X_test, y_train, y_test):
#     """
#     Comprehensive model evaluation
    
#     Metrics:
#     --------
#     - Accuracy
#     - Precision, Recall, F1-Score
#     - Cross-validation scores
#     - Confusion matrix
#     - Classification report
    
#     Parameters:
#     -----------
#     model : sklearn.tree.DecisionTreeClassifier
#         Trained model
#     X_train, X_test : pd.DataFrame
#         Training and testing features
#     y_train, y_test : pd.Series
#         Training and testing targets
#     """
    
#     print("\n" + "=" * 70)
#     print("STEP 5: MODEL EVALUATION")
#     print("=" * 70)
    
#     # ====================================================================
#     print("\n[5.1] Test Set Performance Metrics")
#     print("-" * 70)
    
#     y_pred = model.predict(X_test)
    
#     test_accuracy = accuracy_score(y_test, y_pred)
#     test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
#     test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
#     test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
#     print(f"\n  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
#     print(f"  Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
#     print(f"  Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
#     print(f"  F1-Score:  {test_f1:.4f}")
    
#     # ====================================================================
#     print("\n[5.2] Cross-Validation Scores (5-Fold)")
#     print("-" * 70)
    
#     cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
#     print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
#     print(f"  Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
#     # ====================================================================
#     print("\n[5.3] Per-Class Performance")
#     print("-" * 70)
    
#     print("\n" + classification_report(y_test, y_pred))
    
#     # ====================================================================
#     print("\n[5.4] Confusion Matrix")
#     print("-" * 70)
    
#     cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
#     # Pretty print confusion matrix
#     print("\n  Predicted ↓ / Actual →")
    
#     # Header row
#     header_width = 18
#     header = "".rjust(header_width)
#     for label in model.classes_:
#         header += f"{label[:8]:>10}"
#     print(header)
    
#     # Data rows
#     for i, true_label in enumerate(model.classes_):
#         row = f"{true_label[:15]:17s}"
#         for j in range(len(model.classes_)):
#             row += f"{cm[i, j]:>10d}"
#         print(row)
    
#     return {
#         'accuracy': test_accuracy,
#         'precision': test_precision,
#         'recall': test_recall,
#         'f1': test_f1,
#         'cv_mean': cv_scores.mean(),
#         'cv_std': cv_scores.std(),
#         'confusion_matrix': cm
#     }


# # ============================================================================
# # STEP 6: FEATURE IMPORTANCE ANALYSIS
# # ============================================================================

# def analyze_feature_importance(model, feature_names):
#     """
#     Analyze and display feature importance
    
#     Parameters:
#     -----------
#     model : sklearn.tree.DecisionTreeClassifier
#         Trained model
#     feature_names : list
#         List of feature names
#     """
    
#     print("\n" + "=" * 70)
#     print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
#     print("=" * 70)
    
#     # Get feature importances
#     importances = model.feature_importances_
    
#     # Create dataframe for better visualization
#     feature_importance_df = pd.DataFrame({
#         'feature': feature_names,
#         'importance': importances
#     }).sort_values('importance', ascending=False)
    
#     print("\n  Feature Importance Ranking:")
#     print("-" * 50)
    
#     for idx, row in feature_importance_df.iterrows():
#         feature = row['feature']
#         importance = row['importance']
#         bar_width = int(importance * 50)
#         bar = "█" * bar_width
#         print(f"  {feature:20s} | {importance:.4f} | {bar}")
    
#     return feature_importance_df


# # ============================================================================
# # STEP 7: TREE VISUALIZATION
# # ============================================================================

# def visualize_tree(model, feature_names, class_names, output_path='decision_tree.png'):
#     """
#     Visualize and save decision tree diagram
    
#     Parameters:
#     -----------
#     model : sklearn.tree.DecisionTreeClassifier
#         Trained model
#     feature_names : list
#         Feature names
#     class_names : list
#         Class labels
#     output_path : str
#         Path to save visualization
#     """
    
#     if not PLOTTING_AVAILABLE:
#         print("\n[Warning] Matplotlib not available - skipping tree visualization")
#         return
    
#     print("\n" + "=" * 70)
#     print("STEP 7: TREE VISUALIZATION")
#     print("=" * 70)
    
#     plt.figure(figsize=(20, 10))
#     plot_tree(
#         model,
#         feature_names=feature_names,
#         class_names=class_names,
#         filled=True,
#         rounded=True,
#         fontsize=10
#     )
    
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     print(f"\n✓ Tree visualization saved to: {output_path}")
#     plt.close()


# # ============================================================================
# # STEP 8: EXPORT TREE AS TEXT
# # ============================================================================

# def export_tree_text(model, feature_names, output_path='decision_tree.txt'):
#     """
#     Export tree structure as human-readable text
    
#     Parameters:
#     -----------
#     model : sklearn.tree.DecisionTreeClassifier
#         Trained model
#     feature_names : list
#         Feature names
#     output_path : str
#         Path to save text
#     """
    
#     print("\n" + "=" * 70)
#     print("STEP 8: TREE TEXT EXPORT")
#     print("=" * 70)
    
#     tree_rules = export_text(model, feature_names=feature_names)
    
#     # Save to file
#     with open(output_path, 'w') as f:
#         f.write("DECISION TREE RULES\n")
#         f.write("=" * 70 + "\n\n")
#         f.write(tree_rules)
    
#     print(f"\n✓ Tree rules exported to: {output_path}")
#     print(f"\nPreview (first 50 lines):")
#     print("-" * 70)
#     lines = tree_rules.split('\n')
#     for line in lines[:50]:
#         print(line)


# # ============================================================================
# # STEP 9: SAVE MODEL & ENCODERS
# # ============================================================================

# def save_model(model, encoders, feature_names, class_names, output_dir='models'):
#     """
#     Save trained model and encoders for production deployment
    
#     Parameters:
#     -----------
#     model : sklearn.tree.DecisionTreeClassifier
#         Trained model
#     encoders : dict
#         Feature encoders
#     feature_names : list
#         Feature names
#     class_names : list
#         Class labels
#     output_dir : str
#         Directory to save models
#     """
    
#     print("\n" + "=" * 70)
#     print("STEP 9: MODEL PERSISTENCE")
#     print("=" * 70)
    
#     # Create output directory
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True)
    
#     # Save model
#     model_file = output_path / 'dt_model.pkl'
#     with open(model_file, 'wb') as f:
#         pickle.dump(model, f)
#     print(f"\n✓ Model saved to: {model_file}")
    
#     # Save encoders
#     encoders_file = output_path / 'encoders.pkl'
#     with open(encoders_file, 'wb') as f:
#         pickle.dump(encoders, f)
#     print(f"✓ Encoders saved to: {encoders_file}")
    

#     # Save metadata
#     metadata = {
#         'model_type': 'DecisionTreeClassifier',
#         'feature_names': [str(f) for f in feature_names],
#         'class_names': [str(c) for c in class_names],
#         'encoders': {k: {'classes_': [str(x) for x in v.classes_]} for k, v in encoders.items()},
#         'tree_depth': int(model.get_depth()),
#         'n_leaves': int(model.get_n_leaves()),
#         'trained_at': datetime.now().isoformat(),
#         'feature_importances': {str(f): float(i) for f, i in zip(feature_names, model.feature_importances_)}
#     }
    
#     metadata_file = output_path / 'metadata.json'
#     with open(metadata_file, 'w') as f:
#         json.dump(metadata, f, indent=4)
    
#     return output_path


# # ============================================================================
# # STEP 10: LOAD SAVED MODEL
# # ============================================================================

# def load_model(model_dir='models'):
#     """
#     Load trained model, encoders, and metadata from disk
    
#     Parameters:
#     -----------
#     model_dir : str
#         Directory containing saved models
    
#     Returns:
#     --------
#     tuple : (model, encoders, metadata)
#     """
    
#     print("\n" + "=" * 70)
#     print("LOADING SAVED MODEL")
#     print("=" * 70)
    
#     model_path = Path(model_dir)
    
#     # Load model
#     with open(model_path / 'dt_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     print(f"\n✓ Model loaded from: {model_path / 'dt_model.pkl'}")
    
#     # Load encoders
#     with open(model_path / 'encoders.pkl', 'rb') as f:
#         encoders = pickle.load(f)
#     print(f"✓ Encoders loaded from: {model_path / 'encoders.pkl'}")
    
#     # Load metadata
#     with open(model_path / 'metadata.json', 'r') as f:
#         metadata = json.load(f)
#     print(f"✓ Metadata loaded from: {model_path / 'metadata.json'}")
    
#     return model, encoders, metadata


# # ============================================================================
# # MAIN TRAINING PIPELINE
# # ============================================================================

# def main():
#     """
#     Execute complete training pipeline
#     """
    
#     print("\n")
#     print("╔" + "=" * 68 + "╗")
#     print("║" + " " * 68 + "║")
#     print("║" + "  NETWORK TROUBLESHOOTING ML TRAINING PIPELINE".center(68) + "║")
#     print("║" + "  Complete ML workflow: Data → Train → Evaluate → Deploy".center(68) + "║")
#     print("║" + " " * 68 + "║")
#     print("╚" + "=" * 68 + "╝")
    
#     # Step 1: Load dataset
#     df = load_dataset('network_dataset.csv')
    
#     # Step 2: Preprocess data
#     X, y, encoders = preprocess_data(df)
    
#     # Step 3: Train-test split
#     X_train, X_test, y_train, y_test = split_data(X, y)
    
#     # Step 4: Train model
#     model = train_decision_tree(X_train, y_train, max_depth=5)
    
#     # Step 5: Evaluate model
#     metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
#     # Step 6: Feature importance
#     feature_importance_df = analyze_feature_importance(model, X.columns.tolist())
    
#     # Step 7: Visualize tree
#     visualize_tree(model, X.columns.tolist(), model.classes_.tolist())
    
#     # Step 8: Export tree rules
#     export_tree_text(model, X.columns.tolist())
    
#     # Step 9: Save model
#     output_path = save_model(model, encoders, X.columns.tolist(), model.classes_.tolist())
    
#     # Final summary
#     print("\n" + "=" * 70)
#     print("TRAINING PIPELINE COMPLETE")
#     print("=" * 70)
#     print(f"\n✓ Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
#     print(f"✓ Cross-validation Score: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
#     print(f"✓ Models saved to: {output_path}")
#     print(f"\nNext step: Use model for inference with Streamlit app!")


# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# FEATURES
# =========================
NUMERIC_FEATURES = [
    "ping_gateway", "has_ip", "ping_ip", "ping_domain",
    "ip_conflict", "arp_table_ok", "subnet_matches_gw",
    "dns_response_time_ms", "packet_loss_pct", "traceroute_hops"
]

CATEGORICAL_FEATURES = ["network_type", "os_type"]
TEXT_FEATURE = "symptom_text"

DROP_COLS = ["solutions", "timestamp", "dataset_version", "severity"]


# =========================
# LOAD DATA
# =========================
def load_dataset(path="network_dataset_v3.csv"):
    df = pd.read_csv(path)
    return df


# =========================
# PREPROCESS
# =========================
def preprocess(df):
    df = df.drop(columns=DROP_COLS, errors="ignore").copy()

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    return X, y, encoders


# =========================
# BUILD PIPELINE
# =========================
def build_pipeline():

    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=2
        ), TEXT_FEATURE),

        ("num", "passthrough", NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    ])

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("features", preprocessor),
        ("clf", model)
    ])

    return pipeline


# =========================
# TRAIN
# =========================
def train_model(X_train, y_train):
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # SAFE accuracy calculation
    y_pred = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)

    print(f"\nTrain Accuracy: {train_acc:.4f}")

    return pipeline


# =========================
# EVALUATE
# =========================
def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross validation
    cv = cross_val_score(pipeline, X_test, y_test, cv=5)
    print(f"\nCV Score: {cv.mean():.4f} (+/- {cv.std():.4f})")

    return acc


# =========================
# SAVE
# =========================
def save_model(pipeline, encoders):
    Path("models").mkdir(exist_ok=True)

    with open("models/pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    with open("models/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    print("\nModel saved!")


# =========================
# MAIN
# =========================
def main():
    print("\n===== NETWORK TROUBLESHOOTING ML (FINAL) =====")

    df = load_dataset()

    X, y, encoders = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = train_model(X_train, y_train)

    evaluate(pipeline, X_test, y_test)

    save_model(pipeline, encoders)


if __name__ == "__main__":
    main()

