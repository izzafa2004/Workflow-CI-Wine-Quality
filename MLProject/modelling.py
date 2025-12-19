"""
MODELLING.PY - MLFLOW PROJECT VERSION
Wine Quality Classification for CI/CD Pipeline

Nama: Iklima Fatma A
Purpose: Automated model training with MLflow Project
"""

import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
import shutil
from pathlib import Path
warnings.filterwarnings('ignore')

# ==================== ARGUMENT PARSER ====================
def parse_args():
    parser = argparse.ArgumentParser(description='Train Wine Quality Model')
    
    parser.add_argument('--n_estimators', type=int, default=200,
                       help='Number of trees in random forest')
    parser.add_argument('--max_depth', type=int, default=15,
                       help='Maximum depth of trees')
    parser.add_argument('--min_samples_split', type=int, default=5,
                       help='Minimum samples required to split')
    parser.add_argument('--min_samples_leaf', type=int, default=2,
                       help='Minimum samples required at leaf')
    parser.add_argument('--class_weight', type=str, default='balanced',
                       help='Class weight strategy')
    
    return parser.parse_args()

# ==================== MAIN TRAINING FUNCTION ====================
def main():
    # Parse arguments
    args = parse_args()
    
    print("="*60)
    print("MLFLOW PROJECT - WINE QUALITY MODEL TRAINING")
    print("="*60)
    print(f"\nHyperparameters:")
    print(f"  n_estimators: {args.n_estimators}")
    print(f"  max_depth: {args.max_depth}")
    print(f"  min_samples_split: {args.min_samples_split}")
    print(f"  min_samples_leaf: {args.min_samples_leaf}")
    print(f"  class_weight: {args.class_weight}")
    
    # ==================== LOAD DATA ====================
    print("\n" + "="*60)
    print("LOADING PROCESSED DATA")
    print("="*60)
    
    # Determine data path (works both locally and in CI)
    data_dir = "dataset_preprocessing"
    if not os.path.exists(data_dir):
        data_dir = "MLProject/dataset_preprocessing"
    
    # Load datasets
    train_data = pd.read_csv(os.path.join(data_dir, "wine_train_processed.csv"))
    val_data = pd.read_csv(os.path.join(data_dir, "wine_val_processed.csv"))
    test_data = pd.read_csv(os.path.join(data_dir, "wine_test_processed.csv"))
    
    X_train = train_data.drop('quality_binary', axis=1)
    y_train = train_data['quality_binary']
    X_val = val_data.drop('quality_binary', axis=1)
    y_val = val_data['quality_binary']
    X_test = test_data.drop('quality_binary', axis=1)
    y_test = test_data['quality_binary']
    
    print(f"✅ Data loaded successfully!")
    print(f"   Training: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # ==================== START MLFLOW RUN ====================
    with mlflow.start_run() as run:
        
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        print(f"\n🔬 MLflow Run ID: {run.info.run_id}")
        
        # ========== LOG PARAMETERS ==========
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("training_date", datetime.now().strftime('%Y-%m-%d'))
        
        # ========== TRAIN MODEL ==========
        print("\n📊 Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=args.class_weight,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        print("✅ Model training completed!")
        
        # ========== PREDICTIONS ==========
        print("\n🔮 Making predictions...")
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # ========== CALCULATE AND LOG METRICS ==========
        print("\n📈 Calculating metrics...")
        
        def log_metrics(y_true, y_pred, y_proba, prefix):
            metrics = {
                f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
                f'{prefix}_precision': precision_score(y_true, y_pred),
                f'{prefix}_recall': recall_score(y_true, y_pred),
                f'{prefix}_f1': f1_score(y_true, y_pred),
                f'{prefix}_roc_auc': roc_auc_score(y_true, y_proba)
            }
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            print(f"\n{prefix.upper()} Metrics:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")
            
            return metrics
        
        train_metrics = log_metrics(y_train, y_train_pred, y_train_proba, "train")
        val_metrics = log_metrics(y_val, y_val_pred, y_val_proba, "val")
        test_metrics = log_metrics(y_test, y_test_pred, y_test_proba, "test")
        
        # ========== CREATE ARTIFACTS ==========
        print("\n" + "="*60)
        print("CREATING ARTIFACTS")
        print("="*60)
        
        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # --- Artifact 1: Confusion Matrix ---
        print("\n📊 Creating confusion matrix...")
        cm = confusion_matrix(y_test, y_test_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Bad Wine', 'Good Wine'],
                    yticklabels=['Bad Wine', 'Good Wine'])
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path)
        print(f"✅ Logged: confusion_matrix.png")
        
        # --- Artifact 2: Feature Importance ---
        print("📊 Creating feature importance...")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        
        fi_path = os.path.join(artifacts_dir, "feature_importance.png")
        plt.savefig(fi_path, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(fi_path)
        print(f"✅ Logged: feature_importance.png")
        
        # Save as CSV
        fi_csv_path = os.path.join(artifacts_dir, "feature_importance.csv")
        feature_importance.to_csv(fi_csv_path, index=False)
        mlflow.log_artifact(fi_csv_path)
        
        # --- Artifact 3: Classification Report ---
        print("📊 Creating classification report...")
        report_dict = classification_report(y_test, y_test_pred,
                                           target_names=['Bad Wine', 'Good Wine'],
                                           output_dict=True)
        
        report_path = os.path.join(artifacts_dir, "classification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        mlflow.log_artifact(report_path)
        print(f"✅ Logged: classification_report.json")
        
        # --- Artifact 4: Training Summary ---
        print("📊 Creating training summary...")
        summary = {
            "run_id": run.info.run_id,
            "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "model_type": "RandomForestClassifier",
            "hyperparameters": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_split": args.min_samples_split,
                "min_samples_leaf": args.min_samples_leaf,
                "class_weight": args.class_weight
            },
            "dataset_info": {
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "n_features": X_train.shape[1]
            },
            "test_metrics": {
                "accuracy": float(test_metrics['test_accuracy']),
                "precision": float(test_metrics['test_precision']),
                "recall": float(test_metrics['test_recall']),
                "f1_score": float(test_metrics['test_f1']),
                "roc_auc": float(test_metrics['test_roc_auc'])
            },
            "top_features": feature_importance.head(5).to_dict('records')
        }
        
        summary_path = os.path.join(artifacts_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)
        print(f"✅ Logged: training_summary.json")
        
        # ========== LOG MODEL ==========
        print("\n💾 Logging model...")
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="wine_quality_rf_ci"
        )
        print("✅ Model logged!")
        
        # ========== SAVE ARTIFACTS FOR GITHUB ACTIONS ==========
        print("\n" + "="*60)
        print("PREPARING ARTIFACTS FOR GITHUB ACTIONS")
        print("="*60)
        
        # Buat folder ci_artifacts (untuk GitHub Actions)
        ci_artifacts_dir = Path("ci_artifacts")
        ci_artifacts_dir.mkdir(exist_ok=True)
        
        # Copy file penting ke ci_artifacts
        important_files = [
            "artifacts/confusion_matrix.png",
            "artifacts/feature_importance.png",
            "artifacts/feature_importance.csv",
            "artifacts/training_summary.json",
            "artifacts/classification_report.json"
        ]
        
        print(f"\n📦 Copying artifacts for GitHub Actions...")
        files_copied = []
        for file_path in important_files:
            if os.path.exists(file_path):
                dest_path = ci_artifacts_dir / os.path.basename(file_path)
                shutil.copy2(file_path, dest_path)
                files_copied.append(os.path.basename(file_path))
                print(f"   ✅ Copied: {os.path.basename(file_path)}")
            else:
                print(f"   ⚠️  Missing: {file_path}")
        
        # Buat file success marker
        success_file = ci_artifacts_dir / "workflow_success.txt"
        with open(success_file, "w") as f:
            f.write("="*60 + "\n")
            f.write("GITHUB ACTIONS WORKFLOW - TRAINING SUCCESS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Workflow completed successfully at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"MLflow Run ID: {run.info.run_id}\n")
            f.write(f"Test F1-Score: {test_metrics['test_f1']:.4f}\n")
            f.write(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}\n")
            f.write(f"Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}\n")
            f.write(f"Artifacts generated: {len(files_copied)} files\n")
            f.write("\nFiles available for download:\n")
            for file in files_copied:
                f.write(f"  - {file}\n")
        
        print(f"✅ Created: workflow_success.txt")
        print(f"\n📁 GitHub Actions artifacts ready at: {ci_artifacts_dir.absolute()}")
        
        # ========== SUMMARY ==========
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\n📊 Final Test Metrics:")
        print(f"   Accuracy:  {test_metrics['test_accuracy']:.4f}")
        print(f"   Precision: {test_metrics['test_precision']:.4f}")
        print(f"   Recall:    {test_metrics['test_recall']:.4f}")
        print(f"   F1-Score:  {test_metrics['test_f1']:.4f}")
        print(f"   ROC-AUC:   {test_metrics['test_roc_auc']:.4f}")
        
        print(f"\n📁 MLflow Artifacts saved:")
        print(f"   - confusion_matrix.png")
        print(f"   - feature_importance.png + .csv")
        print(f"   - classification_report.json")
        print(f"   - training_summary.json")
        print(f"   - model/")
        
        print(f"\n📦 GitHub Actions Artifacts ready:")
        print(f"   Location: ci_artifacts/")
        print(f"   Files: {len(files_copied)} files for download")
        
        return test_metrics['test_f1']

if __name__ == "__main__":
    f1_score_result = main()
    print(f"\n🎯 Final F1-Score: {f1_score_result:.4f}")