import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import joblib
import json
import os
import argparse
from datetime import datetime
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

class PhishingModelEvaluator:
    def __init__(self, model_dir, test_data_path, output_dir="evaluation_results"):
        """
        Initialize the evaluator with model directory and test data path
        """
        self.model_dir = model_dir
        self.test_data_path = test_data_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"evaluation_{self.timestamp}")
        self.models = {}
        self.scaler = None
        self.training_results = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Evaluation results will be saved to: {self.output_dir}")
        
        # Load test data
        self.load_test_data()
        
        # Load models and training results
        self.load_models()
        
    def load_test_data(self):
        """Load test dataset from numpy files"""
        print("Loading test dataset...")
        
        # Construct test data paths
        test_dir = Path(self.test_data_path)
        x_test_path = test_dir / "X_test.npy"
        y_test_path = test_dir / "y_test.npy"
        
        # Load numpy arrays
        self.X_test = np.load(x_test_path)
        self.y_test = np.load(y_test_path)
        
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Test set class distribution: {np.bincount(self.y_test.astype(int))}")
        
    def load_models(self):
        """Load trained models and scaler from model directory"""
        print("Loading trained models...")
        
        model_dir = Path(self.model_dir)
        
        # Load training results
        results_path = model_dir / "training_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                self.training_results = json.load(f)
            print(f"Loaded training results from {results_path}")
        
        # Load scaler
        scaler_path = model_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            print("Loaded scaler and transformed test data")
        
        # Load individual models
        model_files = {
            'Decision Tree': 'decision_tree.pkl',
            'Random Forest': 'random_forest.pkl',
            'Logistic Regression': 'logistic_regression.pkl',
            'Naive Bayes': 'naive_bayes.pkl',
            'XGBoost': 'xgboost.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = model_dir / filename
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name}")
        
        print(f"Successfully loaded {len(self.models)} models")
        
    def get_test_data(self, model_name):
        """Get appropriate test data (scaled or unscaled) for a model"""
        if model_name in self.training_results.get('models', {}):
            use_scaled = self.training_results['models'][model_name].get('use_scaled', False)
            return self.X_test_scaled if use_scaled else self.X_test
        else:
            # Default logic for models that typically need scaling
            scaled_models = ['Logistic Regression', 'Naive Bayes']
            return self.X_test_scaled if model_name in scaled_models else self.X_test
            
    def evaluate_single_model(self, model_name, model):
        """Evaluate a single model and return metrics"""
        print(f"Evaluating {model_name}...")
        
        # Get appropriate test data
        X_test = self.get_test_data(model_name)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'avg_precision': average_precision_score(self.y_test, y_pred_proba)
        }
        
        # Get ROC curve data
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        
        # Get Precision-Recall curve data
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Classification report
        class_report = classification_report(self.y_test, y_pred, output_dict=True)
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision, recall)
        }
        
    def evaluate_all_models(self):
        """Evaluate all loaded models"""
        print("Evaluating all models...")
        
        self.evaluation_results = {}
        
        for model_name, model in self.models.items():
            self.evaluation_results[model_name] = self.evaluate_single_model(model_name, model)
            
        print("Model evaluation completed!")
        
    def create_visualizations(self):
        """Create visualization plots"""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. ROC Curves Comparison
        plt.figure(figsize=(12, 8))
        
        for model_name, results in self.evaluation_results.items():
            fpr, tpr = results['roc_curve']
            auc = results['metrics']['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        roc_path = os.path.join(self.output_dir, "roc_curves_comparison.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Precision-Recall Curves
        plt.figure(figsize=(12, 8))
        
        for model_name, results in self.evaluation_results.items():
            precision, recall = results['pr_curve']
            avg_precision = results['metrics']['avg_precision']
            plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Model Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        pr_path = os.path.join(self.output_dir, "precision_recall_curves.png")
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion Matrices
        n_models = len(self.evaluation_results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        cm_path = os.path.join(self.output_dir, "confusion_matrices.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Performance Metrics Comparison
        metrics_df = pd.DataFrame({
            model_name: results['metrics'] 
            for model_name, results in self.evaluation_results.items()
        }).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot each metric
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx//2, idx%2]
            bars = ax.bar(metrics_df.index, metrics_df[metric])
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        metrics_path = os.path.join(self.output_dir, "metrics_comparison.png")
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved successfully!")
        
    def create_performance_table(self):
        """Create a comprehensive performance comparison table"""
        print("Creating performance comparison table...")
        
        # Extract metrics for all models
        performance_data = []
        
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            
            # Add training metrics if available
            training_metrics = {}
            if model_name in self.training_results.get('models', {}):
                training_info = self.training_results['models'][model_name]
                training_metrics = {
                    'CV_Score_Mean': training_info.get('cv_score_mean', 'N/A'),
                    'CV_Score_Std': training_info.get('cv_score_std', 'N/A'),
                    'Val_Score': training_info.get('val_score', 'N/A')
                }
            
            row = {
                'Model': model_name,
                'Test_Accuracy': metrics['accuracy'],
                'Test_Precision': metrics['precision'],
                'Test_Recall': metrics['recall'],
                'Test_F1': metrics['f1'],
                'Test_ROC_AUC': metrics['roc_auc'],
                'Test_Avg_Precision': metrics['avg_precision'],
                **training_metrics
            }
            performance_data.append(row)
        
        # Create DataFrame
        performance_df = pd.DataFrame(performance_data)
        
        # Sort by Test_ROC_AUC (descending)
        performance_df = performance_df.sort_values('Test_ROC_AUC', ascending=False)
        
        # Save to CSV
        table_path = os.path.join(self.output_dir, "performance_comparison.csv")
        performance_df.to_csv(table_path, index=False)
        
        # Save formatted table
        formatted_path = os.path.join(self.output_dir, "performance_summary.txt")
        with open(formatted_path, 'w') as f:
            f.write("PHISHING DETECTION MODEL EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write("Performance Comparison (Sorted by ROC-AUC)\n")
            f.write("-" * 50 + "\n\n")
            
            for _, row in performance_df.iterrows():
                f.write(f"Model: {row['Model']}\n")
                f.write(f"  Test Accuracy:     {row['Test_Accuracy']:.4f}\n")
                f.write(f"  Test Precision:    {row['Test_Precision']:.4f}\n")
                f.write(f"  Test Recall:       {row['Test_Recall']:.4f}\n")
                f.write(f"  Test F1-Score:     {row['Test_F1']:.4f}\n")
                f.write(f"  Test ROC-AUC:      {row['Test_ROC_AUC']:.4f}\n")
                f.write(f"  Test Avg Precision: {row['Test_Avg_Precision']:.4f}\n")
                if 'Val_Score' in row and row['Val_Score'] != 'N/A':
                    f.write(f"  Validation AUC:    {row['Val_Score']:.4f}\n")
                f.write("\n")
        
        print(f"Performance table saved to {table_path}")
        print(f"Performance summary saved to {formatted_path}")
        
        return performance_df
        
    def save_detailed_results(self):
        """Save detailed evaluation results to JSON"""
        print("Saving detailed results...")
        
        # Prepare results for JSON serialization
        json_results = {
            'evaluation_timestamp': self.timestamp,
            'test_data_info': {
                'test_samples': int(len(self.y_test)),
                'positive_samples': int(np.sum(self.y_test)),
                'negative_samples': int(len(self.y_test) - np.sum(self.y_test))
            },
            'models': {}
        }
        
        for model_name, results in self.evaluation_results.items():
            json_results['models'][model_name] = {
                'metrics': results['metrics'],
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'classification_report': results['classification_report']
            }
        
        # Save to JSON
        json_path = os.path.join(self.output_dir, "detailed_results.json")
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Detailed results saved to {json_path}")
        
    def run_evaluation(self):
        """Run the complete evaluation process"""
        print("Starting model evaluation process...")
        
        # Evaluate all models
        self.evaluate_all_models()
        
        # Create visualizations
        self.create_visualizations()
        
        # Create performance table
        performance_df = self.create_performance_table()
        
        # Save detailed results
        self.save_detailed_results()
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved in: {self.output_dir}")
        print("\nBest performing models (by ROC-AUC):")
        print("-" * 40)
        
        top_models = performance_df.head(3)
        for _, row in top_models.iterrows():
            print(f"{row['Model']}: {row['Test_ROC_AUC']:.4f}")
            
        print(f"\nGenerated files:")
        print(f"- performance_comparison.csv")
        print(f"- performance_summary.txt")
        print(f"- detailed_results.json")
        print(f"- roc_curves_comparison.png")
        print(f"- precision_recall_curves.png")
        print(f"- confusion_matrices.png")
        print(f"- metrics_comparison.png")

def find_latest_model_dir(models_base_path):
    """Find the latest training directory"""
    models_path = Path(models_base_path)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_base_path}")
    
    # Find all training directories
    training_dirs = [d for d in models_path.iterdir() if d.is_dir() and d.name.startswith('training_')]
    
    if not training_dirs:
        raise FileNotFoundError("No training directories found")
    
    # Return the most recent one
    latest_dir = max(training_dirs, key=lambda x: x.name)
    return latest_dir

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate phishing detection models')
    parser.add_argument('--model-dir', type=str, 
                       help='Path to specific model directory (if not specified, uses latest)')
    parser.add_argument('--test-data', type=str, 
                       default=r"C:\Users\jasud\OneDrive - Sunway Education Group\school\sem 8\CP2\Phishing detection\data_splits",
                       help='Path to test data directory')
    parser.add_argument('--models-base', type=str,
                       default=r"C:\Users\jasud\OneDrive - Sunway Education Group\school\sem 8\CP2\Phishing detection\models",
                       help='Base path for models directory')
    
    args = parser.parse_args()
    
    # Determine model directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = find_latest_model_dir(args.models_base)
        print(f"Using latest model directory: {model_dir}")
    
    # Initialize and run evaluator
    evaluator = PhishingModelEvaluator(
        model_dir=model_dir,
        test_data_path=args.test_data
    )
    
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()