import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class PhishingModelTrainer:
    def __init__(self, train_path, val_path, output_dir="models"):
        """
        Initialize the trainer with dataset paths
        """
        self.train_path = train_path
        self.val_path = val_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"training_{self.timestamp}")
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.scaler = StandardScaler()
        
        # Create timestamped output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Models will be saved to: {self.output_dir}")
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load training and validation datasets from numpy files"""
        print("Loading datasets...")
        
        # Load numpy arrays
        self.X_train = np.load(self.train_path.replace('X_train.npy', 'X_train.npy'))
        self.y_train = np.load(self.train_path.replace('X_train.npy', 'y_train.npy'))
        self.X_val = np.load(self.val_path.replace('X_val.npy', 'X_val.npy'))
        self.y_val = np.load(self.val_path.replace('X_val.npy', 'y_val.npy'))
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_val.shape}")
        print(f"Class distribution in training: {np.bincount(self.y_train.astype(int))}")
        
    def preprocess_data(self):
        """Preprocess the data - scaling for models that need it"""
        print("Preprocessing data...")
        
        # Fit scaler on training data and transform both sets
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        
        # Calculate class weights for imbalanced dataset
        self.class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y_train), 
            y=self.y_train
        )
        self.class_weight_dict = dict(zip(np.unique(self.y_train), self.class_weights))
        
        print(f"Class weights: {self.class_weight_dict}")
        
    def define_models(self):
        """Define models and their hyperparameter grids"""
        print("Defining models and hyperparameter grids...")
        
        # Decision Tree
        self.models['Decision Tree'] = {
            'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },
            'use_scaled': False
        }
        
        # Random Forest
        self.models['Random Forest'] = {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'use_scaled': False
        }
        
       # Logistic Regression (final fix)
        self.models['Logistic Regression'] = {
            'model': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000,
                solver='lbfgs'
            ),
            'params': {
                'C': [0.1, 1],  # Narrow grid
                'penalty': ['l2']
            },
            'use_scaled': True
        }


        
        # Naive Bayes
        self.models['Naive Bayes'] = {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            },
            'use_scaled': True
        }
        
        # XGBoost
        self.models['XGBoost'] = {
            'model': XGBClassifier(
                random_state=42, 
                eval_metric='logloss',
                scale_pos_weight=self.class_weights[1]/self.class_weights[0]
            ),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'use_scaled': False
        }
        
    def train_model(self, model_name):
        """Train a single model with hyperparameter tuning"""
        print(f"\nTraining {model_name}...")
        
        model_info = self.models[model_name]
        model = model_info['model']
        param_grid = model_info['params']
        use_scaled = model_info['use_scaled']
        
        # Choose appropriate data
        X_train = self.X_train_scaled if use_scaled else self.X_train
        X_val = self.X_val_scaled if use_scaled else self.X_val
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, self.y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on validation set
        val_pred = best_model.predict(X_val)
        val_pred_proba = best_model.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(self.y_val, val_pred_proba)
        
        # Cross-validation score
        cv_score_mean = grid_search.best_score_
        cv_score_std = np.std(grid_search.cv_results_['mean_test_score'])

        
        # Store results
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'cv_score_mean': cv_score_mean,
            'cv_score_std': cv_score_std,
            'val_score': val_score,
            'use_scaled': use_scaled
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Cross-validation score: {cv_score_mean:.4f} (+/- {cv_score_std * 2:.4f})")
        print(f"Validation AUC: {val_score:.4f}")
        
        return results
        
    def train_all_models(self):
        """Train all models and find the best one"""
        print("Starting model training...")
        
        # Preprocess data
        self.preprocess_data()
        
        # Define models
        self.define_models()
        
        # Train each model
        results = {}
        for model_name in self.models.keys():
            results[model_name] = self.train_model(model_name)
            
            # Check if this is the best model so far
            if results[model_name]['val_score'] > self.best_score:
                self.best_score = results[model_name]['val_score']
                self.best_model = model_name
                
        # Store all results
        self.results = results
        
        print(f"\nBest model: {self.best_model} with validation AUC: {self.best_score:.4f}")
        
    def save_models(self):
        """Save trained models and results"""
        print("Saving models and results...")
        
        # Save each model
        for model_name, result in self.results.items():
            model_filename = f"{model_name.replace(' ', '_').lower()}.pkl"
            model_path = os.path.join(self.output_dir, model_filename)
            joblib.dump(result['model'], model_path)
            print(f"Saved {model_name} to {model_path}")
            
        # Save best model separately
        best_model_path = os.path.join(self.output_dir, "best_model.pkl")
        joblib.dump(self.results[self.best_model]['model'], best_model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.output_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # Save results summary
        results_summary = {
            'best_model': self.best_model,
            'best_score': self.best_score,
            'timestamp': self.timestamp,
            'models': {}
        }
        
        for model_name, result in self.results.items():
            results_summary['models'][model_name] = {
                'best_params': result['best_params'],
                'cv_score_mean': result['cv_score_mean'],
                'cv_score_std': result['cv_score_std'],
                'val_score': result['val_score'],
                'use_scaled': result['use_scaled']
            }
            
        # Save results to JSON
        results_path = os.path.join(self.output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
            
        print(f"Training results saved to {results_path}")
        print(f"Best model saved to {best_model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"All files saved in directory: {self.output_dir}")
        
    def get_feature_importance(self):
        """Get feature importance for tree-based models"""
        print("Extracting feature importance...")
        
        for model_name, result in self.results.items():
            model = result['model']
            
            if hasattr(model, 'feature_importances_'):
                # Create feature names (since we don't have column names from numpy arrays)
                feature_names = [f'feature_{i}' for i in range(self.X_train.shape[1])]
                importances = model.feature_importances_
                
                # Create feature importance dataframe
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Save feature importance
                importance_path = os.path.join(
                    self.output_dir, 
                    f"feature_importance_{model_name.replace(' ', '_').lower()}.csv"
                )
                feature_importance_df.to_csv(importance_path, index=False)
                
                print(f"Feature importance for {model_name} saved to {importance_path}")
                print(f"Top 10 features for {model_name}:")
                print(feature_importance_df.head(10))
                print()

def main():
    """Main training function"""
    # Update these paths according to your file structure
    base_path = r"C:\Users\jasud\OneDrive - Sunway Education Group\school\sem 8\CP2\Phishing detection\data_splits"
    train_path = os.path.join(base_path, "X_train.npy")
    val_path = os.path.join(base_path, "X_val.npy")
    
    # Initialize trainer
    trainer = PhishingModelTrainer(train_path, val_path)
    
    # Train all models
    trainer.train_all_models()
    
    # Save models and results
    trainer.save_models()
    
    # Get feature importance
    trainer.get_feature_importance()
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()