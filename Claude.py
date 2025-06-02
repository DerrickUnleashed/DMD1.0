import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class DMDGeneIdentifier:
    def __init__(self):
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.ensemble_model = None
        self.feature_names = None
        
    def create_enhanced_features(self, df):
        """Create enhanced features for DMD identification"""
        features = []
        feature_names = []
        
        # 1. GO Term Features (weighted by relevance)
        dmd_go_weights = {
            'muscle': 3.0,
            'dystrophin': 5.0,
            'cytoskeleton': 2.0,
            'membrane': 2.0,
            'calcium': 1.5,
            'contraction': 2.5
        }
        
        for col in ['GO:Function', 'GO:Process', 'GO:Component']:
            if col in df.columns:
                for term, weight in dmd_go_weights.items():
                    feature_col = df[col].astype(str).str.lower().str.contains(term, na=False).astype(float) * weight
                    features.append(feature_col)
                    feature_names.append(f'{col}_{term}_weighted')
        
        # 2. Gene Title Features
        if 'Gene title' in df.columns:
            dmd_keywords = ['duchenne', 'dystrophin', 'muscle', 'myosin', 'actin', 'membrane']
            for keyword in dmd_keywords:
                feature_col = df['Gene title'].astype(str).str.lower().str.contains(keyword, na=False).astype(float)
                features.append(feature_col)
                feature_names.append(f'title_{keyword}')
        
        # 3. Gene Symbol Features
        if 'Gene symbol' in df.columns:
            # Known DMD-related gene families
            dmd_families = ['DMD', 'DCM', 'SGCA', 'SGCB', 'SGCD', 'SGCG', 'CAPN3', 'DYSF']
            for family in dmd_families:
                feature_col = df['Gene symbol'].astype(str).str.upper().str.contains(family, na=False).astype(float)
                features.append(feature_col)
                feature_names.append(f'symbol_{family}')
        
        # 4. Interaction Features (combinations of GO terms)
        if len(features) >= 3:
            # Muscle + membrane interaction
            muscle_idx = next((i for i, name in enumerate(feature_names) if 'muscle' in name), None)
            membrane_idx = next((i for i, name in enumerate(feature_names) if 'membrane' in name), None)
            
            if muscle_idx is not None and membrane_idx is not None:
                interaction_feature = features[muscle_idx] * features[membrane_idx]
                features.append(interaction_feature)
                feature_names.append('muscle_membrane_interaction')
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(np.column_stack(features), columns=feature_names)
        self.feature_names = feature_names
        
        return feature_df
    
    def create_labels(self, df):
        """Create labels based on known DMD genes"""
        known_dmd_genes = [
            'DMD', 'SGCA', 'SGCB', 'SGCD', 'SGCG', 'CAPN3', 'DYSF', 'TCAP', 
            'POMT1', 'POMT2', 'POMGNT1', 'FKTN', 'FKRP', 'LARGE', 'DAG1'
        ]
        
        if 'Gene symbol' in df.columns:
            labels = df['Gene symbol'].astype(str).str.upper().isin(known_dmd_genes).astype(int)
        else:
            # Fallback to keyword-based labeling
            labels = np.zeros(len(df))
            if 'Gene title' in df.columns:
                dmd_keywords = ['duchenne', 'dystrophin']
                for keyword in dmd_keywords:
                    labels += df['Gene title'].astype(str).str.lower().str.contains(keyword, na=False).astype(int)
                labels = (labels > 0).astype(int)
        
        return labels
    
    def advanced_feature_selection(self, X, y):
        """Advanced feature selection combining multiple methods"""
        # 1. Statistical filter (ANOVA F-test)
        k_best = SelectKBest(score_func=f_classif, k=min(50, X.shape[1]))
        X_filtered = k_best.fit_transform(X, y)
        selected_features_mask = k_best.get_support()
        
        # 2. Recursive Feature Elimination with Random Forest
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFECV(rf_selector, step=1, cv=3, scoring='roc_auc', min_features_to_select=5)
        
        # Apply RFE on filtered features
        X_rfe = rfe.fit_transform(X_filtered, y)
        
        # Combine selection masks
        final_mask = np.zeros(X.shape[1], dtype=bool)
        selected_indices = np.where(selected_features_mask)[0]
        rfe_selected = selected_indices[rfe.get_support()]
        final_mask[rfe_selected] = True
        
        return final_mask, X_rfe
    
    def create_ensemble_model(self):
        """Create ensemble of different models"""
        # Individual models
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=10,  # For imbalanced data
            random_state=42
        )
        
        lr = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # Voting ensemble
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb), ('lr', lr)],
            voting='soft'  # Use probabilities
        )
        
        return ensemble
    
    def train_and_evaluate(self, datasets):
        """Train and evaluate on multiple datasets"""
        all_results = {}
        
        for dataset_name in datasets:
            print(f"\n{'='*50}")
            print(f"Processing Dataset: {dataset_name}")
            print(f"{'='*50}")
            
            # Load data
            try:
                df = pd.read_csv(f'Dataset_{dataset_name}.csv')
                print(f"Dataset shape: {df.shape}")
                
                # Create features and labels
                X = self.create_enhanced_features(df)
                y = self.create_labels(df)
                
                print(f"Features created: {X.shape[1]}")
                print(f"Positive samples: {y.sum()}/{len(y)} ({y.mean()*100:.2f}%)")
                
                if y.sum() == 0:
                    print("Warning: No positive samples found!")
                    continue
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Feature selection
                feature_mask, X_selected = self.advanced_feature_selection(X_scaled, y)
                selected_feature_names = [name for i, name in enumerate(self.feature_names) if feature_mask[i]]
                
                print(f"Selected features: {len(selected_feature_names)}")
                print("Top selected features:", selected_feature_names[:10])
                
                # Create and train ensemble model
                ensemble = self.create_ensemble_model()
                
                # Cross-validation evaluation
                cv_scores = cross_val_score(
                    ensemble, X_selected, y, 
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='roc_auc'
                )
                
                print(f"Cross-validation ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
                # Train final model
                ensemble.fit(X_selected, y)
                
                # Get feature importance from Random Forest component
                rf_model = ensemble.named_estimators_['rf']
                feature_importance = rf_model.feature_importances_
                
                # Create feature importance ranking
                importance_df = pd.DataFrame({
                    'feature': selected_feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                print("\nTop 10 Most Important Features:")
                print(importance_df.head(10))
                
                # Predict on all samples
                y_pred_proba = ensemble.predict_proba(X_selected)[:, 1]
                
                # Find high-confidence DMD-related genes
                high_conf_threshold = 0.7
                high_conf_indices = np.where(y_pred_proba > high_conf_threshold)[0]
                
                if len(high_conf_indices) > 0:
                    high_conf_genes = df.iloc[high_conf_indices]['Gene symbol'].tolist()
                    high_conf_scores = y_pred_proba[high_conf_indices]
                    
                    print(f"\nHigh-confidence DMD-related genes (score > {high_conf_threshold}):")
                    for gene, score in zip(high_conf_genes, high_conf_scores):
                        print(f"  {gene}: {score:.3f}")
                else:
                    print(f"\nNo genes found with confidence > {high_conf_threshold}")
                
                # Store results
                all_results[dataset_name] = {
                    'cv_scores': cv_scores,
                    'feature_importance': importance_df,
                    'predictions': y_pred_proba,
                    'genes': df['Gene symbol'].tolist() if 'Gene symbol' in df.columns else []
                }
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {str(e)}")
                continue
        
        return all_results

# Usage example
if __name__ == "__main__":
    # Initialize the identifier
    dmd_identifier = DMDGeneIdentifier()
    
    # List of datasets to process
    datasets = ["GSE38417", "GSE6011", "GSE19303", "GSE42955"]
    
    # Train and evaluate
    results = dmd_identifier.train_and_evaluate(datasets)
    
    # Summary across all datasets
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL DATASETS")
    print(f"{'='*60}")
    
    for dataset_name, result in results.items():
        if 'cv_scores' in result:
            mean_score = result['cv_scores'].mean()
            print(f"{dataset_name}: ROC-AUC = {mean_score:.3f}")
