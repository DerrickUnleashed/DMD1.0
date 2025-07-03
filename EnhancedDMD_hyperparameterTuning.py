import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from boruta import BorutaPy
import shap
import warnings

warnings.filterwarnings('ignore')

class EnhancedDMDGeneIdentifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.models = {}
        self.selected_features_mask = None
        self.final_feature_names = None

    def create_enhanced_features(self, df):
        features = []
        feature_names = []

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

        if 'Gene title' in df.columns:
            dmd_keywords = ['duchenne', 'dystrophin', 'muscle', 'myosin', 'actin', 'membrane']
            for keyword in dmd_keywords:
                feature_col = df['Gene title'].astype(str).str.lower().str.contains(keyword, na=False).astype(float)
                features.append(feature_col)
                feature_names.append(f'title_{keyword}')

        if 'Gene symbol' in df.columns:
            dmd_families = ['DMD', 'DCM', 'SGCA', 'SGCB', 'SGCD', 'SGCG', 'CAPN3', 'DYSF']
            for family in dmd_families:
                feature_col = df['Gene symbol'].astype(str).str.upper().str.contains(family, na=False).astype(float)
                features.append(feature_col)
                feature_names.append(f'symbol_{family}')

        if len(features) >= 3:
            muscle_idx = next((i for i, name in enumerate(feature_names) if 'muscle' in name), None)
            membrane_idx = next((i for i, name in enumerate(feature_names) if 'membrane' in name), None)
            if muscle_idx is not None and membrane_idx is not None:
                interaction_feature = features[muscle_idx] * features[membrane_idx]
                features.append(interaction_feature)
                feature_names.append('muscle_membrane_interaction')

        feature_df = pd.DataFrame(np.column_stack(features), columns=feature_names)
        self.feature_names = feature_names
        return feature_df

    def create_labels(self, df):
        known_dmd_genes = [
            'DMD', 'SGCA', 'SGCB', 'SGCD', 'SGCG', 'CAPN3', 'DYSF', 'TCAP',
            'POMT1', 'POMT2', 'POMGNT1', 'FKTN', 'FKRP', 'LARGE', 'DAG1'
        ]
        if 'Gene symbol' in df.columns:
            labels = df['Gene symbol'].astype(str).str.upper().isin(known_dmd_genes).astype(int)
        else:
            labels = np.zeros(len(df))
            if 'Gene title' in df.columns:
                dmd_keywords = ['duchenne', 'dystrophin']
                for keyword in dmd_keywords:
                    labels += df['Gene title'].astype(str).str.lower().str.contains(keyword, na=False).astype(int)
                labels = (labels > 0).astype(int)
        return labels

    def feature_selection(self, X, y):
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced')
        boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=self.random_state)
        boruta_selector.fit(X, y)
        boruta_mask = boruta_selector.support_

        lasso = LassoCV(cv=5, random_state=self.random_state).fit(X, y)
        lasso_mask = np.abs(lasso.coef_) > 1e-4

        combined_mask = np.logical_or(boruta_mask, lasso_mask)

        self.selected_features_mask = combined_mask
        self.final_feature_names = [name for i, name in enumerate(self.feature_names) if combined_mask[i]]

        return combined_mask

    def create_models(self):
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=self.random_state
        )
        # Add bagging wrapper to Random Forest to reduce variance
        rf_bagging = BaggingClassifier(
            estimator=rf,
            n_estimators=10,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            n_jobs=-1,
            random_state=self.random_state
        )

        self.models = {
            'RandomForestBagging': rf_bagging,
            'SVM': SVC(probability=True, kernel='rbf', class_weight='balanced', random_state=self.random_state),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=self.random_state
            ),
            'LogisticRegression': LogisticRegression(
                class_weight='balanced', max_iter=1000, random_state=self.random_state
            ),
            'NeuralNetwork': MLPClassifier(
                hidden_layer_sizes=(100,), max_iter=500, random_state=self.random_state
            )
        }

    def hyperparameter_tuning(self, X, y, cv=3,
                              rf_param_grid=None,
                              svm_param_grid=None,
                              gb_param_grid=None,
                              lr_param_grid=None,
                              nn_param_grid=None):

        tuned_models = {}

        rf = RandomForestClassifier(class_weight='balanced', random_state=self.random_state)
        if rf_param_grid is None:
            rf_param_grid = {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        rf_search = GridSearchCV(rf, rf_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        rf_search.fit(X, y)
        # Wrap best RF in BaggingClassifier
        rf_bagging = BaggingClassifier(
            base_estimator=rf_search.best_estimator_,
            n_estimators=10,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            n_jobs=-1,
            random_state=self.random_state
        )
        tuned_models['RandomForestBagging'] = rf_bagging

        svm = SVC(probability=True, class_weight='balanced', random_state=self.random_state)
        if svm_param_grid is None:
            svm_param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto']
            }
        svm_search = GridSearchCV(svm, svm_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        svm_search.fit(X, y)
        tuned_models['SVM'] = svm_search.best_estimator_

        gb = GradientBoostingClassifier(random_state=self.random_state)
        if gb_param_grid is None:
            gb_param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 6, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0]
            }
        gb_search = GridSearchCV(gb, gb_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        gb_search.fit(X, y)
        tuned_models['GradientBoosting'] = gb_search.best_estimator_

        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=self.random_state)
        if lr_param_grid is None:
            lr_param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'solver': ['saga', 'lbfgs']
            }
        lr_param_grid_filtered = []
        for c in lr_param_grid['C']:
            for penalty in lr_param_grid['penalty']:
                for solver in lr_param_grid['solver']:
                    if penalty == 'none' and solver != 'lbfgs':
                        continue
                    if penalty == 'elasticnet' and solver != 'saga':
                        continue
                    if penalty == 'l1' and solver not in ['saga', 'liblinear']:
                        continue
                    if penalty == 'l2' and solver not in ['lbfgs', 'saga', 'liblinear']:
                        continue
                    lr_param_grid_filtered.append({'C': c, 'penalty': penalty, 'solver': solver})
        lr_search = GridSearchCV(lr, lr_param_grid_filtered, cv=cv, scoring='roc_auc', n_jobs=-1)
        lr_search.fit(X, y)
        tuned_models['LogisticRegression'] = lr_search.best_estimator_

        nn = MLPClassifier(max_iter=500, random_state=self.random_state)
        if nn_param_grid is None:
            nn_param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 25)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        nn_search = GridSearchCV(nn, nn_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        nn_search.fit(X, y)
        tuned_models['NeuralNetwork'] = nn_search.best_estimator_

        self.models = tuned_models
        print("Hyperparameter tuning completed. Best models updated.")

    def cross_validate_models(self, X, y, cv=5):
        results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
            results[name] = scores
            print(f"{name} ROC-AUC: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
        return results

    def stacking_ensemble(self, X, y):
        estimators = [(name, model) for name, model in self.models.items() if name != 'LogisticRegression']
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=self.random_state),
            cv=5,
            n_jobs=-1,
            passthrough=True
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(stack, X, y, cv=skf, scoring='roc_auc')
        print(f"Stacking Ensemble ROC-AUC: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
        return stack, scores

    def train_final_model(self, X, y, model):
        model.fit(X, y)
        return model

    def interpret_model(self, model, X, feature_names):
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, features=X, feature_names=feature_names, show=True)

    def cross_dataset_validation(self, datasets):
        results = {}
        for i, train_ds in enumerate(datasets):
            try:
                train_df = pd.read_csv(f'Dataset_{train_ds}.csv')
                X_train = self.create_enhanced_features(train_df)
                y_train = self.create_labels(train_df)
                X_train_scaled = self.scaler.fit_transform(X_train)
                feature_mask = self.feature_selection(X_train_scaled, y_train)
                X_train_selected = X_train_scaled[:, feature_mask]

                self.create_models()
                stack_model, _ = self.stacking_ensemble(X_train_selected, y_train)
                final_model = self.train_final_model(X_train_selected, y_train, stack_model)

                results[train_ds] = {}
                for j, test_ds in enumerate(datasets):
                    if i == j:
                        continue
                    test_df = pd.read_csv(f'Dataset_{test_ds}.csv')
                    X_test = self.create_enhanced_features(test_df)
                    y_test = self.create_labels(test_df)
                    X_test_scaled = self.scaler.transform(X_test)
                    X_test_selected = X_test_scaled[:, feature_mask]

                    y_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_proba)
                    results[train_ds][test_ds] = auc
                    print(f"Train on {train_ds}, Test on {test_ds}: ROC-AUC = {auc:.3f}")
            except Exception as e:
                print(f"Error in cross-dataset validation for train {train_ds}: {e}")
        return results

    def permutation_test(self, model, X, y, n_permutations=100):
        baseline_score = roc_auc_score(y, model.predict_proba(X)[:, 1])
        permuted_scores = []
        for i in range(n_permutations):
            y_permuted = np.random.permutation(y)
            model.fit(X, y_permuted)
            score = roc_auc_score(y_permuted, model.predict_proba(X)[:, 1])
            permuted_scores.append(score)
        p_value = np.mean([score >= baseline_score for score in permuted_scores])
        print(f"Permutation test p-value: {p_value:.4f}")
        return p_value

    def final_candidate_selection(self, results, top_n=20):
        gene_scores = {}
        for dataset_name, res in results.items():
            if 'feature_importance' in res:
                imp_df = res['feature_importance']
                for _, row in imp_df.iterrows():
                    gene = row['feature']
                    score = row['importance']
                    gene_scores[gene] = gene_scores.get(gene, []) + [score]

        avg_scores = {gene: np.mean(scores) for gene, scores in gene_scores.items()}
        sorted_genes = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        top_genes = [gene for gene, score in sorted_genes[:top_n]]

        print(f"Top {top_n} candidate genes:")
        for gene in top_genes:
            print(f"  {gene} (avg importance: {avg_scores[gene]:.4f})")

        return top_genes

    def run_full_pipeline(self, datasets):
        all_results = {}

        for dataset_name in datasets:
            print(f"\n{'='*50}")
            print(f"Processing Dataset: {dataset_name}")
            print(f"{'='*50}")

            try:
                df = pd.read_csv(f'Dataset_{dataset_name}.csv')
                X = self.create_enhanced_features(df)
                y = self.create_labels(df)

                if y.sum() == 0:
                    print("Warning: No positive samples found!")
                    continue

                X_scaled = self.scaler.fit_transform(X)
                feature_mask = self.feature_selection(X_scaled, y)
                X_selected = X_scaled[:, feature_mask]

                self.create_models()
                cv_results = self.cross_validate_models(X_selected, y)

                stack_model, stack_scores = self.stacking_ensemble(X_selected, y)
                final_model = self.train_final_model(X_selected, y, stack_model)

                rf_model = None
                for est_name, est in stack_model.named_estimators_.items():
                    if est_name == 'RandomForest':
                        rf_model = est
                        break

                if rf_model is not None:
                    feature_importance = rf_model.feature_importances_
                    importance_df = pd.DataFrame({
                        'feature': self.final_feature_names,
                        'importance': feature_importance
                    }).sort_values('importance', ascending=False)
                else:
                    importance_df = pd.DataFrame(columns=['feature', 'importance'])

                all_results[dataset_name] = {
                    'cv_scores': cv_results,
                    'stacking_scores': stack_scores,
                    'final_model': final_model,
                    'feature_importance': importance_df,
                    'feature_mask': feature_mask,
                    'genes': df['Gene symbol'].tolist() if 'Gene symbol' in df.columns else []
                }

                print("\nTop 10 Most Important Features:")
                print(importance_df.head(10))

                print("Generating SHAP summary plot...")
                self.interpret_model(final_model, X_selected, self.final_feature_names)

            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")

        print("\nCross-dataset validation:")
        cross_val_results = self.cross_dataset_validation(datasets)

        top_candidates = self.final_candidate_selection(all_results)

        return all_results, cross_val_results, top_candidates


if __name__ == "__main__":
    datasets = ["GSE38417", "GSE6011", "GSE19303", "GSE42955"]
    identifier = EnhancedDMDGeneIdentifier()
    all_results, cross_val_results, top_candidates = identifier.run_full_pipeline(datasets)
