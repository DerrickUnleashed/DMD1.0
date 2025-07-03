import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, jaccard_score, precision_recall_curve, matthews_corrcoef, cohen_kappa_score, log_loss, auc
from boruta import BorutaPy
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

class DMDGeneIdentifierSingleDataset:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.selected_features_mask = None
        self.feature_names = None
        self.final_feature_names = None

    def create_enhanced_features(self, df):
        features = []
        feature_names = []

        dmd_go_weights = {
            'muscle': 3.0, 'dystrophin': 5.0, 'cytoskeleton': 2.0,
            'membrane': 2.0, 'calcium': 1.5, 'contraction': 2.5
        }

        for col in ['GO:Function', 'GO:Process', 'GO:Component']:
            if col in df.columns:
                for term, weight in dmd_go_weights.items():
                    match = df[col].astype(str).str.lower().str.contains(term, na=False).astype(float)
                    features.append(match * weight)
                    feature_names.append(f"{col}_{term}_weighted")

        if 'Gene title' in df.columns:
            for keyword in ['duchenne', 'dystrophin', 'muscle', 'myosin', 'actin', 'membrane']:
                match = df['Gene title'].astype(str).str.lower().str.contains(keyword, na=False).astype(float)
                features.append(match)
                feature_names.append(f"title_{keyword}")

        if 'Gene symbol' in df.columns:
            for family in ['DMD', 'DCM', 'SGCA', 'SGCB', 'SGCD', 'SGCG', 'CAPN3', 'DYSF']:
                match = df['Gene symbol'].astype(str).str.upper().str.contains(family, na=False).astype(float)
                features.append(match)
                feature_names.append(f"symbol_{family}")

        # Interaction
        if len(features) >= 3:
            try:
                i1 = next(i for i, name in enumerate(feature_names) if 'muscle' in name)
                i2 = next(i for i, name in enumerate(feature_names) if 'membrane' in name)
                interaction = features[i1] * features[i2]
                features.append(interaction)
                feature_names.append('muscle_membrane_interaction')
            except StopIteration:
                pass

        self.feature_names = feature_names
        return pd.DataFrame(np.column_stack(features), columns=feature_names)

    def create_labels(self, df):
        known_genes = [
            'DMD', 'SGCA', 'SGCB', 'SGCD', 'SGCG', 'CAPN3',
            'DYSF', 'TCAP', 'POMT1', 'POMT2', 'POMGNT1',
            'FKTN', 'FKRP', 'LARGE', 'DAG1'
        ]
        if 'Gene symbol' in df.columns:
            return df['Gene symbol'].astype(str).str.upper().isin(known_genes).astype(int)
        return pd.Series(np.zeros(len(df)), dtype=int)

    def feature_selection(self, X, y):
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced')
        boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=self.random_state)
        boruta_selector.fit(X, y)
        boruta_mask = boruta_selector.support_

        lasso = LassoCV(cv=5, random_state=self.random_state).fit(X, y)
        lasso_mask = np.abs(lasso.coef_) > 1e-4

        combined_mask = np.logical_or(boruta_mask, lasso_mask)
        self.selected_features_mask = combined_mask
        self.final_feature_names = [self.feature_names[i] for i, keep in enumerate(combined_mask) if keep]
        return combined_mask

    def create_models(self):
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                                                   class_weight='balanced', random_state=self.random_state),
            'SVM': SVC(probability=True, kernel='rbf', class_weight='balanced', random_state=self.random_state),
            'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=self.random_state)
        }

    def cross_validate_models(self, X, y, cv=5):
        results = {}

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Cross-validation setup
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for name, model in self.models.items():
            print(f"\n==== {name} ====")

            # Cross-validation ROC-AUC scores
            scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
            results[name] = scores
            print(f"Cross-validated ROC-AUC: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

            # Fit on training set
            model.fit(X_train_scaled, y_train)

            # Predictions on test set
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

            # Evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=["NORMAL", "DMD"])
            cm = confusion_matrix(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred)
            jaccard = jaccard_score(y_test, y_pred)
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)
            mcc = matthews_corrcoef(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            loss = log_loss(y_test, y_prob)

            # Output metrics
            print("Accuracy:", acc)
            print("Classification Report:\n", report)
            print("Confusion Matrix:\n", cm)
            print("AUC-ROC Score:", auc_score)
            print("Jaccard Score:", jaccard)
            print("Precision-Recall AUC:", pr_auc)
            print("Matthews Correlation Coefficient:", mcc)
            print("Cohen's Kappa:", kappa)
            print("Log Loss:", loss)

        return results


    def stacking_ensemble(self, X, y):
        estimators = [(name, model) for name, model in self.models.items()]
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=self.random_state),
            cv=5,
            n_jobs=-1,
            passthrough=True
        )
        scores = cross_val_score(stack, X, y, cv=5, scoring='roc_auc')
        print(f"Stacking Ensemble ROC-AUC: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
        return stack, scores

    def interpret_model(self, model, X, feature_names, dataset_name):
        rf_model = model.named_estimators_['RandomForest']
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, features=X, feature_names=feature_names, show=False)
        plt.savefig(f'shap_summary_plot_{dataset_name}.png', bbox_inches='tight', dpi=300)
        plt.close()

    def run_pipeline_on_single_dataset(self, dataset_name):
        print(f"\n{'='*50}")
        print(f"Processing Dataset: {dataset_name}")
        print(f"{'='*50}")

        try:
            df = pd.read_csv(f'Dataset_{dataset_name}.csv')
            print(f"Dataset shape: {df.shape}")

            X = self.create_enhanced_features(df)
            y = self.create_labels(df)

            print(f"Features created: {X.shape[1]}")
            print(f"Positive samples: {y.sum()}/{len(y)} ({y.mean()*100:.2f}%)")

            if y.sum() == 0:
                print("Warning: No positive samples found!")
                return None

            X_scaled = self.scaler.fit_transform(X)
            feature_mask = self.feature_selection(X_scaled, y)
            X_selected = X_scaled[:, feature_mask]

            self.create_models()
            cv_scores = self.cross_validate_models(X_selected, y)

            stack_model, stack_scores = self.stacking_ensemble(X_selected, y)
            stack_model.fit(X_selected, y)

            rf_model = stack_model.named_estimators_.get('RandomForest')
            if rf_model:
                importance_df = pd.DataFrame({
                    'feature': self.final_feature_names,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                print("\nTop 10 Most Important Features:")
                print(importance_df.head(10))
            else:
                importance_df = pd.DataFrame()

            print("Generating SHAP summary plot...")
            self.interpret_model(stack_model, X_selected, self.final_feature_names, dataset_name)

            return {
                'cv_scores': cv_scores,
                'stacking_scores': stack_scores,
                'final_model': stack_model,
                'feature_importance': importance_df,
                'feature_mask': feature_mask,
                'genes': df['Gene symbol'].tolist() if 'Gene symbol' in df.columns else []
            }

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            return None


if __name__ == "__main__":
    dataset = "Combined"
    identifier = DMDGeneIdentifierSingleDataset()
    results = identifier.run_pipeline_on_single_dataset(dataset)
