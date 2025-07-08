
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np
import shap
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('synthetic_sourcing_data.csv')


# Feature engineering: cost margin between local and import for each component
df['cost_margin'] = 0.0
for cid in df['component_id'].unique():
    subset = df[df['component_id'] == cid]
    if len(subset) == 2:
        local_cost = subset[subset['source_type'] == 'Local']['total_landed_cost_inr'].values[0]
        import_cost = subset[subset['source_type'] == 'Import']['total_landed_cost_inr'].values[0]
        df.loc[(df['component_id'] == cid) & (df['source_type'] == 'Local'), 'cost_margin'] = local_cost - import_cost
        df.loc[(df['component_id'] == cid) & (df['source_type'] == 'Import'), 'cost_margin'] = import_cost - local_cost
    else:
        df.loc[df['component_id'] == cid, 'cost_margin'] = 0.0

# Prepare features
X = df.drop([
    'component_id', 'component_name', 'source_country',
    'total_landed_cost_inr', 'recommended_source'
], axis=1)
y = df['recommended_source']

# Encode categorical features
for col in ['metal_type', 'form_factor', 'industry_usage', 'source_type']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Address class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Hyperparameter tuning with class_weight='balanced'
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train_res, y_train_res)

print("Best parameters:", grid_search.best_params_)
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# Results
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Import', 'Local']))

# Precision, recall, f1 for Import class
prf = precision_recall_fscore_support(y_test, y_pred, labels=['Import'], average=None)
print(f"Import class - Precision: {prf[0][0]:.3f}, Recall: {prf[1][0]:.3f}, F1: {prf[2][0]:.3f}")

# SHAP values for feature importance
explainer = shap.TreeExplainer(best_clf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)

# Save the best model to disk
import joblib
joblib.dump(best_clf, 'sourcing_recommendation_model.pkl')
print('Model saved as sourcing_recommendation_model.pkl')
