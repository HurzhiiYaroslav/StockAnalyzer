import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def analyze_feature_importance(model, X_test_scaled, y_test_scalar, feature_names, model_type="mord"):
    print("\n--- Feature Importance Analysis ---")
    
    if model_type == "mord" and hasattr(model, 'coef_'):
        try:
            feature_coefficients = model.coef_
            importance_df = None
            if feature_coefficients.ndim == 2:
                avg_abs_coeffs = np.mean(np.abs(feature_coefficients), axis=0)
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance_abs_coef_mean': avg_abs_coeffs
                }).sort_values(by='importance_abs_coef_mean', ascending=False)
            elif feature_coefficients.ndim == 1:
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': feature_coefficients,
                    'importance_abs_coef': np.abs(feature_coefficients)
                }).sort_values(by='importance_abs_coef', ascending=False)
            
            if importance_df is not None:
                print("\nFeature Importance (based on model coefficients):")
                print(importance_df)
                plt.figure(figsize=(10, 8))
                col_to_plot = 'importance_abs_coef_mean' if 'importance_abs_coef_mean' in importance_df.columns else 'importance_abs_coef'
                sns.barplot(x=col_to_plot, y='feature', data=importance_df)
                plt.title('Top Features (by model coefficients)')
                plt.tight_layout()
                plt.show(block=False)
            else:
                print("  Model coefficients have an unexpected shape or are missing.")
        except Exception as e:
            print(f"  Error analyzing model coefficients: {e}")
    elif model_type == "mord":
        print("  Model is of type 'mord' but lacks 'coef_' attribute. Coefficient analysis is not possible.")


    print("\nEstimating feature importance using a helper Random Forest...")
    try:
        X_for_rf = X_test_scaled.values if isinstance(X_test_scaled, pd.DataFrame) else X_test_scaled
        y_for_rf = y_test_scalar.values.ravel() if isinstance(y_test_scalar, pd.Series) else y_test_scalar.ravel()


        if X_for_rf is None or y_for_rf is None or len(X_for_rf) == 0 or len(y_for_rf) == 0:
            print("  Not enough data to train the helper Random Forest.")
        elif len(np.unique(y_for_rf)) < 2 : 
            print(f"  Not enough classes ({len(np.unique(y_for_rf))}) in y_test_scalar for Random Forest. At least 2 are required.")
        else:
            rf_model_temp = RandomForestClassifier(n_estimators=100, random_state=4, n_jobs=-1)
            rf_model_temp.fit(X_for_rf, y_for_rf)

            importances_rf = rf_model_temp.feature_importances_
            feature_importance_rf_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances_rf
            }).sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10, 11))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_rf_df, color="steelblue")
            plt.title('Feature Importance (by Gini impurity)', fontsize=18)
            plt.xlabel('Importance', fontsize=16)
            plt.ylabel('Feature', fontsize=16)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt.tight_layout()
            plt.show(block=False)
    except Exception as e:
        print(f"  Error calculating RF importance: {e}")
        import traceback
        traceback.print_exc()
    print("\n--- Feature Importance Analysis Complete ---")
    plt.show()