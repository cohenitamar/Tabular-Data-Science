# -------------------------------
# Import libraries and configure settings
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')  # Ignore warnings for clean output

# Pattern mining imports (using FPGrowth)
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Scikit-learn imports for modeling and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

sns.set()  # Use seaborn default theme for plots
plt.rcParams['figure.figsize'] = (8, 4)

# Define a set of columns that are treated as severity indicators (to skip during some transformations)
severity_columns = {"severity", "level", "lung_cancer", "stage"}


# -------------------------------
# Helper functions
# -------------------------------
def fix_cols(d):
    """
    Standardize DataFrame column names: lowercase and strip spaces.
    """
    d.columns = [c.strip().lower() for c in d.columns]
    return d


print("Imports and helper functions ready.")


def prepare_for_apriori(df, ignore_cols=None, bins='median'):
    """
    Convert numeric columns into binary columns (<= threshold and > threshold)
    and one-hot encode categorical columns.
    """
    if ignore_cols is None:
        ignore_cols = []
    df_temp = df.copy()
    out_df = pd.DataFrame()

    for col in df_temp.columns:
        if col in ignore_cols:  # Skip ignored columns
            continue

        if pd.api.types.is_numeric_dtype(df_temp[col]):
            # Use median or mean as threshold
            threshold = df_temp[col].median() if bins == 'median' else df_temp[col].mean()
            out_df[f"{col}_<=_{round(threshold, 2)}"] = (df_temp[col] <= threshold).astype(int)
            out_df[f"{col}_>_{round(threshold, 2)}"] = (df_temp[col] > threshold).astype(int)
        else:
            # One-hot encode categoricals
            unique_vals = df_temp[col].dropna().unique()
            for uv in unique_vals:
                uv_str = str(uv).strip().replace(' ', '_')
                out_df[f"{col}={uv_str}"] = (df_temp[col] == uv).astype(int)

    # Remove columns with no variability
    out_df = out_df.loc[:, (out_df.sum(axis=0) > 0)]
    return out_df


def run_apriori(df, ignore_cols=None, min_support=0.25, bins='median'):
    """
    Prepares data for mining (using prepare_for_apriori) and runs FPGrowth.
    Returns frequent itemsets sorted by support.
    """
    if df.empty:
        return pd.DataFrame()

    encoded = prepare_for_apriori(df, ignore_cols=ignore_cols, bins=bins)
    freq_items = fpgrowth(encoded, min_support=min_support, use_colnames=True)
    freq_items.sort_values('support', ascending=False, inplace=True)
    return freq_items


print("FPGrowth functions ready.")

# -------------------------------
# Load datasets and adjust data
# -------------------------------
df1 = pd.read_csv("datasets/cancer_patients_air_pollution.csv")
df2 = pd.read_csv("datasets/lung_cancer_prediction.csv")
df3 = pd.read_csv("datasets/lung_cancer_risk_dataset.csv")
df4 = pd.read_csv("datasets/data.csv")


def adjust_f_stage(df):
    """
    For the 'Stage' column, set 0/1 for stages 0/1 and 1 for stages 2/3.
    """
    df['Stage'] = pd.to_numeric(df['Stage'], errors='coerce')
    df.loc[df['Stage'].isin([0, 1]), 'Stage'] = 0
    df.loc[df['Stage'].isin([2, 3]), 'Stage'] = 1
    return df


df3 = adjust_f_stage(df3)

# Standardize column names
df1 = fix_cols(df1)
df2 = fix_cols(df2)
df3 = fix_cols(df3)
df4 = fix_cols(df4)

print("Shapes of each dataframe:")
print("df1:", df1.shape)
print("df2:", df2.shape)
print("df3:", df3.shape)
print("df4:", df4.shape)


# -------------------------------
# Exploratory Data Analysis (EDA)
# -------------------------------
def basic_eda(df, name):
    """
    Print basic info, head, and summary statistics of the DataFrame.
    """
    print(f"\n=== {name} ===")
    if df.empty:
        print("Empty DataFrame")
        return
    print(df.info())
    print(df.head(3))
    print(df.describe(include='all'))


basic_eda(df1, "df1")
basic_eda(df2, "df2")
basic_eda(df3, "df3")
basic_eda(df4, "df4")


def visualize_data(df, name="DataFrame", max_cat_values=10):
    """
    Plot histograms for numeric columns, count plots for low-cardinality categoricals,
    and a correlation heatmap for numeric data.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df.empty:
        print(f"{name} is empty, skipping visualization.")
        return

    # Plot numeric histograms
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    if numeric_cols.any():
        df[numeric_cols].hist(figsize=(10, 6), bins=15)
        plt.suptitle(f"{name} - Numeric Histograms", y=1.02, fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No numeric columns in {name} to plot histograms.")

    # Plot count plots for categoricals with few unique values
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if df[col].nunique(dropna=True) <= max_cat_values:
            plt.figure()
            sns.countplot(x=col, data=df)
            plt.title(f"{name} - Count Plot of '{col}'")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # Plot a correlation heatmap if more than one numeric column
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='viridis')
        plt.title(f"{name} - Correlation Heatmap")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Not enough numeric columns in {name} for a correlation heatmap.")


visualize_data(df1, name="df1")
visualize_data(df2, name="df2")
visualize_data(df3, name="df3")
visualize_data(df4, name="df4")

# -------------------------------
# Pattern Mining and Feature Creation
# -------------------------------
from xgboost import XGBRegressor
from itertools import combinations


def pattern_mining(df, name="DF", severity_columns=None, min_support=0.20):
    """
    Run FPGrowth on the dataset to find frequent itemsets.
    Prints top 10 itemsets and shows a bar plot of the top 5.
    """
    if df.empty:
        print(f"Skipping Apriori for {name}, it's empty.")
        return pd.DataFrame()

    print(f"\n--- Pattern Mining on {name} (min_support={min_support}) ---")
    freq_items = run_apriori(df, ignore_cols=severity_columns if severity_columns else set(), min_support=min_support)
    if freq_items.empty:
        print("No frequent itemsets found.")
        return freq_items

    print("Top itemsets:")
    print(freq_items.head(10))

    top_n = 5 if freq_items.shape[0] >= 5 else freq_items.shape[0]
    top_items = freq_items.head(top_n).copy()
    top_items['itemset_str'] = top_items['itemsets'].apply(lambda x: ','.join(list(x)))

    plt.figure()
    sns.barplot(data=top_items, x='support', y='itemset_str')
    plt.title(f"Top {top_n} itemsets by support ({name})")
    plt.xlabel("Support")
    plt.ylabel("Itemsets")
    plt.show()

    return freq_items


def create_association_features(df, freq_items, top_k=5, name="DF"):
    """
    Create new binary features from the top frequent itemsets.
    """
    if freq_items.empty:
        print(f"{name}: No frequent itemsets to create features from.")
        return df

    freq_items_sorted = freq_items.sort_values('support', ascending=False)
    top_itemsets = freq_items_sorted.head(top_k).copy()
    df_new = df.copy()

    for idx, row in top_itemsets.iterrows():
        itemset = row['itemsets']
        sorted_conds = sorted(list(itemset))
        combined_name = "_AND_".join(sorted_conds).replace("=", "_EQ_")
        # Clean up feature name by replacing forbidden characters
        for ch in ["[", "]", "<", ">", "(", ")", ":"]:
            combined_name = combined_name.replace(ch, "_")
        feature_name = f"assoc_rule_{combined_name}"

        def check_row_contains_itemset(df_row):
            # Check if all conditions in the itemset are met for a row
            for condition in itemset:
                if '=' in condition:
                    col, val = condition.split('=', 1)
                    if col not in df_row or str(df_row[col]) != val:
                        return 0
                else:
                    if condition not in df_row or df_row[condition] != 1:
                        return 0
            return 1

        df_new[feature_name] = df_new.apply(check_row_contains_itemset, axis=1)
        print(f"Created association feature '{feature_name}' from itemset: {itemset}")

    return df_new


def feature_engineering(df, severity_columns=None, name="DF"):
    """
    For each numeric column (excluding severity columns), create a new binary feature
    indicating if the value is above the mean.
    """
    if severity_columns is None:
        severity_columns = set()
    if df.empty:
        print(f"{name}: empty, skipping feature engineering.")
        return df

    df_new = df.copy()
    numeric_cols = [c for c in df_new.select_dtypes(include=[np.number]).columns if c not in severity_columns]
    if not numeric_cols:
        print(f"No numeric columns found in {name} for feature engineering.")
        return df_new

    for col in numeric_cols:
        mean_val = df_new[col].mean(skipna=True)
        new_feat_name = col + "_above_mean"
        df_new[new_feat_name] = (df_new[col] > mean_val).astype(int)
        print(f"Created feature '{new_feat_name}' in {name} (mean={mean_val:.2f})")
    return df_new


def create_combination_features(df, target_col, severity_columns=None, name="DF"):
    """
    Create pairwise combination features for all columns except the target and severity columns.
    """
    if severity_columns is None:
        severity_columns = set()
    if df.empty:
        print(f"{name}: empty, skipping combination features.")
        return df

    df_new = df.copy()
    exclude = set(severity_columns) | {target_col}
    candidate_cols = [c for c in df_new.columns if c not in exclude]
    pairs = list(combinations(candidate_cols, 2))
    print(f"{name}: Creating combination features for {len(pairs)} pairs...")

    for (colA, colB) in pairs:
        new_col_name = f"{colA}__{colB}"
        for ch in ["[", "]", "<", ">", "(", ")", ":"]:
            new_col_name = new_col_name.replace(ch, "_")
        df_new[new_col_name] = df_new[colA].astype(str) + "_" + df_new[colB].astype(str)
    return df_new


# -------------------------------
# Feature Selection and XGBoost Regression
# -------------------------------
def select_best_features(df, target_col, top_n=5, severity_columns=None):
    """
    Split data into features and target, run an XGBoost regressor with grid search,
    and return the top_n features based on importance.
    """
    if df.empty or target_col not in df.columns:
        print(f"select_best_features: Invalid df or missing target '{target_col}'.")
        return []

    if severity_columns is None:
        severity_columns = set()

    df_model = df.dropna(subset=[target_col]).copy()
    y = df_model[target_col].values

    if not np.issubdtype(y.dtype, np.number):
        print(f"WARNING: target '{target_col}' is not numeric. Converting...")
        try:
            y = pd.to_numeric(y)
        except ValueError:
            print(f"ERROR: Could not convert '{target_col}' to numeric.")
            return []

    X = df_model.drop(columns=[target_col])
    drop_cols = list(severity_columns.intersection(X.columns))
    if drop_cols:
        X.drop(columns=drop_cols, inplace=True)

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.median(numeric_only=True))
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = StandardScaler().fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [1, 2, 3],
        'learning_rate': [1.0, 0.1, 0.01],
        'reg_alpha': [0, 10, 20],
        'reg_lambda': [0, 10, 20]
    }
    xgb_reg = XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42)
    grid_search = GridSearchCV(xgb_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    importances = best_model.feature_importances_
    features = X.columns
    feat_imp_pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in feat_imp_pairs[:top_n]]

    print("\nFeature importances (from best XGB model):")
    for feat, imp in feat_imp_pairs:
        print(f"  {feat}: {imp:.4f}")
    print(f"\nTop {len(top_features)} features (max {top_n}): {top_features}\n")
    return top_features


def run_xgboost_regression(df, target_col='some_numeric_target', dataset_name="Dataset"):
    """
    Train an XGBoost regressor on the DataFrame and print regression metrics.
    Also computes classification-like metrics using a threshold.
    """
    if df.empty or target_col not in df.columns:
        print(f"{dataset_name}: Missing data for target '{target_col}'.")
        return None

    df_model = df.dropna(subset=[target_col]).copy()
    y = df_model[target_col]
    X = df_model.drop(columns=[target_col])

    if not np.issubdtype(y.dtype, np.number):
        print(f"WARNING: '{target_col}' is not numeric. Converting...")
        try:
            y = pd.to_numeric(y)
        except ValueError:
            print(f"ERROR: Could not convert '{target_col}' to numeric.")
            return None

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.median(numeric_only=True))
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = StandardScaler().fit_transform(X[num_cols])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    xgb_reg = XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42)
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [1, 2, 3],
        'learning_rate': [1.0, 0.1, 0.01],
        'reg_alpha': [0, 10, 20],
        'reg_lambda': [0, 10, 20]
    }
    grid_search = GridSearchCV(xgb_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_

    print(f"\n--- {dataset_name} ---")
    print("Best Parameters:", grid_search.best_params_)

    y_pred = best_xgb.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE:  {rmse:.4f}")
    print(f"R^2:   {r2:.4f}")

    # Create binary predictions for classification-like metrics
    preds_bin = (y_pred >= 0.1).astype(int)
    acc = accuracy_score(y_test, preds_bin)
    cr_dict = classification_report(y_test, preds_bin, output_dict=True)
    weighted_avg = cr_dict.get('weighted avg', {})
    prec = weighted_avg.get('precision', 0.0)
    rec = weighted_avg.get('recall', 0.0)
    f1 = weighted_avg.get('f1-score', 0.0)
    print("\nClassification Metrics (threshold=0.1):")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds_bin))

    return {'RMSE': rmse, 'R^2': r2, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


# -------------------------------
# Utility: Sanitize feature names
# -------------------------------
import re


def sanitize_feature_name(name):
    """
    Replace forbidden characters ([, ], <, >) with underscores.
    """
    return re.sub(r'[\[\]<>]', '_', name)


# -------------------------------
# Run Pattern Mining and Association Rule Extraction
# -------------------------------
freq_df1 = pattern_mining(df1, name="df1", severity_columns={"level"}, min_support=0.25)
freq_df2 = pattern_mining(df2, name="df2", severity_columns={"lung_cancer"}, min_support=0.25)
freq_df3 = pattern_mining(df3, name="df3", severity_columns={"stage"}, min_support=0.25)
freq_df4 = pattern_mining(df4, name="df4", severity_columns={"lung_cancer"}, min_support=0.25)


def get_top_rules(freq_items, min_conf=0.6, min_lift=1.0, top_k=5):
    """
    Compute and return the top association rules from frequent itemsets.
    """
    if freq_items.empty:
        return pd.DataFrame()
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
    rules = rules[rules['lift'] >= min_lift].sort_values('confidence', ascending=False)
    return rules.head(top_k)


rules_df1 = get_top_rules(freq_df1, min_conf=0.6, min_lift=1.0, top_k=5)
rules_df2 = get_top_rules(freq_df2, min_conf=0.6, min_lift=1.0, top_k=5)
rules_df3 = get_top_rules(freq_df3, min_conf=0.6, min_lift=1.0, top_k=5)
rules_df4 = get_top_rules(freq_df4, min_conf=0.6, min_lift=1.0, top_k=5)


def create_rule_features(df, rules_df, name="DF"):
    """
    Create new binary features in df based on the antecedents of association rules.
    """
    if rules_df.empty:
        print(f"{name}: No association rules found for rule features.")
        return df

    df_new = df.copy()
    for i, row in rules_df.iterrows():
        antecedent = row['antecedents']
        sorted_conds = sorted(list(antecedent))
        cond_str = "_AND_".join(cond.replace("=", "_EQ_") for cond in sorted_conds)
        cond_str = sanitize_feature_name(cond_str)
        feature_name = sanitize_feature_name(f"rule_{i}_{cond_str}")

        def check_row_contains_antecedent(r):
            for cond in antecedent:
                if "=" in cond:
                    col, val = cond.split("=", 1)
                    if col not in r or str(r[col]) != val:
                        return 0
                else:
                    if cond not in r or r[cond] != 1:
                        return 0
            return 1

        df_new[feature_name] = df_new.apply(check_row_contains_antecedent, axis=1)
        print(f"{name}: Created rule feature '{feature_name}' from antecedent={antecedent}")
    return df_new


df1_assoc = create_association_features(df1, freq_df1, top_k=5, name="df1")
df2_assoc = create_association_features(df2, freq_df2, top_k=5, name="df2")
df3_assoc = create_association_features(df3, freq_df3, top_k=5, name="df3")
df4_assoc = create_association_features(df4, freq_df4, top_k=5, name="df4")

df1_rules_assoc = create_rule_features(df1_assoc, rules_df1, name="df1")
df2_rules_assoc = create_rule_features(df2_assoc, rules_df2, name="df2")
df3_rules_assoc = create_rule_features(df3_assoc, rules_df3, name="df3")
df4_rules_assoc = create_rule_features(df4_assoc, rules_df4, name="df4")

# -------------------------------
# Additional Feature Engineering
# -------------------------------
df1_fe = feature_engineering(df1_rules_assoc, severity_columns={"level"}, name="df1")
df2_fe = feature_engineering(df2_rules_assoc, severity_columns={"lung_cancer"}, name="df2")
df3_fe = feature_engineering(df3_rules_assoc, severity_columns={"stage"}, name="df3")
df4_fe = feature_engineering(df4_rules_assoc, severity_columns={"lung_cancer"}, name="df4")

df1_combined = create_combination_features(df1_fe, target_col="level", severity_columns={"level"}, name="df1")
df2_combined = create_combination_features(df2_fe, target_col="lung_cancer", severity_columns={"lung_cancer"},
                                           name="df2")
df3_combined = create_combination_features(df3_fe, target_col="stage", severity_columns={"stage"}, name="df3")
df4_combined = create_combination_features(df4_fe, target_col="lung_cancer", severity_columns={"lung_cancer"},
                                           name="df4")

# -------------------------------
# Feature Selection and Final Modeling
# -------------------------------
best_features_df1 = select_best_features(df1_combined, target_col="level", top_n=5, severity_columns={"level"})
best_features_df2 = select_best_features(df2_combined, target_col="lung_cancer", top_n=5,
                                         severity_columns={"lung_cancer"})
best_features_df3 = select_best_features(df3_combined, target_col="stage", top_n=5, severity_columns={"stage"})
best_features_df4 = select_best_features(df4_combined, target_col="lung_cancer", top_n=5,
                                         severity_columns={"lung_cancer"})

print("\n===== FINAL XGBREGRESSOR TRAINING & EVALUATION =====")

results1 = None
if best_features_df1:
    subset_cols = best_features_df1 + ["level"]
    results1 = run_xgboost_regression(df1_combined[subset_cols], target_col="level", dataset_name="df1_combined_top5")
    print("\nDF1 Results:", results1)
else:
    print("No top features found for df1; skipping final model.")

results2 = None
if best_features_df2:
    subset_cols = best_features_df2 + ["lung_cancer"]
    results2 = run_xgboost_regression(df2_combined[subset_cols], target_col="lung_cancer",
                                      dataset_name="df2_combined_top5")
    print("\nDF2 Results:", results2)
else:
    print("No top features found for df2; skipping final model.")

results3 = None
if best_features_df3:
    subset_cols = best_features_df3 + ["stage"]
    results3 = run_xgboost_regression(df3_combined[subset_cols], target_col="stage", dataset_name="df3_combined_top5")
    print("\nDF3 Results:", results3)
else:
    print("No top features found for df3; skipping final model.")

results4 = None
if best_features_df4:
    subset_cols = best_features_df4 + ["lung_cancer"]
    results4 = run_xgboost_regression(df4_combined[subset_cols], target_col="lung_cancer",
                                      dataset_name="df4_combined_top5")
    print("\nDF4 Results:", results4)
else:
    print("No top features found for df4; skipping final model.")

if results1 is not None:
    print(f"DF1 Accuracy: {results1['accuracy']:.3f}")
if results2 is not None:
    print(f"DF2 Accuracy: {results2['accuracy']:.3f}")
if results3 is not None:
    print(f"DF3 Accuracy: {results3['accuracy']:.3f}")
if results4 is not None:
    print(f"DF4 Accuracy: {results4['accuracy']:.3f}")


# -------------------------------
# Classification using XGBoost
# -------------------------------
def run_classification(df, target_col, name="DF"):
    from xgboost import XGBRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import numpy as np

    if df.empty or target_col not in df.columns:
        print(f"{name}: Missing data for target '{target_col}'. Skipping classification.")
        return None

    print(f"\n--- Classification on {name}, Target={target_col} ---")
    dtemp = df.dropna(subset=[target_col]).copy()

    if pd.api.types.is_numeric_dtype(dtemp[target_col]):
        unique_vals = dtemp[target_col].unique()
        if set(unique_vals).issubset({0, 1}):
            print(f"{name}: Target is already binary.")
            dtemp["_target_"] = dtemp[target_col]
        else:
            median_val = dtemp[target_col].median()
            dtemp["_target_"] = (dtemp[target_col] > median_val).astype(int)
            print(f"{name}: Numeric target binarized at median={median_val:.2f}")
    else:
        print(f"{name}: Categorical target, label encoding.")
        dtemp["_target_"] = LabelEncoder().fit_transform(dtemp[target_col].astype(str))

    if dtemp["_target_"].nunique() < 2:
        print(f"{name}: Only one class in target. Skipping.")
        return None

    if target_col in dtemp.columns:
        dtemp.drop(columns=[target_col], inplace=True)

    X = dtemp.drop(columns=["_target_"])
    y = dtemp["_target_"].values

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.median(numeric_only=True))
    if X.shape[1] < 1:
        print(f"No usable features left in {name}.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"{name}: y_train distribution: {np.bincount(y_train)}")
    print(f"{name}: y_test  distribution: {np.bincount(y_test)}")

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    regr = XGBRegressor(n_estimators=10, max_depth=1, learning_rate=0.1, reg_alpha=1, reg_lambda=10,
                        eval_metric='rmse', random_state=42)
    regr.fit(X_train_scaled, y_train)
    preds_continuous = regr.predict(X_test_scaled)
    preds = (preds_continuous >= 0.1).astype(int)

    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, preds))
    cr_dict = classification_report(y_test, preds, output_dict=True)
    weighted_avg = cr_dict.get('weighted avg', {})
    return {'accuracy': acc, 'precision': weighted_avg.get('precision'),
            'recall': weighted_avg.get('recall'), 'f1': weighted_avg.get('f1-score')}


metrics_df = []
m1 = run_classification(df1, target_col="level", name="df1")
if m1:
    m1['dataset'] = 'df1'
    metrics_df.append(m1)
m2 = run_classification(df2, target_col="lung_cancer", name="df2")
if m2:
    m2['dataset'] = 'df2'
    metrics_df.append(m2)
m3 = run_classification(df3, target_col="stage", name="df3")
if m3:
    m3['dataset'] = 'df3'
    metrics_df.append(m3)
m4 = run_classification(df4, target_col="lung_cancer", name="df4")
if m4:
    m4['dataset'] = 'df4'
    metrics_df.append(m4)

print("\nClassification Results:")
for row in metrics_df:
    print(row)
