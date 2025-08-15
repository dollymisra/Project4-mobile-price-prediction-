#!/usr/bin/env python
# coding: utf-8

# ### =========================
# ## Mobile Price Prediction 
# ## =========================

# ### Imports & settings

# In[50]:


import os
import re
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance

import joblib

# Output directory
OUT_DIR = "/mnt/data/mobile_price_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def save_and_show(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()
    print("Saved:", path)


# ### -------------------------
# ## 1) Data Loading & Exploration
# ### -------------------------
# 

# ### 1.1 Load dataset

# In[51]:


df = pd.read_csv('mobiledata.csv')
print("Dataset Shape:", df.shape)
df.head()


# ### 1.2 Drop index column if exists

# In[52]:


if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)


# ### 1.3 Backup raw data

# In[53]:


df_raw = df.copy()


# In[54]:


## 1.4 Quick info
print(df.info())
print(df.head())


# 
# ### -------------------------
# ## 2) Preprocessing(missing values, type conversion, basic cleaning)
# ## & Feature Engineering
# ### -------------------------
# 

# #### 2.1 Convert 'Prize' to numeric
# 

# In[55]:


df['Prize'] = df['Prize'].apply(lambda x: float(str(x).replace(",", "")) if pd.notna(x) else np.nan)


# #### Insight: The Prize column has been converted to numeric, removing commas to ensure proper float values for analysis and modeling.

# #### 2.2 Parse camera columns

# In[56]:


for col_in, col_out in [('Rear Camera','Rear Camera MP'), ('Front Camera','Front Camera MP')]:
    if col_in in df.columns:
        df[col_out] = df[col_in].apply(lambda x: float(re.search(r'(\d+(\.\d+)?)', str(x)).group(1)) if pd.notna(x) and re.search(r'(\d+(\.\d+)?)', str(x)) else np.nan)
    else:
        df[col_out] = np.nan


# #### Insight: Rear and front camera columns have been standardized to numeric MP values, extracting numbers from text while handling missing or malformed entries.

# #### 2.3 Rename battery column if needed
# 

# In[57]:


if 'Battery_' in df.columns:
    df.rename(columns={'Battery_':'Battery'}, inplace=True)


# #### Insight: The column Battery_ has been renamed to Battery for consistency and easier reference.

# #### 2.4 Handle missing values

# In[58]:


for col in df.columns:
    if df[col].isnull().sum()>0:
        if df[col].dtype in [np.float64,np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode().iloc[0], inplace=True)


# #### Insight: All missing values have been handled: numerical columns filled with the median and categorical columns with the mode to ensure a complete dataset.

# #### 2.5 Feature engineering

# In[59]:


if 'Rear Camera MP' in df.columns and 'Front Camera MP' in df.columns:
    df['Total Camera MP'] = df['Rear Camera MP'] + df['Front Camera MP']
    df['Rear_front_ratio'] = df['Rear Camera MP'] / df['Front Camera MP'].replace(0,np.nan)
    df['Rear_front_ratio'] = df['Rear_front_ratio'].fillna(0)

if 'RAM' in df.columns:
    df['High_RAM'] = (df['RAM']>=6).astype(int)
else:
    df['High_RAM'] = 0

if 'Memory' in df.columns:
    df['High_Storage'] = (df['Memory']>=128).astype(int)
else:
    df['High_Storage'] = 0


# #### Insight: New features created: Total Camera MP sums rear and front cameras, Rear/Front ratio captures camera balance, and binary flags High_RAM (≥6 GB) and High_Storage (≥128 GB) highlight premium specs.

# 
# 
# ### -------------------------
# ## 3) Exploratory Data Analysis (EDA)
# ### -------------------------

# #### 3.1 Price distribution

# In[60]:


fig = plt.figure(figsize=(8,4))
sns.histplot(df['Prize'].dropna(), bins=30, kde=True)
plt.title("Distribution of Mobile Phone Prices")
save_and_show(fig, "01_price_distribution.png")


# #### Insight: Mobile phone prices are right-skewed, with most devices clustered at lower to mid-range prices and fewer high-end models.

# #### 3.2 Price boxplot

# In[61]:


fig = plt.figure(figsize=(6,3))
sns.boxplot(x=df['Prize'].dropna())
plt.title("Price Boxplot")
save_and_show(fig, "02_price_boxplot.png")


# #### Insight: The price distribution shows some high-value outliers, suggesting a wide range of mobile prices with a few premium devices.

# #### 3.3 Top Models countplot

# In[62]:


if 'Model' in df.columns:
    top_models = df['Model'].value_counts().nlargest(15).index
    fig = plt.figure(figsize=(10,5))
    sns.countplot(y='Model', data=df[df['Model'].isin(top_models)], order=top_models)
    plt.title("Top 15 Models by Count")
    save_and_show(fig, "03_top_models_count.png")


# #### Insight: A small set of models dominates sales, with the top 15 models representing the majority of devices in the dataset.

# #### 3.4 Top Processors countplot

# In[63]:


if 'Processor_' in df.columns:
    top_procs = df['Processor_'].value_counts().nlargest(12).index
    fig = plt.figure(figsize=(10,5))
    sns.countplot(y='Processor_', data=df[df['Processor_'].isin(top_procs)], order=top_procs)
    plt.title("Top Processors by Count")
    save_and_show(fig, "04_top_processors_count.png")


# #### Insight: A few processor types dominate the market, with the top 12 accounting for the majority of devices, indicating limited variety in popular processors.

# #### 3.5 Numeric feature distributions

# In[64]:


num_feats = ['Memory','RAM','Battery','Rear Camera MP','Front Camera MP','Mobile Height']
num_feats = [c for c in num_feats if c in df.columns]
for col in num_feats:
    fig = plt.figure(figsize=(6,3))
    sns.histplot(df[col].dropna(), bins=25, kde=True)
    plt.title(f"Distribution of {col}")
    save_and_show(fig, f"dist_{col.replace(' ','_')}.png")


# #### Insight: The numerical features show varied distributions—some are right-skewed (e.g., Battery, Rear Camera MP), while others are more symmetric—indicating diverse ranges and potential for transformation before modeling.

# #### 3.6 Average price by RAM & Memory

# In[65]:


if 'RAM' in df.columns:
    ram_price = df.groupby('RAM')['Prize'].median().reset_index().sort_values('RAM')
    fig = plt.figure(figsize=(8,4))
    sns.barplot(x='RAM', y='Prize', data=ram_price)
    plt.title("Median Price by RAM (GB)")
    save_and_show(fig, "avgprice_by_RAM.png")


# #### Insight: Phones with more RAM tend to have higher median prices, indicating RAM significantly influences mobile pricing.

# In[66]:


if 'Memory' in df.columns:
    mem_price = df.groupby('Memory')['Prize'].median().reset_index().sort_values('Memory')
    fig = plt.figure(figsize=(8,4))
    sns.barplot(x='Memory', y='Prize', data=mem_price)
    plt.title("Median Price by Storage Memory (GB)")
    save_and_show(fig, "avgprice_by_Memory.png")


# #### Insight: Higher storage memory generally corresponds to a higher median price, showing that memory is a key factor in mobile pricing.

# ### -------------------------
# ## 4) MORE PREPROCESSING & FEATURE ENGINEERING
# ### -------------------------
# #### - Handle missing values
# #### - Create derived features
# #### - Option to remove outliers
# ### -------------------------

# #### 4.1 Missing values treatment (if any)

# In[67]:


missing_pct = df.isnull().mean().sort_values(ascending=False)
print("\nMissing percentage (desc):\n", (missing_pct*100).round(2))


# #### Insight: Some columns have a high percentage of missing values, highlighting the need for targeted data cleaning.

# #### Strategy: if numeric columns have a few missing values, fill with median; for many missing values, consider dropping feature
# 

# In[68]:


for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode().iloc[0], inplace=True)


# #### Insight: Fills missing numeric values with the median and categorical values with the mode to ensure a complete dataset for modeling.

# #### 4.2 Feature engineering ideas implemented:
# 
# 

# #### - Total camera (sum of rear + front)

# In[69]:


if 'Rear Camera MP' in df.columns and 'Front Camera MP' in df.columns:
    df['Total Camera MP'] = df['Rear Camera MP'].fillna(0) + df['Front Camera MP'].fillna(0)


# #### Insight: Creates a feature for the total camera megapixels by summing rear and front camera resolutions, reflecting overall camera capability.

# #### - Camera ratio (rear/front) - avoid division by zero

# In[70]:


if 'Rear Camera MP' in df.columns and 'Front Camera MP' in df.columns:
    df['Rear_front_ratio'] = df['Rear Camera MP'] / (df['Front Camera MP'].replace(0, np.nan))
    df['Rear_front_ratio'] = df['Rear_front_ratio'].fillna(0)


# #### Insight: Generates a new feature representing the ratio of rear to front camera megapixels, which may highlight devices with a strong emphasis on either rear or front photography.

# #### - Create a simple high_spec flag: RAM>=6 or Memory>=128

# In[71]:


df['High_RAM'] = (df['RAM'] >= 6).astype(int) if 'RAM' in df.columns else 0
df['High_Storage'] = (df['Memory'] >= 128).astype(int) if 'Memory' in df.columns else 0


# #### Insight: Creates binary features indicating whether a phone has high RAM (≥ 6 GB) and high storage (≥ 128 GB), enabling the model to capture performance-related effects.

# #### 4.3 Optionally remove extreme outliers (IQR method) 
# 

# In[72]:


REMOVE_OUTLIERS = False
if REMOVE_OUTLIERS:
    numeric_for_outliers = ['Prize','RAM','Memory','Battery','Rear Camera MP','Front Camera MP']
    numeric_for_outliers = [c for c in numeric_for_outliers if c in df.columns]
    for col in numeric_for_outliers:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = df.shape[0]
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        after = df.shape[0]
        print(f"Removed {before - after} rows as outliers from {col}")

print("Shape after preprocessing:", df.shape)


# #### Insight: Optionally removes outliers from key numeric features using the IQR method to improve model robustness and reduce noise.

# ### -------------------------
# ## 5) Feature Selection
# ####  Methods shown:
# #### - correlation with target
# ####  - SelectKBest (f_regression) on numeric features
# ####  - RandomForest feature importances (after encoding in pipeline)
# #### -------------------------

# ##### 5.1 Correlation with target

# In[73]:


numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('Prize') if 'Prize' in numeric_features else None
corr_with_target = df[numeric_features + ['Prize']].corr()['Prize'].abs().sort_values(ascending=False)
print("\nAbsolute correlation with Prize (top):")
display(corr_with_target.head(15))


# #### Insight: Calculates and ranks the absolute correlation of numeric features with the target variable to identify the most influential predictors.

# #### 5.2 SelectKBest

# In[74]:


k = min(8, len(numeric_features))
X_num = df[numeric_features].fillna(0)
y = df['Prize']
skb = SelectKBest(score_func=f_regression, k=k)
skb.fit(X_num, y)
scores = pd.Series(skb.scores_, index=numeric_features).sort_values(ascending=False)
print(f"Top {k} features by SelectKBest:")
print(scores.head(k))


# #### Insight: Uses SelectKBest with ANOVA F-test to rank and select the top numerical features most correlated with the target variable.

# ### -------------------------
# ### 6) MODEL BUILDING: Baseline -> RandomForest -> GridSearchCV
# ### -------------------------
# #### We'll create pipelines with ColumnTransformer and try:
# ####  - Linear Regression baseline
# ####  - RandomForest (default)
# ####  - GridSearchCV for RandomForest (optional/time-consuming)
# ### -------------------------
# 

# #### 6.1 Define features to use (drop raw string camera columns, keep engineered features)

# In[75]:


drop_cols = ['Rear Camera', 'Front Camera']  # remove raw string camera columns
features = [c for c in df.columns if c not in ['Prize'] + drop_cols]

# Keep only categorical columns that actually exist in features
cat_cols = [c for c in ['Model', 'Colour', 'Processor_'] if c in features]

# Remaining numeric columns
num_cols = [c for c in features if c not in cat_cols]

print("\nCategorical columns:", cat_cols)
print("Numeric columns:", num_cols)


# #### Insight: Prepares the feature lists by dropping unused camera columns, identifying available categorical and numerical variables for preprocessing.

# #### 6.2 Train-Test Split

# In[76]:


X = df[cat_cols + num_cols]
y = df['Prize']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### Insight: Splits the dataset into training and test sets, separating features from the target variable to enable model training and evaluation.

# #### 6.3 ColumnTransformer with OneHotEncoder (for categorical) and StandardScaler (for numeric)

# In[77]:


preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols),
    ("num", StandardScaler(), num_cols)  
])


# #### Insight: Defines a ColumnTransformer to one-hot encode categorical features and standardize numerical features, preparing data for modeling.

# #### 6.4 Pipelines
# #### Baseline: linear regression

# In[78]:


pipe_lr = Pipeline(steps=[('preproc', preprocessor), ('lr', LinearRegression())])


# #### Insight: Builds a pipeline that applies preprocessing steps before fitting a Linear Regression model for baseline performance comparison.

# #### 6.5 Tree: RandomForest

# In[79]:


pipe_rf = Pipeline(steps=[('preproc', preprocessor), ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))])


# #### Insight: Creates a pipeline combining preprocessing and a RandomForestRegressor, ensuring data transformation and modeling occur seamlessly in one workflow.

# 
# 
# #### 6.6  Cross-validation

# In[80]:


from sklearn.model_selection import cross_val_score

print("\nCross-validating baseline models (5-fold):")
for name, pipe in [('LinearRegression', pipe_lr), ('RandomForest', pipe_rf)]:
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"{name}: CV R2 mean={scores.mean():.4f}, std={scores.std():.4f}")


# #### Insight: Performs 5-fold cross-validation to compare baseline Linear Regression and RandomForest models based on R² scores.

# #### 6.6 Fit RandomForest

# In[81]:


print("\nFitting RandomForest on training data...")
pipe_rf.fit(X_train, y_train)


# #### Insight: Trains the RandomForest pipeline on the prepared training data to learn patterns for price prediction.

# #### 6.7 GridSearchCV for RandomForest

# In[82]:


DO_GRID_SEARCH = False
if DO_GRID_SEARCH:
    param_grid = {
        'rf__n_estimators': [100, 250],
        'rf__max_depth': [None, 10, 25],
        'rf__max_features': ['sqrt', 0.5]
    }
    gs = GridSearchCV(pipe_rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print("Best params:", gs.best_params_)
    print("Best CV R2:", gs.best_score_)
    pipe_rf = gs.best_estimator_  # replace with best model


# #### Insight: performs a grid search to find the optimal RandomForest hyperparameters, improving model performance through cross-validation.

# ### -------------------------
# ### 7) EVALUATION 
# ### -------------------------
# #### - MAE, RMSE, R2
# #### - Predicted vs Actual scatter
# #### - Residual plot and distribution
# #### -------------------------
# 

# #### 7.1 Predictions

# In[83]:


y_pred = pipe_rf.predict(X_test)


# #### Insight: Generates predicted values for the test set using the trained RandomForest pipeline.

# 
# 
# #### 7.2 Metrics

# In[84]:


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nTest set evaluation metrics:")
print(f"MAE = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R2 = {r2:.4f}")


# #### Insight: Calculates MAE, RMSE, and R² on the test set to quantify the model’s prediction error and overall explanatory power.

# #### 7.3 Save metrics to a CSV

# In[85]:


metrics_df = pd.DataFrame([{'MAE': mae, 'RMSE': rmse, 'R2': r2}])
metrics_df.to_csv(os.path.join(OUT_DIR, "model_metrics.csv"), index=False)
print("Saved metrics to model_metrics.csv")


# #### Insight: This step records key performance metrics—MAE, RMSE, and R²—to evaluate the model’s accuracy and saves them for reference and comparison.

# #### 7.4 Predicted vs Actual scatter

# In[86]:


fig = plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') 
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
save_and_show(fig, "predicted_vs_actual.png")


# #### Insight: The actual vs predicted plot compares model predictions with true values, where points close to the 45° line indicate high prediction accuracy

# #### 7.4 Residuals

# In[87]:


residuals = y_test - y_pred
fig = plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Residuals distribution (Actual - Predicted)")
save_and_show(fig, "residuals_distribution.png")


# #### Insight: The residuals distribution shows how prediction errors are spread, with a roughly symmetric, centered shape indicating minimal bias in the model’s predictions.

# In[88]:


fig = plt.figure(figsize=(6,4))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
save_and_show(fig, "residuals_vs_predicted.png")


# #### Insight: The residuals vs predicted plot helps detect patterns or bias in predictions, with random scatter around zero indicating a well-fitted model.

# ### -------------------------
# ## 8) Feature Importance Visualization 
#  #### Get feature names after OneHotEncoder + numeric features
# #### - Plot top N importances
# #### - Permutation importance (optional)
# #### - SHAP (optional if installed)
# ### -------------------------
# ### -------------------------

# #### 8.1 Extract feature names from ColumnTransformer

# In[89]:


def get_feature_names_from_coltransformer(coltransformer):
    # Assumes transformers are [('cat', OneHotEncoder, cat_cols), ('num', StandardScaler, num_cols)]
    feature_names = []
    for name, trans, cols in coltransformer.transformers_:
        if name == 'remainder':
            continue
        if hasattr(trans, 'get_feature_names_out'):
            try:
                fn = trans.get_feature_names_out(cols)
            except:
                # If transformer is a pipeline or scaler, fallback:
                fn = cols
        else:
            fn = cols
        feature_names.extend(list(fn))
    return feature_names


# #### Insight: This function extracts readable feature names from a ColumnTransformer, ensuring both encoded categorical and scaled numerical features are captured for interpretation and analysis.

# In[90]:


# preprocessor is inside pipeline at step 'preproc'
preproc = pipe_rf.named_steps['preproc']
# If we used ColumnTransformer directly, get feature names:
try:
    # OneHotEncoder integrated in ColumnTransformer: use get_feature_names_out
    cat_feature_names = pipe_rf.named_steps['preproc'].transformers_[0][1].get_feature_names_out(cat_cols)
    feature_names = list(cat_feature_names) + num_cols
except Exception as e:
    
    feature_names = cat_cols + num_cols


# #### Insight: This block retrieves the full list of feature names after preprocessing, combining encoded categorical features with numerical features for use in model interpretation.

# #### 8.2 Get importances from RandomForest
# 

# In[91]:


rf_model = pipe_rf.named_steps['rf']
importances = rf_model.feature_importances_
fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
print("\nTop feature importances (RandomForest):")
display(fi_df.head(20))


# #### Insight: RandomForest feature importance ranking reveals which variables have the greatest impact on the model’s predictions based on their contribution to reducing prediction error.

# #### 8.3 Plot top 20 feature importances

# In[92]:


top_n = 20
fig = plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='feature', data=fi_df.head(top_n))
plt.title(f"Top {top_n} Feature Importances (RandomForest)")
save_and_show(fig, "top_feature_importances.png")


# #### Insight: The top 20 RandomForest feature importances highlight the predictors that contribute most to the model’s decisions, ranked by their relative influence

# #### 8.4 Permutation importance (more reliable sometimes)

# In[93]:


# Transform the test set
X_test_transformed = pipe_rf.named_steps['preproc'].transform(X_test)
feature_names = pipe_rf.named_steps['preproc'].get_feature_names_out()

# Extract RandomForest model
rf_model = pipe_rf.named_steps['rf']

# Run permutation importance on the full set (but with fewer repeats for speed)
perm_res_full = permutation_importance(rf_model, X_test_transformed, y_test,
                                       n_repeats=5, random_state=42, n_jobs=-1)

# Convert to DataFrame
perm_df_full = pd.DataFrame({
    'feature': feature_names,
    'perm_mean': perm_res_full.importances_mean,
    'perm_std': perm_res_full.importances_std
})

# Select top 50 features from the result
perm_df = perm_df_full.sort_values('perm_mean', ascending=False).head(50)

# Save results
perm_df.to_csv(os.path.join(OUT_DIR, "permutation_importance_fast.csv"), index=False)
print("Saved top 50 permutation importances (fast) to CSV")

# Plot top 15
fig = plt.figure(figsize=(8,6))
sns.barplot(x='perm_mean', y='feature', data=perm_df.head(15))
plt.title("Top 15 Permutation Importances (Fast Version)")
save_and_show(fig, "top_permutation_importances_fast.png")


# #### Insight: Fast permutation importance identifies the top 50 features influencing model accuracy, with the plotted top 15 providing a quick view of the most critical predictors.

# #### Plot top 15 permutation importances

# In[94]:


fig = plt.figure(figsize=(8,6))
sns.barplot(x='perm_mean', y='feature', data=perm_df.head(15))
plt.title("Top 15 Permutation Importances")
save_and_show(fig, "top_permutation_importances.png")


# #### Insight: The top 15 permutation importance scores show which features most strongly affect the model’s predictive accuracy, with higher values indicating greater influence.

# 
# #### SHAP
# 

# In[95]:


try:
    import shap
    X_sample = X_train.sample(min(200, X_train.shape[0]), random_state=42)
    X_preprocessed = pipe_rf.named_steps['preproc'].transform(X_sample)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_preprocessed)
    shap.summary_plot(shap_values, X_preprocessed, feature_names=feature_names, show=True)
except Exception as e:
    print("SHAP not available:", e)


# #### Insight: SHAP analysis reveals the features with the strongest influence on the RandomForest model’s predictions, providing an interpretable view of feature importance and direction of impact.

# ### -------------------------
# ## 8) Save Model & Artifacts
# ### -------------------------

# In[96]:


model_path = os.path.join(OUT_DIR, "rf_pipeline_model.joblib")
joblib.dump(pipe_rf, model_path)
print("Saved trained pipeline model to:", model_path)

fi_df.to_csv(os.path.join(OUT_DIR, "feature_importance_results.csv"), index=False)
print("Saved feature importance CSV to:", os.path.join(OUT_DIR, "feature_importance_results.csv"))

# Save a sample predictions CSV (actual vs predicted for test set)
pred_df = X_test.copy()
pred_df['actual_price'] = y_test
pred_df['predicted_price'] = y_pred
pred_df.reset_index(drop=True, inplace=True)
pred_df.to_csv(os.path.join(OUT_DIR, "test_set_predictions.csv"), index=False)
print("Saved test set predictions to:", os.path.join(OUT_DIR, "test_set_predictions.csv"))


#  ### -------------------------
# ### 10) End - Summary for report
# ### -------------------------
# 

# In[97]:


print("\n--- Summary ---")
print(f"Data shape after preprocessing: {df.shape}")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Test MAE: {mae:.2f} ; RMSE: {rmse:.2f} ; R2: {r2:.4f}")
print("All plots and CSVs saved to:", OUT_DIR)


# In[ ]:




