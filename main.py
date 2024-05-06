import json

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from fuzzywuzzy import process
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
from scipy.special import boxcox1p
from scipy.stats import (anderson, boxcox, chi2, chi2_contingency, f, f_oneway,
                         iqr, kstest, kurtosis, levene, pearsonr, shapiro,
                         skew, yeojohnson)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso,
                                  LinearRegression, LogisticRegression, Ridge)
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler,
                                   OneHotEncoder, OrdinalEncoder,
                                   PolynomialFeatures, PowerTransformer,
                                   StandardScaler)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

config_path = '/home/armando/repos/personal/AutoML_pulse/config.json' 
with open(config_path, 'r') as file:
    config = json.load(file, object_pairs_hook=dict)

def detect_ordinality(col_data, column_name):
    if config.get(column_name) and config[column_name]['proposed_order']:
        proposed_order = [value for key, value in sorted(config[column_name]['proposed_order'].items())]        
        return (True, f"Column {column_name} is marked as having ordinality in the config file with proposed order: {proposed_order}")
    
    if col_data.dtype.name == 'category' and col_data.cat.ordered:
        return (True, "Column is in df marked as an ordered category")
        
    # Check for numeric patterns
    if col_data.apply(lambda x: str(x).isdigit()).all():
        return (True, "All values in the column are digits, suggests ordinality")
    
    # Check for ordinal patterns 
    ordinal_patterns = ['low', 'medium', 'high', 'none', 'mild', 'moderate', 'severe', 'excellent', 'good', 'fair', 'poor']
    threshold = 80 
    
    fuzzy_matches = set()
    for value in col_data.unique():
        for pattern in ordinal_patterns:
            match, score = process.extractOne(value.lower(), ordinal_patterns)
            if score > threshold:
                fuzzy_matches.add(match)

    if fuzzy_matches:
        return (True, f"Detected similar ordinal patterns in the column: {', '.join(fuzzy_matches)}.")

    order_symbols = ['<', '<=', '>', '>=']
    if any(symbol in value for value in col_data.unique() for symbol in order_symbols):
        return (True, "Column values indicate ordinality based on '<', '<=', '>', or '>=' symbols")
    
    return (False, "No clear ordinal patterns detected")

def get_contextual_cardinality_type(col_data, len_data):
    n_unique = col_data.nunique()
    relative_cardinality = n_unique / len_data

    low_threshold = max(5, int(0.01 * len_data))
    high_threshold = min(20, int(0.05 * len_data))
    dominant_category_threshold = 0.5 

    dominant_category_present = (col_data.value_counts(normalize=True) > dominant_category_threshold).any()

    if n_unique <= low_threshold:
        return 'low'
    elif low_threshold < n_unique <= high_threshold or (relative_cardinality < 0.1 and dominant_category_present):
        return 'medium'
    else:
        return 'high'
    
def is_suitable_for_embeddings(col_data):
    unique_ratio = col_data.nunique() / len(col_data)
    text_length = col_data.apply(lambda x: len(str(x))).mean()
    high_diversity = unique_ratio > 0.5  
    complex_text = text_length > 50 

    return high_diversity or complex_text
    
def evaluate_categorical_transformations(col_data, column_name, target=None, suggestions=None, n_splits=5):
    """
    OrdinalEncoder, Embeddings, and TargetEncoder, the categories need to be represented as numbers. 
    For OneHotEncoder and FrequencyEncoding, numeric representation is not strictly necessary.
    """
    
    # Adapt it based on the objective model
    # Weight of Evidence (WoE) Encoding
    # chi-squared tests or what where the ones for multicat and 1 target or multicat and multi target??
    response = {}
    len_data = len(col_data)
    cardinality_type = get_contextual_cardinality_type(col_data, len_data)
    is_ordinal, reason = detect_ordinality(col_data, column_name)

    if is_ordinal:
        response = {'transformation': 'OrdinalEncoder', 'reason': reason}
    elif cardinality_type in ['low', 'medium']:
        response = {'transformation': 'OneHotEncoder', 'reason': f"{cardinality_type.capitalize()} cardinality, suitable for OneHotEncoder"}
    else:
        if target is not None and is_suitable_for_embeddings(col_data):
            response = {'transformation': 'Embeddings', 'reason': "High cardinality with text data suitable for embeddings based on diversity, text complexity, or availability of pre-trained models"}
        elif target is not None:
            kf = KFold(n_splits=n_splits)
            scores = []
            for train_index, test_index in kf.split(col_data):
                encoder = TargetEncoder()
                encoder.fit(col_data.iloc[train_index], target.iloc[train_index])
                scores.append(encoder.score(col_data.iloc[test_index], target.iloc[test_index]))
            response = {'transformation': 'TargetEncoder', 'reason': f"High cardinality with mean CV score: {np.mean(scores):.2f} using TargetEncoder"}
        else:
            response = {'transformation': 'FrequencyEncoding', 'reason': "High cardinality without target provided, consider FrequencyEncoding"}

    return {**response, "suggestions": suggestions} if suggestions else response

    # Relative Cardinality: total number of records.
    # Category Frequency Distribution: distribution of categories
    # Low Observation Categories: Laplace OR Lidstone smoothing 
    # Model based: leave one out  OR James Stein encoding
    # Statistical Robustness
    
def evaluate_numerical_transformations(col_data):
    # Allow other distributions based on the model objective, bimodal or multimodal 
    # Adapt it based on the objective model
    # change the test based on dataset size (large and small)
    # Reinforcement Learning for Pipeline Optimization
    
    transformations = {}
    reasons = {}
        
    original_skew = abs(skew(col_data))
    original_kurtosis = kurtosis(col_data)
    transformations['original'] = original_skew 
    reasons['original'] = f"No transformation. Skewness: {original_skew:.2f}, Kurtosis: {original_kurtosis:.2f}"
    
    stat, p_value = shapiro(col_data.sample(min(1000, len(col_data))))  
    normality = "passes" if p_value > 0.05 else "fails"
    
    # ADd logarithmic and Box-Cox
    # Refine the skewness and kurtosis thresholds, maybe dynamic ?
    # Incorporate normality tests (Anderson-Darling)
    # Add  (heteroscedasticity) Variance Stabilization - log transformation or the Anscombe transform
    # add more scaling options and transformations
    # add a way to handle composite or assure normality (recusrive call the function until normality or the the test of desired distribution)
    if original_skew > 0.5 or normality == "fails": 
        pt = PowerTransformer(method='yeo-johnson')
        transformed_data = pt.fit_transform(col_data.values.reshape(-1, 1)).flatten()
        transformed_skew = abs(skew(transformed_data))
        transformations['yeojohnson'] = transformed_skew
        reasons['yeojohnson'] = f"Yeo-Johnson transformation. New Skewness: {transformed_skew:.2f}"
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(transformed_data.reshape(-1, 1)).flatten()
        scaled_skew = abs(skew(scaled_data))
        transformations['composite'] = scaled_skew
        reasons['composite'] = f"Yeo-Johnson + StandardScaler. Final Skewness: {scaled_skew:.2f}"
    
    best_transformation = min(transformations, key=transformations.get)
    return {'transformation': best_transformation, 'reason': reasons[best_transformation], 'normality_test': normality}
    
    
def little_mcar_test(column):

    observed_values = pd.crosstab(index=column.isna(), columns=column.notna())
    chi2, p_value, _, _ = chi2_contingency(observed_values)
    return p_value
    
def detect_missingness_pattern(df, column, target_column=None):
    # advanced it Little's MCAR test.
    # more tests for MNAR and MAR OR MCAR
    # Bootsrao hypothesis testin, sensitivity analysis for mnar
    mcar_test_result = little_mcar_test(df[column])
    if mcar_test_result > 0.05:
        return "MCAR"  # Fail to reject null hypothesis (MCAR)
    else:
        dtype = df[column].dtype
        if dtype == 'object' or dtype == 'category':
            # Perform additional tests for MNAR or MAR for categorical variables
            # You can consider methods like bootstrapping or hypothesis testing
            # For simplicity, assuming MNAR if the column is categorical
            return "MNAR"
        elif dtype in ['int64', 'float64']:
            # For numerical variables, more sophisticated tests can be applied
            if target_column:
                # Test for missingness related to target
                contingency_table = pd.crosstab(df[target_column].isnull(), df[column].notnull())
                chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                if p_value < 0.05:
                    return "MNAR"
            else:
                # For simplicity, assuming MAR if the column is numerical
                return "MAR"
        else:
            # If the data type is not numeric or categorical, handle it accordingly
            return "MAR"  # Assumption: Treat other data types as MAR

def select_relevant_features(df, target_column):
    # mutual information, recursive feature elimination, boruta, Shapley or model based selection
    # Multicollinearity (was this just for linear models? ?)
    # Adapt it based on the objective model
    # nonlinear feature techniques
    # feedback based on model performance 
    correlations = df.corr()[target_column].abs().sort_values(ascending=False)
    top_features = correlations.index[1:5] 
    return top_features

def test_imputation_strategies(df, column_name, target_variable, strategies):
    numerical_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    encoder = OrdinalEncoder()
    df[categorical_columns] = encoder.fit_transform(df[categorical_columns])

    X = df.drop(columns=[target_variable])  # Drop the target variable column
    X = df[numerical_columns.union(categorical_columns)]    
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_score = float('inf')  # Assuming a lower score is better; adjust based on your evaluation metric.
    best_strategy = None

    for strategy in strategies:
        imputer = strategy['imputer']
        X_train_imputed = X_train.copy()
        X_test_imputed = X_test.copy()

        
        if imputer != "StratifiedImputer":

            X_train_imputed[column_name] = imputer.fit_transform(X_train[[column_name]])
            X_test_imputed[column_name] = imputer.transform(X_test[[column_name]])
        else:
            X_train_imputed[column_name] = apply_stratified_imputation(df, column_name, target_variable)
            X_test_imputed[column_name] = apply_stratified_imputation(df, column_name, target_variable)
        # Assuming a regression problem; replace with appropriate model and metrics for classification.
        # Allow the user to cross_validate against several model by sending the target outcomet for now just regression or classification
        model = RandomForestRegressor(random_state=42) if df[target_variable].dtype == 'float' else RandomForestClassifier(random_state=42)
        model.fit(X_train_imputed, y_train)
        predictions = model.predict(X_test_imputed)
        
        score = mean_squared_error(y_test, predictions) if df[target_variable].dtype == 'float' else accuracy_score(y_test, predictions)

        if score < best_score:
            best_score = score
            best_strategy = strategy['name']

    return best_strategy, best_score

def assess_distribution(data):
    """
    Assess the distribution characteristics of a dataset, handling both univariate
    and multivariate data.

    Parameters:
    - data: pandas Series (for univariate) or DataFrame (for multivariate) containing the dataset to be analyzed.

    Returns:
    - A dictionary with distribution characteristics.
    """
    distribution_characteristics = {}

    if isinstance(data, pd.Series):
        column = data.dropna()  # Ensure no NaN values
        distribution_characteristics['skewness'] = skew(column)
        distribution_characteristics['kurtosis'] = kurtosis(column)

        _, p_value_normality_shapiro = shapiro(column.sample(min(1000, len(column))))
        distribution_characteristics['p_value_normality_shapiro'] = p_value_normality_shapiro

        result_anderson = anderson(column)
        distribution_characteristics['anderson'] = {
            'statistic': result_anderson.statistic,
            'critical_values': result_anderson.critical_values,
            'significance_level': result_anderson.significance_level
        }

        column_iqr = iqr(column)
        outliers = len([x for x in column if x < np.percentile(column, 25) - 1.5 * column_iqr or x > np.percentile(column, 75) + 1.5 * column_iqr])
        distribution_characteristics['outlier_ratio'] = outliers / len(column)

        _, p_value_levene = levene(column, column)
        distribution_characteristics['p_value_levene'] = p_value_levene

        _, p_value_ks = kstest((column - np.mean(column)) / np.std(column, ddof=1), 'norm')
        distribution_characteristics['p_value_ks'] = p_value_ks

        distribution_characteristics['multimodal'] = len(find_peaks(column)[0]) > 1

        _, p_exponential = kstest(column, 'expon', args=(column.min(), column.mean()-column.min()))
        distribution_characteristics['exponential'] = {'p_value': p_exponential}
        
        _, p_uniform = kstest(column, 'uniform', args=(column.min(), column.max()-column.min()))
        distribution_characteristics['uniform'] = {'p_value': p_uniform}
        
    elif isinstance(data, pd.DataFrame):
        data_clean = data.dropna()
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)

        pca = PCA(n_components=min(data_scaled.shape))
        pca.fit(data_scaled)
        distribution_characteristics['pca_explained_variance_ratio'] = pca.explained_variance_ratio_

        try:
            inv_cov_matrix = np.linalg.inv(np.cov(data_scaled, rowvar=False))
            mean_dists = np.mean(data_scaled, axis=0)
            mahalanobis_distances = pd.Series([np.sqrt((x-mean_dists) @ inv_cov_matrix @ (x-mean_dists).T) for x in data_scaled])
            outlier_threshold = np.mean(mahalanobis_distances) + 2 * np.std(mahalanobis_distances)
            distribution_characteristics['mahalanobis_outlier_ratio'] = (mahalanobis_distances > outlier_threshold).mean()
        except np.linalg.LinAlgError:
            distribution_characteristics['mahalanobis_outlier_ratio'] = 'Covariance matrix not invertible'
    else:
        raise ValueError("data must be a pandas Series or DataFrame")

    return distribution_characteristics

def apply_stratified_imputation(X, column_name, target_var):
    imputed_frames = []
    
    for category in X[target_var].unique():
        # Subset of the data for the current category
        subset = X.loc[X[target_var] == category, column_name]
        
        # Assuming 'numeric_imputation_strategy' returns a list of strategies and 'test_imputation_strategies'
        # tests these strategies and returns the name of the best strategy and its score
        strategies = numeric_imputation_strategy(subset)
        best_st, score = test_imputation_strategies(X, column_name, target_var, strategies)
        
        # Find the imputer for the best strategy
        imputer = [s['imputer'] for s in strategies if s['name'] == best_st][0]
        
        # Apply the imputation and get a series with the same index as the subset
        imputed_series = pd.Series(imputer.fit_transform(subset.values.reshape(-1, 1)).flatten(), index=subset.index)
        
        # Append the imputed series to the list of frames
        imputed_frames.append(imputed_series)
    
    # Concatenate all the imputed series along the index to form a single series
    imputed_data = pd.concat(imputed_frames, axis=0).sort_index()

    return imputed_data

def numeric_imputation_strategy(distribution, missingness_size, missingness_related_to_target, missingness_interrelated, missingness_type):
    strategies = []

    # Data characteristics
    is_skewed = abs(distribution['skewness']) > 0.5
    is_not_normal_shapiro = distribution['p_value_normality_shapiro'] < 0.05
    is_not_normal_ks = distribution['p_value_ks'] < 0.05
    has_outliers = distribution['outlier_ratio'] > 0.05
    high_kurtosis = abs(distribution['kurtosis']) > 3
    very_large_missingness = missingness_size >= 0.3
    variance_inhomogeneity = distribution['p_value_levene'] < 0.05
          
    # 3 misssingness analysis
    # missingness_interrelated - misssingness is related to other cateorical or numerical column or many other columns toether i.e analyze_missingness_interrelated()
    # missingness_type - missingness in the column alone by itself, i.e 'MCAR', 'MNAR'
    # missingness_related_to_target - misssingness is related to target var 
    
    if not (is_skewed or has_outliers or high_kurtosis or variance_inhomogeneity or missingness_related_to_target):
        dist_type = 'median' if distribution['is_skewed'] else 'mean'
        strategies.append({'name': dist_type, 'imputer': SimpleImputer(strategy=dist_type)})

    if is_skewed or has_outliers or variance_inhomogeneity or missingness_related_to_target:
        strategies.append({'name': 'knn', 'imputer': KNNImputer(n_neighbors=5)})

    models = []
    if very_large_missingness:
        if is_not_normal_shapiro or is_not_normal_ks or variance_inhomogeneity or missingness_related_to_target:
            models += [
                (make_pipeline(PolynomialFeatures(degree=3), Ridge()), 'ridge_regression_poly'),
            ]
        else:
            models += [
                (GradientBoostingRegressor(n_estimators=100, random_state=42), 'gradient_boosting'),
                (ElasticNet(alpha=0.01), 'elastic_net'),
                (Lasso(alpha=0.01), 'lasso_regression')
            ]
    else:
        if is_not_normal_shapiro or is_not_normal_ks or variance_inhomogeneity or missingness_related_to_target:
            models += [
                (Ridge(), 'ridge_regression'),
            ]
        else:
            models += [
                (GradientBoostingRegressor(n_estimators=100, random_state=42), 'gradient_boosting_moderate_missing'),
                (ElasticNet(alpha=0.01), 'elastic_net_moderate_missing'),
            ]
    
    for model, name in models:
        imputer = IterativeImputer(
            estimator=model,
            initial_strategy='median' if is_skewed or has_outliers or variance_inhomogeneity else 'mean',
            random_state=42
        )
        strategies.append({'name': name, 'imputer': imputer})

    if missingness_interrelated:
        # Based on feature importance do feature selection backward and forward 
        # Allow here to provide functions that allow for feature engineering... like a dict or something
        # that based on column name allows me to add other columns
        multivariate_imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, random_state=42), 
                                                initial_strategy='median', imputation_order='ascending', random_state=42)
        strategies.append({'name': 'multivariate', 'imputer': multivariate_imputer})
        # ensembled model of the other columns or gmm or stratified based on groups of the other columns 
    if missingness_related_to_target["is"] and missingness_related_to_target["target_var"] == "categorical":
        strategies.append({'name': 'stratified', 'imputer': 'StratifiedImputer()'})
    return strategies


def categorical_imputation_strategy(distribution, missingness_size, missingness_related_to_target, missingness_interrelated, missingness_type):
    strategies = []
    if missingness_size < 0.2 or missingness_type == 'MCAR':
        strategies.append({'name': 'most_frequent', 'imputer': SimpleImputer(strategy='most_frequent')})
    elif missingness_type in ['MAR', 'MNAR'] or missingness_size >= 0.2:
        strategies.append({'name': 'predictive', 'imputer': 'PredictiveImputer'})  # Placeholder for actual implementation
    else:
        strategies.append({'name': 'advanced_custom', 'imputer': 'AdvancedCustomImputer'})  # Placeholder for actual implementation
    
    return strategies

def define_imputation_strategies(dtype, distribution, missingness_size, missingness_related_to_target, missingness_interrelated, missingness_type):
    if dtype in ['int64', 'float64']:
        return numeric_imputation_strategy(distribution, missingness_size, missingness_related_to_target, missingness_interrelated, missingness_type)
    else:
        return categorical_imputation_strategy(distribution, missingness_size, missingness_related_to_target,  missingness_interrelated, missingness_type)

def analyze_missingness_interrelated(data, feature_to_analyze):
    results = {}
    
    # Create missingness indicator for the feature_to_analyze
    missing_indicator = data[feature_to_analyze].isnull().astype(int)
    
    # Initialize OneHotEncoder and StandardScaler for preprocessing
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    scaler = StandardScaler()
    
    # Separate features into categorical and numerical
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.drop(feature_to_analyze)
    
    # Categorical variables analysis
    for feature in categorical_cols:
        if feature != feature_to_analyze:
            contingency_table = pd.crosstab(index=missing_indicator, columns=data[feature])
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            results[feature] = {'test': 'chi2', 'p_value': p_value}
    
    # Logistic Regression for categorical predictors
    X_cat = ohe.fit_transform(data[categorical_cols].fillna('missing'))
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    try:
        log_reg.fit(X_cat, missing_indicator)
        categorical_predictor_importance = log_reg.coef_[0]
        for idx, cat_feature in enumerate(ohe.get_feature_names_out()):
            results[cat_feature] = {'logistic_regression_coefficient': categorical_predictor_importance[idx]}
    except Exception as e:
        results['logistic_regression_error'] = str(e)

    # Numerical variables analysis
    for feature in numerical_cols:
        if feature != feature_to_analyze:
            correlation, p_value = pearsonr(missing_indicator, data[feature].fillna(data[feature].mean()))
            results[feature] = {'test': 'pearson', 'p_value': p_value}

            group_missing = data.loc[missing_indicator == 1, feature].dropna()
            group_not_missing = data.loc[missing_indicator == 0, feature].dropna()
            f_value, p_value_anova = f_oneway(group_missing, group_not_missing)
            results[feature].update({'f_value': f_value, 'p_value_anova': p_value_anova})
    
    # MANOVA for numerical variables
    try:
        manova_data = data.copy()
        manova_data['missing_group'] = missing_indicator
        formula = ' + '.join(numerical_cols) + ' ~ missing_group'
        maov = MANOVA.from_formula(formula, data=manova_data.dropna(subset=numerical_cols))
        manova_results = maov.mv_test()
        results['manova'] = {'summary': str(manova_results)}
    except Exception as e:
        results['manova_error'] = str(e)

    # Feature importance analysis using RandomForest
    X_num = scaler.fit_transform(data[numerical_cols].fillna(0))
    X_preprocessed = np.hstack((X_cat, X_num))
    y = missing_indicator
    
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    feature_names = ohe.get_feature_names_out(categorical_cols).tolist() + numerical_cols.tolist()
    feature_importances = clf.feature_importances_
    for name, importance in zip(feature_names, feature_importances):
        if name in results:
            results[name]['feature_importance'] = importance
        else:
            results[name] = {'feature_importance': importance}
    
    return results
  
def analyze_missingness_related_to_target(dtype, data, feature_to_analyze, target_column):
    if dtype== 'object':
        missing_counts = data[data[feature_to_analyze].isnull()].groupby(target_column).size()
        total_counts = data.groupby(target_column).size()
        missing_ratio_per_category = missing_counts / total_counts
        return {
            'is': True if missing_ratio_per_category.var() > 0.05 else False,
            'target_var': 'categorical',
            'details': missing_ratio_per_category.to_dict()
        }
    else:
        return {
            'is': False,
            'target_var': 'continuous',
            'details': {}
        }

               
def automated_imputation_strategy(df, column_name, strategy=None, custom_strategy=None, target_variable=None):
    # Adapt it based on the objective model
   
    column = df[column_name]
    
    if not column.isna().any():
        return column
            
    # Imrpove the Outlier Detection -> IQR, Z-score, or Isolation Forests based filtering or Winsorization or Grubbs' test or DixonsQ est or DBSCAN
    # Use the technique to alter the imputation and transformation
    # Dynamic Imputation based on data patterns: Use model based imputation for numerical data, add logic to choose between mean, median, or mode based on distribution.
    # KNN, MICE, MissForest, bayesian
    # Clustering-Based Imputation - missing values based on cluster centroids
    # Feature Engineering for Imputation
    # group-wise or row-wise imputation
    # Use dtype, msssinnes type, missingesss size domain and best fit distribution to alter the strategy and warnings
    # Dynamically alter the percent based on domain knowledge for when to use a model imputation
    # Data Distribution Analysis Enhancements - Kolmogorov-Smirnov Test, Anderson-Darling Test
    # Model based imputation autoencoder use feature selection and feature enineering...
    # PROMPT - using all your machine learning and most importantly stat knowledge improve this function... consider that it might run both before and after preprocessing and cleanin ...  and that data might not be normal nor linear 
    # Mahalanobis distance, Jarque-Bera test or the D'Agostino-Pearson
    # gans, vaes, amelia imputation, locf, interpolation, decomposition, stl 
    # ensemble imputation -stacking, blending weighted av, 
    # bootstrapping montecarlo. ffor confidence intervals
    # How can I make this more robust ?? like maybe make it test alone against targget but also emsembled with other columns already cleaned or not cleaned
    # cluster based imputation.... like do groups randomly with vars and fill
    # Gaussian Mixture Models  gmm, ensembled models, stacking, 
    
    dtype = column.dtype
    
    if dtype in ['int64', 'float64']:     
        unexpected_elements = column[~column.isna() & ~column.astype(str).str.isnumeric() & ~column.astype(str).str.match(r'^\d+?\.\d+?$')]
        if not unexpected_elements.empty:
            raise ValueError("Column '{}' contains unexpected elements: {}".format(column_name, unexpected_elements.unique()))   
        
    missingness_size = column.isnull().mean()
    missingness_type = detect_missingness_pattern(df, column_name)
    domain = "" #identify_domain(column)
    distribution = assess_distribution(column) # make this mvar
    missingness_interrelated= analyze_missingness_interrelated(df, column_name)
    missingness_related_to_target=  analyze_missingness_related_to_target(dtype, df, column_name, target_variable)
     
    strategies = define_imputation_strategies(dtype, distribution, missingness_size, missingness_related_to_target, missingness_interrelated, missingness_type)
    if custom_strategy:
        strategies.append({'name': 'custom', 'imputer': custom_strategy(df, column_name)})
    if strategy in ['mean', 'median', 'most_frequent', 'constant'] :
        imputer = SimpleImputer(strategy=strategy)  
    else:
        imputer = KNNImputer(n_neighbors=5)
    strategies.append({'name': 'strategy', 'imputer': imputer})

    best_strategy, best_score = test_imputation_strategies(df, column_name, target_variable, strategies)            
    imputer = [s['imputer'] for s in strategies if s['name'] == best_strategy][0]
    imputed_series = pd.Series(imputer.fit_transform(column.values.reshape(-1, 1)).flatten(), index=column.index)

    if imputed_series.isnull().any():
        raise ValueError("Imputed series contains NaN values for column: {}".format(column_name))
            
    return imputed_series

def clean_categorical_column(column):
    # Adapt it based on the objective model
    
    modified_column = column.copy()  
    if column.dtype.name != 'category':
        return column, {}
    modified_column = column.astype(str)
    unique_values = modified_column.unique()
    word_pairs = [(value, process.extractOne(value, unique_values)) for value in unique_values]
    suggestions = {pair[0]: pair[1][0] for pair in word_pairs if pair[1][0] != pair[0]}
    return column, suggestions

def clean_numerical_column(column):
    # Adapt it based on the objective model
    return column

def preprocess_and_transform_data(df, target_variable):
    # Adapt it based on the objective model
    # allow to alter the order of the stages to evaluate against the objective model
    
    transformations = {}

    # Parallel Processing,  to process several columns same time
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_columns:
        df[column] = clean_numerical_column(df[column])
        imputed_column = automated_imputation_strategy(df, column, None, None,  target_variable)

        transformations[column] = evaluate_numerical_transformations(imputed_column)
        
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns # Consider binary data 
    for column in categorical_columns:
        cleaned_column, suggestions = clean_categorical_column(df[column])
        df[column] =  cleaned_column
        imputed_column = automated_imputation_strategy(df, column, None, None,  target_variable)

        transformations[column] = evaluate_categorical_transformations(imputed_column, column, suggestions)
    
    # clean()
        # stripping whitespace OR converting data types OR handling known bad values OR handle typos OR inconsistent capitalization
        
    # numeric_feature_engineering()
        # feature_interactions        
            
    # categry_feature_engineering()
        # rare category clustering
        # feature_interactions - Non-Linearity
        # MDS
        # Models to do the categorization maybe GEMINI like give the list and column name and ask to suggest categories then just add them  https://www.kdnuggets.com/how-to-access-and-use-gemini-api-for-free
        # This could generate new features based on existing categories... interactions, counts, etc
        # suggest_complementary_datasets()

    # balanced_imabalanced_data()
        # resampling
        # Class Weighting
        # syntethic data
        # ensemble methods
        
    #mix_feature_engineering()
        # feature_interactions    
        # gemini suggest new features
        # Adaptive Binning 
        # Temporal Features Extraction
        # time series decomposition for forecasting, interaction terms for regression
        # Wavelet Transforms for Time-Series Data
        # Filters for Noise Reduction
        
    for column, details in transformations.items():
        print(f"{column}: {details['transformation']} - {details['reason']} - {details.get('suggestions')}")
        
    return transformations

# create a config file.. or make everything more personalizable based on the objective to alter my existing code
# Add a way I can change all this based on performance (cross validation?? towards target, several metrics or stats tests based on objective model or model(s))

# automate_feature_selection()
    # BEFORE/AFTER transformations and imputations
    # permutation
    # feature_analysis()
    # statistical_test
    # polynomials
    # multivar tests
    # dimensionality reduction -  PCA, t-SNE, UMAP,
    # Recursive Feature Elimination with Cross-Validation (RFECV)

# automate_model_selection
# optimize_model() - Kernels 
# Underfiiting, overfitting
# Uncertainty Estimation
# Dynamic Adjustment of Model Metrics like ROC AUC
# model ensembking with altered origin pipeline or just different model from the same pipeline
#   Applies a range of preprocessing steps.
#   Trains different models.
#   Evaluates performance using appropriate metrics.
#   Selects the best preprocessing-model combination.
# Sensitivity and Comparative Analysis
# Pipeline and hyperparameter optimization tools (GridSearchCV or RandomizedSearchCV) to fine tune the selection of transformations and the parameters of imputation models.
# Would it be easier to apply multi heuristics ?
# validate_data against each model needs
# validate results against each model tests
# Louvain or Girvan-Newman for detecting communities within the data
# Spectral Clustering
# GNNs, GSP
# Space-Time Cube Representation:
# KNN for Spatio-Temporal Data
# Inverse Distance Weighting (IDW) for Spatial Interpolation:
# Dynamic Time Warping (DTW
# Kalman filters
# Sensor Fusion
# Spatial Kriging
# integrate_multimodal_dat images -> table or something
# generate_spatiotemporal_data_cGAN
# Neural Architecture Search (NAS)
# Causal Discovery

# optimize the pipeline based on model selection (backwrards)
# tell what the data works for  multebranching of the pipepline aka tell the model that could work  (forward)
# store the models if used in each staged so that each can be later optimized... and built into a slimmer pipeline

# preprocess_and_transform_data(train_data)

# objective_type = input("Enter ML Objective (Regression, Classification...): ")
# category = input("Enter Category (Binary_Classification, Multiclass_Classification...): ")
# target_type = input("Select target type (Models_1_target, ...): ")

# for text data that has more than 1 word cluster it based on the word or words most repited... or title groupings
  # Mr Armando, Sr Armando....  Mr and SR are repeated words.... this can be a ood mining for encoding or rfeature for text columns
# for ordered numeric data test various ranges clustering... and keep the most wondering... 


Primer modelo: realizar tu análisis con todos los datos.
• Segundo modelo: Primero borrar los outliers, luego transformarlos y luego generar tu modelo.
• Tercer modelo: Primero transformarlos con todo y outliers, luego borrar los que queden y luego generar tu modelo.
