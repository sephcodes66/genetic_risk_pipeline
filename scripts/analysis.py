import pandas as pd
import numpy as np
from loguru import logger
import src.utils as utils
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from imblearn.over_sampling import SMOTE

def preprocess_data(genetic_data_path, clinical_data_path, intervention_data_path):
    """
    Loads and preprocesses the data.

    This function is a bit of a mess right now. It's doing too much.
    TODO: Refactor this into smaller, more manageable functions.
    """
    logger.info("Loading data...")
    genetic_df = pd.read_csv(genetic_data_path)
    clinical_df = pd.read_csv(clinical_data_path)
    intervention_df = pd.read_csv(intervention_data_path)

    logger.info("Merging dataframes...")
    df = pd.merge(genetic_df, clinical_df, on='sample_id')
    df = pd.merge(df, intervention_df, on='sample_id')

    logger.info("Handling missing data...")
    # This is a simple imputation strategy. Might need something more
    # sophisticated for a real-world dataset.
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    return df

def calculate_prs(df, snp_weights):
    """
    Calculates the Polygenic Risk Score (PRS).

    This is a simple weighted sum of the number of risk alleles.
    I found a good tutorial on this here: [link to a hypothetical tutorial]
    """
    logger.info("Calculating Polygenic Risk Score (PRS)...")
    prs = np.zeros(len(df))
    for snp, weight in snp_weights.items():
        if snp in df.columns:
            prs += df[snp] * weight
    return prs

def generate_summary_table(df):
    """
    Generates a summary table of the data.
    """
    logger.info("Generating data summary table...")
    summary = df.describe(include='all').transpose()
    summary['missing_values'] = df.isnull().sum()
    summary.to_csv('results/tables/data_summary.csv')

from scipy.stats import ttest_ind

def run_statistical_analysis(df):
    """
    Performs statistical analysis on the data.
    """
    logger.info("Performing statistical analysis...")
    adhd_cases = df[df['adhd_diagnosis'] == 1]['prs']
    adhd_controls = df[df['adhd_diagnosis'] == 0]['prs']
    ttest_result = ttest_ind(adhd_cases, adhd_controls)

    asd_cases = df[df['asd_diagnosis'] == 1]['prs']
    asd_controls = df[df['asd_diagnosis'] == 0]['prs']
    ttest_result_asd = ttest_ind(asd_cases, asd_controls)

    results_df = pd.DataFrame({
        'analysis': ['ADHD PRS T-test', 'ASD PRS T-test'],
        'statistic': [ttest_result.statistic, ttest_result_asd.statistic],
        'p_value': [ttest_result.pvalue, ttest_result_asd.pvalue]
    })
    results_df.to_csv('results/tables/statistical_analysis_results.csv', index=False)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

from imblearn.over_sampling import SMOTE

def train_ml_models(df):
    """
    Trains and compares multiple machine learning models using SMOTE to handle class imbalance.
    """
    logger.info("Training and comparing machine learning models with SMOTE...")
    features = ['prs', 'age', 'sex']
    X = df[features]
    y = df['adhd_diagnosis']

    X['sex'] = X['sex'].apply(lambda x: 1 if x == 'M' else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # "The class_weight='balanced' wasn't enough. Switching to SMOTE to see if it handles
    # the imbalance better. This is a common technique I've seen in a few papers."
    logger.info("Applying SMOTE to the training data...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    model_comparison = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        report_class_0 = report.get('0.0', {})
        report_class_1 = report.get('1.0', {})
        model_comparison[name] = {
            'accuracy': accuracy,
            'precision_class_0': report_class_0.get('precision', 0),
            'recall_class_0': report_class_0.get('recall', 0),
            'f1_score_class_0': report_class_0.get('f1-score', 0),
            'precision_class_1': report_class_1.get('precision', 0),
            'recall_class_1': report_class_1.get('recall', 0),
            'f1_score_class_1': report_class_1.get('f1-score', 0),
        }

        logger.info(f"Saving {name} model...")
        joblib.dump(model, f'results/tables/{name.lower().replace(" ", "_")}_model.joblib')

    comparison_df = pd.DataFrame(model_comparison).transpose()
    comparison_df.to_csv('results/tables/model_comparison.csv')

def main():
    """
    Main function to run the analysis pipeline.
    """
    # --- Configuration ---
    GENETIC_DATA_PATH = 'data/raw/genetic_data.csv'
    CLINICAL_DATA_PATH = 'data/raw/clinical_data.csv'
    INTERVENTION_DATA_PATH = 'data/raw/intervention_data.csv'
    # These weights are arbitrary for now. In a real analysis, they would
    # come from a GWAS study.
    SNP_WEIGHTS = {f'rs{i+1}': np.random.uniform(0, 0.1) for i in range(10)}

    # --- Logging Setup ---
    logger.add("results/tables/analysis.log", rotation="500 MB")

    # --- Data Preprocessing ---
    df = preprocess_data(GENETIC_DATA_PATH, CLINICAL_DATA_PATH, INTERVENTION_DATA_PATH)

    # --- PRS Calculation ---
    df['prs'] = calculate_prs(df, SNP_WEIGHTS)

    # --- Generate Summary Table ---
    generate_summary_table(df)

    # --- Statistical Analysis ---
    run_statistical_analysis(df)

    # --- Machine Learning ---
    train_ml_models(df)

    # --- Save Processed Data ---
    logger.info("Saving processed data to data/processed/...")
    df.to_csv('data/processed/processed_data.csv', index=False)

    logger.info("Analysis complete.")

if __name__ == '__main__':
    main()
