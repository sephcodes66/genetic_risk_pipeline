import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import joblib
from sklearn.metrics import roc_curve, auc

def plot_histograms(df, output_dir):
    """
    Generates histograms for a subset of numeric variables in the dataset.
    """
    logger.info("Generating histograms...")
    # "The user wants fewer plots, so I'm just picking a few representative columns."
    cols_to_plot = ['rs1', 'rs50', 'rs100', 'age', 'adhd_grs', 'asd_grs', 'prs']
    for col in cols_to_plot:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(f'{output_dir}/{col}_histogram.png')
            plt.close()

def plot_correlation_matrix(df, output_dir):
    """
    Generates a correlation matrix heatmap.
    """
    logger.info("Generating correlation matrix...")
    plt.figure(figsize=(20, 15))
    # "I had some issues with the labels being too small, so I'm using annot=False.
    # For a smaller number of variables, annot=True would be better."
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(f'{output_dir}/correlation_matrix.png')
    plt.close()

def plot_boxplots(df, output_dir):
    """
    Generates boxplots for outlier detection for a subset of columns.
    """
    logger.info("Generating boxplots for outlier detection...")
    cols_to_plot = ['rs1', 'rs50', 'rs100', 'age', 'adhd_grs', 'asd_grs', 'prs']
    for col in cols_to_plot:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
            plt.savefig(f'{output_dir}/{col}_boxplot.png')
            plt.close()

def plot_age_distribution_comparison(df, output_dir):
    """
    Compares the age distribution of the synthetic data with a normal distribution.
    """
    logger.info("Generating age distribution comparison plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], kde=True, label='Synthetic Data', stat='density')
    # "I'm assuming a normal distribution for the 'real' data. This is a simplification,
    # but it's a good way to check if my synthetic data is at least plausible."
    mu, sigma = df['age'].mean(), df['age'].std()
    s = np.random.normal(mu, sigma, 1000)
    sns.histplot(s, kde=True, label='Normal Distribution', color='red', stat='density', alpha=0.6)
    plt.title('Age Distribution Comparison')
    plt.legend()
    plt.savefig(f'{output_dir}/age_distribution_comparison.png')
    plt.close()

def plot_feature_importance_comparison(models, features, output_dir):
    """
    Generates a comparison of feature importances for all models.
    """
    logger.info("Generating feature importance comparison plot...")
    fig, axes = plt.subplots(1, len(models), figsize=(20, 6), sharey=True)
    fig.suptitle('Feature Importance Comparison')

    for i, (name, model) in enumerate(models.items()):
        if hasattr(model, 'coef_'):
            importance = model.coef_[0]
        else:
            importance = model.feature_importances_
        
        feature_importance = pd.DataFrame({'feature': features, 'importance': importance})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        sns.barplot(ax=axes[i], x='importance', y='feature', data=feature_importance)
        axes[i].set_title(name)
    
    plt.savefig(f'{output_dir}/feature_importance_comparison.png')
    plt.close()

def plot_roc_curve_comparison(models, X_test, y_test, output_dir):
    """
    Generates a ROC curve comparison plot for all models.
    """
    logger.info("Generating ROC curve comparison plot...")
    plt.figure(figsize=(10, 6))

    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/roc_curve_comparison.png')
    plt.close()

def plot_model_performance_comparison(output_dir):
    """
    Generates a grouped bar chart comparing model performance metrics.
    """
    logger.info("Generating model performance comparison plot...")
    try:
        df = pd.read_csv('results/tables/model_comparison.csv', index_col=0)
    except FileNotFoundError:
        logger.warning("model_comparison.csv not found. Skipping performance plot.")
        return

    # "This is a bit of a hack to get the data in the right format for a grouped bar chart.
    # There's probably a more elegant way to do this with pandas."
    df_plot = df[['accuracy', 'precision_class_1', 'recall_class_1', 'f1_score_class_1']].copy()
    df_plot.rename(columns={
        'precision_class_1': 'precision',
        'recall_class_1': 'recall',
        'f1_score_class_1': 'f1-score'
    }, inplace=True)
    df_plot = df_plot.reset_index()
    df_plot = df_plot.melt(id_vars='index', var_name='metric', value_name='score')
    df_plot.rename(columns={'index': 'model'}, inplace=True)


    plt.figure(figsize=(12, 7))
    sns.barplot(x='model', y='score', hue='metric', data=df_plot)
    plt.title('Model Performance Comparison (Class 1)')
    plt.xticks(rotation=15)
    plt.ylim(0, 1)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_performance_comparison.png')
    plt.close()

def main():
    """
    Main function to generate visualizations.
    """
    # --- Configuration ---
    PROCESSED_DATA_PATH = 'data/processed/processed_data.csv'
    FIGURES_DIR = 'results/figures'
    MODEL_NAMES = ['logistic_regression', 'random_forest', 'gradient_boosting']
    FEATURES = ['prs', 'age', 'sex']

    # --- Logging Setup ---
    logger.add("results/tables/visualization.log", rotation="500 MB", mode="w")

    # --- Load Data ---
    logger.info("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # --- Load Models ---
    logger.info("Loading trained models...")
    models = {}
    for name in MODEL_NAMES:
        model_path = f'results/tables/{name}_model.joblib'
        try:
            models[name.replace("_", " ").title()] = joblib.load(model_path)
        except FileNotFoundError:
            logger.warning(f"Model file not found: {model_path}")
            continue

    # --- Prepare Data for Plots ---
    X = df[FEATURES]
    X['sex'] = X['sex'].apply(lambda x: 1 if x == 'M' else 0)
    y = df['adhd_diagnosis']
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Generate Plots ---
    plot_histograms(df, FIGURES_DIR)
    plot_correlation_matrix(df, FIGURES_DIR)
    plot_boxplots(df, FIGURES_DIR)
    plot_age_distribution_comparison(df, FIGURES_DIR)
    plot_feature_importance_comparison(models, FEATURES, FIGURES_DIR)
    plot_roc_curve_comparison(models, X_test, y_test, FIGURES_DIR)
    plot_model_performance_comparison(FIGURES_DIR)

    logger.info("Visualizations generated.")

if __name__ == '__main__':
    main()