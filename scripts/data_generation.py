import numpy as np
import pandas as pd
from scipy.stats import bernoulli

def generate_genetic_data(n_samples, n_snps, allele_freq_range=(0.05, 0.5)):
    """
    Generates a synthetic genetic dataset (genotypes).

    This is a simplified model of genetic data. In reality, SNP frequencies
    and linkage disequilibrium are much more complex. I'm assuming SNPs are
    independent for now.

    Args:
        n_samples (int): Number of individuals in the dataset.
        n_snps (int): Number of single nucleotide polymorphisms (SNPs).
        allele_freq_range (tuple): Range of minor allele frequencies.

    Returns:
        pandas.DataFrame: A DataFrame of genotypes, where each row is a sample
                          and each column is a SNP. Genotypes are encoded as
                          0, 1, or 2 (number of minor alleles).
    """
    # "I spent a while trying to figure out the best way to model this.
    # Decided to keep it simple for now and assume Hardy-Weinberg equilibrium."
    genotypes = np.zeros((n_samples, n_snps), dtype=np.int8)
    for i in range(n_snps):
        # Generate a random minor allele frequency for each SNP
        maf = np.random.uniform(*allele_freq_range)
        # Under HWE, genotype frequencies are p^2, 2pq, q^2
        p = 1 - maf
        q = maf
        genotype_freqs = [p**2, 2*p*q, q**2]
        # Generate genotypes for all samples for this SNP
        genotypes[:, i] = np.random.choice([0, 1, 2], size=n_samples, p=genotype_freqs)
    return pd.DataFrame(genotypes, columns=[f'rs{i+1}' for i in range(n_snps)])

def generate_clinical_data(n_samples, genetic_data, adhd_risk_snps, asd_risk_snps):
    """
    Generates synthetic clinical data (phenotypes and covariates).

    This is where things get a bit hand-wavy. The relationship between
    genotype and phenotype is complex and polygenic. I'm modeling it as a
    simple linear combination of a few risk SNPs, plus some noise.

    Args:
        n_samples (int): Number of individuals.
        genetic_data (pandas.DataFrame): DataFrame of genotypes.
        adhd_risk_snps (list): List of SNP names that increase risk for ADHD.
        asd_risk_snps (list): List of SNP names that increase risk for ASD.

    Returns:
        pandas.DataFrame: A DataFrame of clinical data.
    """
    # "TODO: This is a super simplified model of disease risk.
    # Should probably incorporate some non-linear effects and gene-gene interactions."
    age = np.random.randint(5, 18, size=n_samples)
    sex = np.random.choice(['M', 'F'], size=n_samples, p=[0.5, 0.5])

    # Calculate a "genetic risk score" for each disorder
    adhd_grs = genetic_data[adhd_risk_snps].sum(axis=1)
    asd_grs = genetic_data[asd_risk_snps].sum(axis=1)

    # Simulate diagnosis based on the genetic risk score and some noise
    # The coefficients and intercept are arbitrary, just to get a reasonable prevalence
    adhd_prob = 1 / (1 + np.exp(-(0.5 * adhd_grs - 3 + np.random.normal(0, 1, n_samples))))
    asd_prob = 1 / (1 + np.exp(-(0.6 * asd_grs - 4 + np.random.normal(0, 1, n_samples))))

    adhd_diagnosis = bernoulli.rvs(adhd_prob)
    asd_diagnosis = bernoulli.rvs(asd_prob)

    # Create a dataframe
    clinical_df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'adhd_diagnosis': adhd_diagnosis,
        'asd_diagnosis': asd_diagnosis,
        'adhd_grs': adhd_grs,
        'asd_grs': asd_grs
    })

    return clinical_df

def generate_intervention_data(clinical_data):
    """
    Generates synthetic intervention response data.

    This is another simplification. Intervention response is influenced by many
    factors. I'm modeling it as being dependent on age, sex, and diagnosis.

    Args:
        clinical_data (pandas.DataFrame): DataFrame of clinical data.

    Returns:
        pandas.DataFrame: A DataFrame of intervention response data.
    """
    # "This part is tricky. Intervention response is a whole field of study.
    # I'm just making some plausible assumptions here."
    intervention_response = np.zeros(len(clinical_data))
    for i, row in clinical_data.iterrows():
        response_prob = 0.5  # Baseline response
        if row['adhd_diagnosis'] == 1:
            response_prob += 0.2
        if row['asd_diagnosis'] == 1:
            response_prob -= 0.1
        if row['age'] < 10:
            response_prob += 0.1
        if row['sex'] == 'F':
            response_prob += 0.05
        # Add some noise
        response_prob += np.random.normal(0, 0.1)
        response_prob = np.clip(response_prob, 0, 1)
        intervention_response[i] = np.random.choice([0, 1], p=[1-response_prob, response_prob])

    intervention_df = pd.DataFrame({'intervention_response': intervention_response})
    return intervention_df

def introduce_missing_data(df, missing_fraction=0.05):
    """
    Introduces missing data into a DataFrame.

    Because real-world data is never perfect.

    Args:
        df (pandas.DataFrame): The DataFrame to introduce missing data into.
        missing_fraction (float): The fraction of data to be replaced with NaN.

    Returns:
        pandas.DataFrame: The DataFrame with missing data.
    """
    df_missing = df.copy()
    for col in df_missing.columns:
        # Don't introduce missing data in the ID column
        if col == 'sample_id':
            continue
        mask = np.random.rand(len(df_missing)) < missing_fraction
        df_missing.loc[mask, col] = np.nan
    return df_missing

if __name__ == '__main__':
    # --- Configuration ---
    N_SAMPLES = 1000
    N_SNPS = 100
    ADHD_RISK_SNPS = [f'rs{i+1}' for i in range(5)]
    ASD_RISK_SNPS = [f'rs{i+1}' for i in range(5, 10)]
    MISSING_FRACTION = 0.05

    # --- Data Generation ---
    print("Generating synthetic data...")
    # "I'm setting a seed here to make the data generation reproducible.
    # This is important for debugging and for anyone else who wants to run this."
    np.random.seed(42)

    # Generate genetic data
    genetic_data = generate_genetic_data(N_SAMPLES, N_SNPS)
    genetic_data['sample_id'] = [f'sample_{i+1}' for i in range(N_SAMPLES)]
    genetic_data = genetic_data[['sample_id'] + [col for col in genetic_data.columns if col != 'sample_id']]

    # Generate clinical data
    clinical_data = generate_clinical_data(N_SAMPLES, genetic_data, ADHD_RISK_SNPS, ASD_RISK_SNPS)
    clinical_data['sample_id'] = [f'sample_{i+1}' for i in range(N_SAMPLES)]
    clinical_data = clinical_data[['sample_id'] + [col for col in clinical_data.columns if col != 'sample_id']]

    # Generate intervention data
    intervention_data = generate_intervention_data(clinical_data)
    intervention_data['sample_id'] = [f'sample_{i+1}' for i in range(N_SAMPLES)]
    intervention_data = intervention_data[['sample_id'] + [col for col in intervention_data.columns if col != 'sample_id']]

    # --- Introduce Missing Data ---
    print("Introducing missing data...")
    genetic_data_missing = introduce_missing_data(genetic_data, MISSING_FRACTION)
    clinical_data_missing = introduce_missing_data(clinical_data, MISSING_FRACTION)
    intervention_data_missing = introduce_missing_data(intervention_data, MISSING_FRACTION)

    # --- Save Data ---
    print("Saving data to data/raw/...")
    genetic_data_missing.to_csv('data/raw/genetic_data.csv', index=False)
    clinical_data_missing.to_csv('data/raw/clinical_data.csv', index=False)
    intervention_data_missing.to_csv('data/raw/intervention_data.csv', index=False)

    print("Data generation complete.")
