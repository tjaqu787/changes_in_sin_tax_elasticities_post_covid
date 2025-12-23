"""
Canadian Alcohol Consumption - Decomposition Analysis
======================================================

Research Question: What is driving the decline in alcohol consumption (2015-2023)?

Approach: Regression-based decomposition
- Estimate: log(consumption) = β₀ + β₁·log(real_price) + β₂·log(real_income) + 
                               β₃·share_65+ + β₄·post_covid + β₅·year_trend + ε
- Decompose change into contributions from each factor
- Bootstrap standard errors for statistical inference
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

BASELINE_YEAR = 2015
COMPARISON_YEAR = 2023
N_BOOTSTRAP = 1000

# ==============================================================================
# 1. LOAD PREPARED DATA
# ==============================================================================

def load_data():
    """Load prepared analysis data"""
    print("Loading prepared data...")
    df = pd.read_csv(DATA_DIR / "analysis_data.csv")
    print(f"✓ Loaded {len(df)} observations")
    return df


# ==============================================================================
# 2. DECOMPOSITION MODEL SPECIFICATIONS
# ==============================================================================

class DecompositionModel:
    """
    Regression-based decomposition model
    
    Key features:
    - Log-log specification for elasticity interpretation
    - Real prices (deflated by overall CPI)
    - Real income (deflated by overall CPI)
    - Demographics (share 65+)
    - COVID shock (discrete dummy)
    - Time trend (smooth preference shifts)
    """
    
    def __init__(self, name, formula):
        self.name = name
        self.formula = formula
        self.model = None
        self.results = None
    
    def fit(self, data):
        """Fit the model"""
        # Parse formula and create design matrix
        y_var, x_vars = self._parse_formula(self.formula)
        
        # Prepare data
        df = data.dropna(subset=[y_var] + x_vars)
        
        y = df[y_var]
        X = df[x_vars]
        X = sm.add_constant(X)
        
        # Fit OLS
        self.model = OLS(y, X)
        self.results = self.model.fit()
        
        return self.results
    
    def _parse_formula(self, formula):
        """Parse formula string into y and X variables"""
        parts = formula.split('~')
        y_var = parts[0].strip()
        x_vars = [v.strip() for v in parts[1].split('+')]
        return y_var, x_vars
    
    def decompose(self, data, baseline_year, comparison_year):
        """
        Decompose the change in consumption
        
        Returns:
        - Total change
        - Contribution from each variable
        - Unexplained residual
        """
        # Get coefficients
        if self.results is None:
            raise ValueError("Model must be fitted first")
        
        params = self.results.params
        
        # Get baseline and comparison values
        baseline = data[data['year'] == baseline_year].iloc[0]
        comparison = data[data['year'] == comparison_year].iloc[0]
        
        # Calculate contributions
        contributions = {}
        
        y_var, x_vars = self._parse_formula(self.formula)
        
        for var in x_vars:
            if var in params.index:
                delta = comparison[var] - baseline[var]
                contribution = params[var] * delta
                contributions[var] = contribution
        
        # Total explained
        total_explained = sum(contributions.values())
        
        # Actual change
        actual_change = comparison[y_var.replace('log_', '')] - baseline[y_var.replace('log_', '')]
        
        # Unexplained (residual)
        unexplained = actual_change - total_explained
        
        return {
            'actual_change': actual_change,
            'contributions': contributions,
            'total_explained': total_explained,
            'unexplained': unexplained,
            'percent_explained': (total_explained / actual_change * 100) if actual_change != 0 else np.nan
        }


# Define model specifications
MODEL_SPECS = {
    'baseline': DecompositionModel(
        name='Baseline Model',
        formula='log_consumption_total ~ log_real_alcohol_price + log_real_income + share_65plus + post_covid + year_centered'
    ),
    
    'no_trend': DecompositionModel(
        name='No Time Trend',
        formula='log_consumption_total ~ log_real_alcohol_price + log_real_income + share_65plus + post_covid'
    ),
    
    'with_interactions': DecompositionModel(
        name='With Age Interactions',
        formula='log_consumption_total ~ log_real_alcohol_price + log_real_income + share_65plus + post_covid + year_centered'
        # Note: Interactions would be added as separate variables in data prep
    )
}


# ==============================================================================
# 3. BOOTSTRAP DECOMPOSITION
# ==============================================================================

def bootstrap_decomposition(data, geography, model_spec, baseline_year, comparison_year, n_boot=1000):
    """
    Bootstrap decomposition with standard errors
    
    Returns:
    - Point estimates for each contribution
    - Standard errors
    - Confidence intervals
    """
    print(f"  Running bootstrap for {geography} ({n_boot} iterations)...")
    
    # Filter data
    geo_data = data[data['Geography'] == geography].copy()
    
    # Initialize storage
    boot_results = []
    
    # Original decomposition
    model = MODEL_SPECS[model_spec]
    model.fit(geo_data)
    original = model.decompose(geo_data, baseline_year, comparison_year)
    
    # Bootstrap
    np.random.seed(42)
    for i in range(n_boot):
        # Resample with replacement
        boot_sample = geo_data.sample(n=len(geo_data), replace=True)
        
        try:
            # Fit model
            boot_model = DecompositionModel(model.name, model.formula)
            boot_model.fit(boot_sample)
            
            # Decompose
            boot_decomp = boot_model.decompose(boot_sample, baseline_year, comparison_year)
            boot_results.append(boot_decomp['contributions'])
            
        except Exception as e:
            continue
    
    # Calculate statistics
    if len(boot_results) == 0:
        print(f"    ⚠ Bootstrap failed for {geography}")
        return None
    
    # Convert to DataFrame
    boot_df = pd.DataFrame(boot_results)
    
    # Calculate point estimates, SEs, and CIs
    results = {
        'geography': geography,
        'actual_change': original['actual_change'],
        'contributions': {},
        'standard_errors': {},
        'confidence_intervals': {}
    }
    
    for var in boot_df.columns:
        results['contributions'][var] = original['contributions'][var]
        results['standard_errors'][var] = boot_df[var].std()
        results['confidence_intervals'][var] = (
            boot_df[var].quantile(0.025),
            boot_df[var].quantile(0.975)
        )
    
    results['total_explained'] = original['total_explained']
    results['unexplained'] = original['unexplained']
    results['percent_explained'] = original['percent_explained']
    
    return results


# ==============================================================================
# 4. RUN DECOMPOSITION ANALYSIS
# ==============================================================================

def run_decomposition_analysis(data):
    """
    Run full decomposition analysis for all geographies
    """
    print("\n" + "="*80)
    print("DECOMPOSITION ANALYSIS")
    print("="*80)
    
    geographies = data['Geography'].unique()
    all_results = {}
    
    for model_name, model in MODEL_SPECS.items():
        print(f"\n--- Model: {model.name} ---")
        all_results[model_name] = {}
        
        for geo in geographies:
            print(f"\nGeography: {geo}")
            
            # Bootstrap decomposition
            result = bootstrap_decomposition(
                data, geo, model_name, 
                BASELINE_YEAR, COMPARISON_YEAR, 
                n_boot=N_BOOTSTRAP
            )
            
            if result is not None:
                all_results[model_name][geo] = result
                
                # Print summary
                print(f"  Actual change: {result['actual_change']:.3f} L")
                print(f"  Total explained: {result['total_explained']:.3f} L ({result['percent_explained']:.1f}%)")
                print(f"  Unexplained: {result['unexplained']:.3f} L")
    
    return all_results


# ==============================================================================
# 5. CREATE RESULTS TABLES
# ==============================================================================

def create_results_table(results, model_name):
    """
    Create formatted results table
    """
    model_results = results[model_name]
    
    # Initialize DataFrame
    rows = []
    
    for geo, result in model_results.items():
        row = {
            'Geography': geo,
            'Actual Change': result['actual_change']
        }
        
        # Add contributions
        for var, contrib in result['contributions'].items():
            se = result['standard_errors'][var]
            ci = result['confidence_intervals'][var]
            
            # Format with significance stars
            t_stat = contrib / se if se > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=10))
            
            stars = ''
            if p_value < 0.001:
                stars = '***'
            elif p_value < 0.01:
                stars = '**'
            elif p_value < 0.05:
                stars = '*'
            elif p_value < 0.1:
                stars = '.'
            
            row[var] = f"{contrib:.3f}{stars} ({se:.3f})"
            row[f"{var}_ci"] = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        
        row['Total Explained'] = result['total_explained']
        row['Unexplained'] = result['unexplained']
        row['% Explained'] = result['percent_explained']
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# ==============================================================================
# 6. VISUALIZATION OF DECOMPOSITION
# ==============================================================================

def create_decomposition_chart(results, geography, model_name='baseline'):
    """
    Create waterfall chart showing decomposition
    
    Note: This prepares data for Plotly visualization
    Returns dictionary ready for Plotly
    """
    result = results[model_name][geography]
    
    # Prepare data for waterfall chart
    categories = []
    values = []
    
    # Start with baseline
    categories.append(f'Baseline ({BASELINE_YEAR})')
    baseline_value = result['actual_change']  # We'll work backwards
    values.append(baseline_value)
    
    # Add each contribution
    for var, contrib in result['contributions'].items():
        var_name = var.replace('_', ' ').title()
        categories.append(var_name)
        values.append(contrib)
    
    # Add unexplained
    categories.append('Unexplained')
    values.append(result['unexplained'])
    
    # End point
    categories.append(f'Actual ({COMPARISON_YEAR})')
    values.append(baseline_value + sum(result['contributions'].values()) + result['unexplained'])
    
    return {
        'categories': categories,
        'values': values,
        'geography': geography,
        'model': model_name
    }


# ==============================================================================
# 7. EXPORT RESULTS
# ==============================================================================

def export_results(results):
    """
    Export results to CSV and JSON
    """
    print("\nExporting results...")
    
    # Export tables for each model
    for model_name in results.keys():
        table = create_results_table(results, model_name)
        filename = OUTPUT_DIR / f"decomposition_{model_name}.csv"
        table.to_csv(filename, index=False)
        print(f"✓ Saved {filename}")
    
    # Export full results as JSON
    # Convert to serializable format
    results_json = {}
    for model_name, model_results in results.items():
        results_json[model_name] = {}
        for geo, result in model_results.items():
            # Convert tuples to lists for JSON
            result_copy = result.copy()
            result_copy['confidence_intervals'] = {
                k: list(v) for k, v in result['confidence_intervals'].items()
            }
            results_json[model_name][geo] = result_copy
    
    with open(OUTPUT_DIR / "decomposition_full_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✓ Saved {OUTPUT_DIR / 'decomposition_full_results.json'}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main execution function
    """
    print("="*80)
    print("DECOMPOSITION ANALYSIS: WHAT IS DRIVING THE DECLINE?")
    print("="*80)
    
    # Load data
    data = load_data()
    
    # Run analysis
    results = run_decomposition_analysis(data)
    
    # Export results
    export_results(results)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey findings:")
    
    # Summary for Canada
    if 'Canada' in results['baseline']:
        canada_result = results['baseline']['Canada']
        print(f"\nCanada ({BASELINE_YEAR}-{COMPARISON_YEAR}):")
        print(f"  Total decline: {canada_result['actual_change']:.3f} L per capita")
        print(f"  Explained by model: {canada_result['percent_explained']:.1f}%")
        print(f"\n  Main contributors:")
        for var, contrib in sorted(canada_result['contributions'].items(), 
                                   key=lambda x: abs(x[1]), reverse=True):
            pct = (contrib / canada_result['actual_change'] * 100)
            print(f"    {var}: {contrib:.3f} L ({pct:.1f}%)")
    
    print(f"\nNext steps:")
    print(f"  1. Review results in {OUTPUT_DIR}")
    print(f"  2. Run 03_elasticity_analysis.py")
    print(f"  3. Create visualizations with 04_visualization_dashboard.py")
    
    return results


if __name__ == "__main__":
    results = main()
