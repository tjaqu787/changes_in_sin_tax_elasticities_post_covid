"""
Canadian Alcohol Consumption - Elasticity Analysis
===================================================

Research Question: Has price elasticity of demand changed over time?

Three key tests:
1. Structural break: Has elasticity changed pre vs post-COVID?
2. Age heterogeneity: Do different age groups respond differently to price?
3. Beverage heterogeneity: Do elasticities differ by beverage type?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR = Path("prepared_data")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

STRUCTURAL_BREAK_YEAR = 2020  # COVID

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

def load_data():
    """Load prepared analysis data"""
    print("Loading prepared data...")
    df = pd.read_csv(DATA_DIR / "analysis_data.csv")
    print(f"✓ Loaded {len(df)} observations")
    return df


# ==============================================================================
# 2. TEST 1: STRUCTURAL BREAK (PRE VS POST-COVID)
# ==============================================================================

def test_structural_break(data, geography='Canada'):
    """
    Test if price elasticity changed after COVID-19
    
    Model: log(consumption) = β₀ + β₁·log(price) + β₂·log(price)×post_covid + 
                             β₃·log(income) + β₄·share_65+ + β₅·post_covid + ε
    
    H₀: β₂ = 0 (no change in elasticity)
    H₁: β₂ ≠ 0 (elasticity changed)
    """
    print(f"\n{'='*80}")
    print(f"TEST 1: STRUCTURAL BREAK IN ELASTICITY")
    print(f"Geography: {geography}")
    print(f"{'='*80}")
    
    # Filter data
    df = data[data['Geography'] == geography].copy()
    df = df.dropna(subset=['log_consumption_total', 'log_real_alcohol_price', 
                           'log_real_income', 'share_65plus'])
    
    # Create interaction term
    df['log_price_x_post_covid'] = df['log_real_alcohol_price'] * df['post_covid']
    
    # === Model 1: Baseline (no interaction) ===
    y = df['log_consumption_total']
    X = df[['log_real_alcohol_price', 'log_real_income', 'share_65plus', 'post_covid']]
    X = sm.add_constant(X)
    
    model_baseline = OLS(y, X).fit()
    
    # === Model 2: With interaction ===
    X_interact = df[['log_real_alcohol_price', 'log_price_x_post_covid', 
                     'log_real_income', 'share_65plus', 'post_covid']]
    X_interact = sm.add_constant(X_interact)
    
    model_interact = OLS(y, X_interact).fit()
    
    # === F-test for structural break ===
    # Compare models with and without interaction
    f_stat = ((model_baseline.ssr - model_interact.ssr) / 1) / (model_interact.ssr / model_interact.df_resid)
    p_value = 1 - stats.f.cdf(f_stat, 1, model_interact.df_resid)
    
    # Calculate elasticities
    elasticity_pre = model_interact.params['log_real_alcohol_price']
    elasticity_post = (model_interact.params['log_real_alcohol_price'] + 
                      model_interact.params['log_price_x_post_covid'])
    
    # Standard errors for elasticities
    # For post-COVID, need variance of sum: Var(β₁ + β₂) = Var(β₁) + Var(β₂) + 2Cov(β₁,β₂)
    se_pre = model_interact.bse['log_real_alcohol_price']
    
    vcov = model_interact.cov_params()
    var_post = (vcov.loc['log_real_alcohol_price', 'log_real_alcohol_price'] + 
                vcov.loc['log_price_x_post_covid', 'log_price_x_post_covid'] + 
                2 * vcov.loc['log_real_alcohol_price', 'log_price_x_post_covid'])
    se_post = np.sqrt(var_post)
    
    # Results
    results = {
        'geography': geography,
        'elasticity_pre_covid': {
            'estimate': elasticity_pre,
            'se': se_pre,
            'ci_lower': elasticity_pre - 1.96 * se_pre,
            'ci_upper': elasticity_pre + 1.96 * se_pre
        },
        'elasticity_post_covid': {
            'estimate': elasticity_post,
            'se': se_post,
            'ci_lower': elasticity_post - 1.96 * se_post,
            'ci_upper': elasticity_post + 1.96 * se_post
        },
        'interaction_coefficient': {
            'estimate': model_interact.params['log_price_x_post_covid'],
            'se': model_interact.bse['log_price_x_post_covid'],
            't_stat': model_interact.tvalues['log_price_x_post_covid'],
            'p_value': model_interact.pvalues['log_price_x_post_covid']
        },
        'f_test': {
            'f_stat': f_stat,
            'p_value': p_value,
            'df1': 1,
            'df2': model_interact.df_resid
        },
        'model_comparison': {
            'baseline_r2': model_baseline.rsquared,
            'interact_r2': model_interact.rsquared,
            'aic_baseline': model_baseline.aic,
            'aic_interact': model_interact.aic
        }
    }
    
    # Print results
    print(f"\nPrice Elasticity Estimates:")
    print(f"  Pre-COVID (pre-{STRUCTURAL_BREAK_YEAR}):")
    print(f"    Elasticity: {elasticity_pre:.3f} (SE: {se_pre:.3f})")
    print(f"    95% CI: [{results['elasticity_pre_covid']['ci_lower']:.3f}, {results['elasticity_pre_covid']['ci_upper']:.3f}]")
    
    print(f"\n  Post-COVID ({STRUCTURAL_BREAK_YEAR}+):")
    print(f"    Elasticity: {elasticity_post:.3f} (SE: {se_post:.3f})")
    print(f"    95% CI: [{results['elasticity_post_covid']['ci_lower']:.3f}, {results['elasticity_post_covid']['ci_upper']:.3f}]")
    
    print(f"\n  Change in elasticity: {elasticity_post - elasticity_pre:.3f}")
    
    print(f"\nStructural Break Test:")
    print(f"  Interaction coefficient: {results['interaction_coefficient']['estimate']:.3f}")
    print(f"  t-statistic: {results['interaction_coefficient']['t_stat']:.3f}")
    print(f"  p-value: {results['interaction_coefficient']['p_value']:.4f}")
    
    print(f"\n  F-test for structural break:")
    print(f"    F({results['f_test']['df1']}, {results['f_test']['df2']}) = {f_stat:.3f}")
    print(f"    p-value: {p_value:.4f}")
    
    significance = ''
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    elif p_value < 0.1:
        significance = '.'
    
    print(f"    Significance: {significance if significance else 'Not significant'}")
    
    if p_value < 0.05:
        print(f"\n  ✓ Structural break detected: Elasticity changed significantly after {STRUCTURAL_BREAK_YEAR}")
        if elasticity_post < elasticity_pre:
            print(f"    → Consumers became MORE price-sensitive post-COVID")
        else:
            print(f"    → Consumers became LESS price-sensitive post-COVID")
    else:
        print(f"\n  ✗ No significant structural break detected")
    
    return results


# ==============================================================================
# 3. TEST 2: AGE HETEROGENEITY
# ==============================================================================

def test_age_heterogeneity(data, geography='Canada'):
    """
    Test if price elasticity differs by age composition
    
    Model: log(consumption) = β₀ + β₁·log(price) + β₂·log(price)×share_65+ + 
                             β₃·log(income) + β₄·share_65+ + ε
    
    H₀: β₂ = 0 (no age heterogeneity)
    H₁: β₂ ≠ 0 (elasticity varies with age)
    """
    print(f"\n{'='*80}")
    print(f"TEST 2: AGE HETEROGENEITY IN ELASTICITY")
    print(f"Geography: {geography}")
    print(f"{'='*80}")
    
    # Filter data
    df = data[data['Geography'] == geography].copy()
    df = df.dropna(subset=['log_consumption_total', 'log_real_alcohol_price', 
                           'log_real_income', 'share_65plus'])
    
    # Create interaction term
    df['log_price_x_share_65plus'] = df['log_real_alcohol_price'] * df['share_65plus']
    
    # === Model 1: Baseline (no interaction) ===
    y = df['log_consumption_total']
    X = df[['log_real_alcohol_price', 'log_real_income', 'share_65plus']]
    X = sm.add_constant(X)
    
    model_baseline = OLS(y, X).fit()
    
    # === Model 2: With interaction ===
    X_interact = df[['log_real_alcohol_price', 'log_price_x_share_65plus', 
                     'log_real_income', 'share_65plus']]
    X_interact = sm.add_constant(X_interact)
    
    model_interact = OLS(y, X_interact).fit()
    
    # === F-test for age heterogeneity ===
    f_stat = ((model_baseline.ssr - model_interact.ssr) / 1) / (model_interact.ssr / model_interact.df_resid)
    p_value = 1 - stats.f.cdf(f_stat, 1, model_interact.df_resid)
    
    # Calculate elasticities at different age compositions
    share_65_low = df['share_65plus'].quantile(0.25)
    share_65_median = df['share_65plus'].median()
    share_65_high = df['share_65plus'].quantile(0.75)
    
    elasticity_low_age = (model_interact.params['log_real_alcohol_price'] + 
                          model_interact.params['log_price_x_share_65plus'] * share_65_low)
    elasticity_median_age = (model_interact.params['log_real_alcohol_price'] + 
                             model_interact.params['log_price_x_share_65plus'] * share_65_median)
    elasticity_high_age = (model_interact.params['log_real_alcohol_price'] + 
                           model_interact.params['log_price_x_share_65plus'] * share_65_high)
    
    # Results
    results = {
        'geography': geography,
        'elasticity_by_age_composition': {
            'low_65plus': {'share_65plus': share_65_low, 'elasticity': elasticity_low_age},
            'median_65plus': {'share_65plus': share_65_median, 'elasticity': elasticity_median_age},
            'high_65plus': {'share_65plus': share_65_high, 'elasticity': elasticity_high_age}
        },
        'interaction_coefficient': {
            'estimate': model_interact.params['log_price_x_share_65plus'],
            'se': model_interact.bse['log_price_x_share_65plus'],
            't_stat': model_interact.tvalues['log_price_x_share_65plus'],
            'p_value': model_interact.pvalues['log_price_x_share_65plus']
        },
        'f_test': {
            'f_stat': f_stat,
            'p_value': p_value
        },
        'model_comparison': {
            'baseline_r2': model_baseline.rsquared,
            'interact_r2': model_interact.rsquared
        }
    }
    
    # Print results
    print(f"\nPrice Elasticity by Age Composition:")
    print(f"  Low % 65+ ({share_65_low*100:.1f}%): {elasticity_low_age:.3f}")
    print(f"  Median % 65+ ({share_65_median*100:.1f}%): {elasticity_median_age:.3f}")
    print(f"  High % 65+ ({share_65_high*100:.1f}%): {elasticity_high_age:.3f}")
    
    print(f"\nAge Interaction Test:")
    print(f"  Interaction coefficient: {results['interaction_coefficient']['estimate']:.3f}")
    print(f"  t-statistic: {results['interaction_coefficient']['t_stat']:.3f}")
    print(f"  p-value: {results['interaction_coefficient']['p_value']:.4f}")
    
    print(f"\n  F-test for age heterogeneity:")
    print(f"    F-statistic: {f_stat:.3f}")
    print(f"    p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"\n  ✓ Age heterogeneity detected: Elasticity varies with age composition")
        if model_interact.params['log_price_x_share_65plus'] < 0:
            print(f"    → Older populations are MORE price-sensitive")
        else:
            print(f"    → Older populations are LESS price-sensitive")
    else:
        print(f"\n  ✗ No significant age heterogeneity detected")
    
    return results


# ==============================================================================
# 4. TEST 3: BEVERAGE-SPECIFIC ELASTICITIES
# ==============================================================================

def test_beverage_elasticities(data, geography='Canada'):
    """
    Estimate price elasticity for each beverage type
    
    Tests if elasticities differ across:
    - Beer
    - Wine
    - Spirits
    """
    print(f"\n{'='*80}")
    print(f"TEST 3: BEVERAGE-SPECIFIC ELASTICITIES")
    print(f"Geography: {geography}")
    print(f"{'='*80}")
    
    # Filter data
    df = data[data['Geography'] == geography].copy()
    
    # Beverage types to analyze
    beverages = {
        'Beer': 'consumption_beer',
        'Wine': 'consumption_wines',
        'Spirits': 'consumption_spirits'
    }
    
    results = {'geography': geography, 'beverages': {}}
    
    for bev_name, bev_col in beverages.items():
        if bev_col not in df.columns:
            print(f"\n  ⚠ {bev_name} not found in data, skipping...")
            continue
        
        # Create log consumption
        df[f'log_{bev_col}'] = np.log(df[bev_col])
        
        # Drop missing
        bev_df = df.dropna(subset=[f'log_{bev_col}', 'log_real_alcohol_price', 
                                   'log_real_income', 'share_65plus'])
        
        if len(bev_df) < 10:
            print(f"\n  ⚠ Insufficient data for {bev_name}, skipping...")
            continue
        
        # Estimate model
        y = bev_df[f'log_{bev_col}']
        X = bev_df[['log_real_alcohol_price', 'log_real_income', 'share_65plus']]
        X = sm.add_constant(X)
        
        model = OLS(y, X).fit()
        
        # Extract elasticity
        elasticity = model.params['log_real_alcohol_price']
        se = model.bse['log_real_alcohol_price']
        
        results['beverages'][bev_name] = {
            'elasticity': elasticity,
            'se': se,
            'ci_lower': elasticity - 1.96 * se,
            'ci_upper': elasticity + 1.96 * se,
            't_stat': model.tvalues['log_real_alcohol_price'],
            'p_value': model.pvalues['log_real_alcohol_price'],
            'r2': model.rsquared,
            'n_obs': len(bev_df)
        }
        
        # Print
        print(f"\n{bev_name}:")
        print(f"  Elasticity: {elasticity:.3f} (SE: {se:.3f})")
        print(f"  95% CI: [{results['beverages'][bev_name]['ci_lower']:.3f}, {results['beverages'][bev_name]['ci_upper']:.3f}]")
        print(f"  t-statistic: {results['beverages'][bev_name]['t_stat']:.3f}")
        print(f"  p-value: {results['beverages'][bev_name]['p_value']:.4f}")
        print(f"  R²: {results['beverages'][bev_name]['r2']:.3f}")
        print(f"  N: {results['beverages'][bev_name]['n_obs']}")
    
    # Test if elasticities are significantly different
    print(f"\nComparison:")
    bev_names = list(results['beverages'].keys())
    if len(bev_names) >= 2:
        for i in range(len(bev_names)):
            for j in range(i+1, len(bev_names)):
                bev1, bev2 = bev_names[i], bev_names[j]
                e1 = results['beverages'][bev1]['elasticity']
                e2 = results['beverages'][bev2]['elasticity']
                se1 = results['beverages'][bev1]['se']
                se2 = results['beverages'][bev2]['se']
                
                # Test difference
                diff = e1 - e2
                se_diff = np.sqrt(se1**2 + se2**2)
                t_stat = diff / se_diff
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=20))
                
                print(f"  {bev1} vs {bev2}: Δ = {diff:.3f} (p = {p_value:.4f})")
    
    return results


# ==============================================================================
# 5. RUN ALL ELASTICITY TESTS
# ==============================================================================

def run_elasticity_analysis(data):
    """
    Run all elasticity tests for all geographies
    """
    print("="*80)
    print("ELASTICITY ANALYSIS: HAS PRICE SENSITIVITY CHANGED?")
    print("="*80)
    
    geographies = data['Geography'].unique()
    all_results = {}
    
    for geo in geographies:
        print(f"\n\n{'#'*80}")
        print(f"# GEOGRAPHY: {geo}")
        print(f"{'#'*80}")
        
        all_results[geo] = {
            'structural_break': test_structural_break(data, geo),
            'age_heterogeneity': test_age_heterogeneity(data, geo),
            'beverage_elasticities': test_beverage_elasticities(data, geo)
        }
    
    return all_results


# ==============================================================================
# 6. EXPORT RESULTS
# ==============================================================================

def export_results(results):
    """
    Export elasticity results
    """
    print("\n\nExporting results...")
    
    # Export as JSON
    with open(OUTPUT_DIR / "elasticity_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved {OUTPUT_DIR / 'elasticity_results.json'}")
    
    # Create summary tables
    
    # 1. Structural break summary
    break_summary = []
    for geo, result in results.items():
        break_data = result['structural_break']
        break_summary.append({
            'Geography': geo,
            'Elasticity_Pre': break_data['elasticity_pre_covid']['estimate'],
            'SE_Pre': break_data['elasticity_pre_covid']['se'],
            'Elasticity_Post': break_data['elasticity_post_covid']['estimate'],
            'SE_Post': break_data['elasticity_post_covid']['se'],
            'Change': break_data['elasticity_post_covid']['estimate'] - break_data['elasticity_pre_covid']['estimate'],
            'F_Stat': break_data['f_test']['f_stat'],
            'P_Value': break_data['f_test']['p_value'],
            'Significant': 'Yes' if break_data['f_test']['p_value'] < 0.05 else 'No'
        })
    
    break_df = pd.DataFrame(break_summary)
    break_df.to_csv(OUTPUT_DIR / "elasticity_structural_break.csv", index=False)
    print(f"✓ Saved {OUTPUT_DIR / 'elasticity_structural_break.csv'}")
    
    # 2. Beverage elasticities
    bev_summary = []
    for geo, result in results.items():
        if 'beverages' in result['beverage_elasticities']:
            for bev_name, bev_data in result['beverage_elasticities']['beverages'].items():
                bev_summary.append({
                    'Geography': geo,
                    'Beverage': bev_name,
                    'Elasticity': bev_data['elasticity'],
                    'SE': bev_data['se'],
                    'CI_Lower': bev_data['ci_lower'],
                    'CI_Upper': bev_data['ci_upper'],
                    'P_Value': bev_data['p_value'],
                    'R2': bev_data['r2'],
                    'N': bev_data['n_obs']
                })
    
    if len(bev_summary) > 0:
        bev_df = pd.DataFrame(bev_summary)
        bev_df.to_csv(OUTPUT_DIR / "elasticity_by_beverage.csv", index=False)
        print(f"✓ Saved {OUTPUT_DIR / 'elasticity_by_beverage.csv'}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main execution
    """
    # Load data
    data = load_data()
    
    # Run analysis
    results = run_elasticity_analysis(data)
    
    # Export results
    export_results(results)
    
    print("\n" + "="*80)
    print("ELASTICITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nNext step: Create visualizations with 04_visualization_dashboard.py")
    
    return results


if __name__ == "__main__":
    results = main()
