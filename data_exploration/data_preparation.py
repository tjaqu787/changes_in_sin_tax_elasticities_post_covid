"""
Canadian Alcohol Consumption Analysis - Data Preparation 
====================================================================

"""

import pandas as pd
import numpy as np
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

BASE_YEAR = 2015

GEOGRAPHIES = ['British_Columbia', 'Alberta', 'Ontario', 'Manitoba', 'Canada']

# ==============================================================================
# 1. LOAD RAW DATA
# ==============================================================================

def load_data():
    """Load all raw data files"""
    print("Loading raw data files...")
    
    data = {
        'consumption': pd.read_csv(DATA_DIR / "alcohol_consuption_data.csv"),
        'population': pd.read_csv(DATA_DIR / "population_estimates.csv"),
        'income': pd.read_csv(DATA_DIR / "income_data.csv"),
        'cpi': pd.read_csv(DATA_DIR / "cpi.csv"),
        'basket': pd.read_csv(DATA_DIR / "basket_weights.csv")
    }
    
    print(f"✓ Loaded {len(data)} datasets")
    return data


# ==============================================================================
# 2. CONSTRUCT CONSUMPTION VARIABLES
# ==============================================================================

def prepare_consumption(df):
    """Extract per capita consumption by beverage type"""
    print("\nPreparing consumption data...")
    
    # Pivot to long format
    consumption_long = df.melt(
        id_vars=['Geography', 'beverage_type', 'Value_volume_and_absolute_volume'],
        var_name='year',
        value_name='value'
    )
    
    # Clean year column
    consumption_long['year'] = consumption_long['year'].str.extract('(\d{4})').astype(int)
    
    # Filter for absolute volume per capita
    consumption = consumption_long[
        consumption_long['Value_volume_and_absolute_volume'] == 'Absolute_volume_for_total_per_capita_sales'
    ].copy()
    
    # Pivot to wide format by beverage type
    consumption_wide = consumption.pivot_table(
        index=['Geography', 'year'],
        columns='beverage_type',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Clean column names
    consumption_wide.columns.name = None
    consumption_wide.columns = ['Geography', 'year'] + [
        f'consumption_{col.lower().replace(" ", "_")}' 
        for col in consumption_wide.columns[2:]
    ]
    
    print(f"✓ Consumption data: {len(consumption_wide)} rows, {consumption_wide['year'].nunique()} years")
    return consumption_wide


# ==============================================================================
# 3. CONSTRUCT DEMOGRAPHIC VARIABLES (IMPROVED)
# ==============================================================================

def prepare_demographics(df):
    """
    Calculate demographic shares with focus on key age groups
    
    Key insight: Both aging (65+) AND youth behavior (18-21) matter
    - Older cohorts drink less (biological, health)
    - Younger cohorts drinking less (social norms, substitutes)
    """
    print("\nPreparing demographic data...")
    
    # Pivot to long format
    pop_long = df.melt(
        id_vars=['Geography', 'age_group'],
        var_name='year',
        value_name='population'
    )
    
    pop_long['year'] = pop_long['year'].astype(int)
    
    # Calculate total population and shares for key age groups
    demographics = pop_long.groupby(['Geography', 'year']).apply(
        lambda x: pd.Series({
            'total_population': x['population'].sum(),
            'population_18_21': x[x['age_group'] == '18_to_21_years']['population'].sum(),
            'population_15_64': x[x['age_group'] == '15_to_64_years']['population'].sum(),
            'population_65plus': x[x['age_group'] == '65_years_and_older']['population'].sum()
        })
    ).reset_index()
    
    # Calculate shares
    demographics['share_18_21'] = demographics['population_18_21'] / demographics['total_population']
    demographics['share_15_64'] = demographics['population_15_64'] / demographics['total_population']
    demographics['share_65plus'] = demographics['population_65plus'] / demographics['total_population']
    
    # Calculate adult population (15+) for per capita calculations
    demographics['adult_population'] = (
        demographics['population_15_64'] + demographics['population_65plus']
    )
    
    # Calculate demographic change rates (useful for analysis)
    demographics = demographics.sort_values(['Geography', 'year'])
    for col in ['share_18_21', 'share_65plus']:
        demographics[f'{col}_change'] = demographics.groupby('Geography')[col].diff()
    
    print(f"✓ Demographics data: {len(demographics)} rows")
    print(f"  Share 65+: {demographics['share_65plus'].min():.3f} to {demographics['share_65plus'].max():.3f}")
    print(f"  Share 18-21: {demographics['share_18_21'].min():.3f} to {demographics['share_18_21'].max():.3f}")
    
    return demographics


# ==============================================================================
# 4. CONSTRUCT INCOME DISTRIBUTION (IMPROVED)
# ==============================================================================

def prepare_income_distribution(df):
    """
    Calculate income distribution measures: quintiles, deciles, and moments
    
    Why this matters:
    - Different income groups may have different elasticities
    - Inequality may affect aggregate consumption
    - Policy impacts vary by income level
    """
    print("\nPreparing income distribution data...")
    
    # Pivot to long format
    income_long = df.melt(
        id_vars=['Geography', 'Statistics'],
        var_name='year',
        value_name='percentage'
    )
    
    income_long['year'] = income_long['year'].astype(int)
    
    # Define bracket midpoints (in dollars)
    bracket_midpoints = {
        'Percentage_under_5000': 2500,
        '5000_to_9999': 7500,
        '10000_to_19999': 15000,
        '20000_to_29999': 25000,
        '30000_to_39999': 35000,
        '40000_to_49999': 45000,
        '50000_to_59999': 55000,
        '60000_to_69999': 65000,
        '70000_to_79999': 75000,
        '80000_to_89999': 85000,
        '90000_to_99999': 95000,
        '100000_and_over': 125000  # Conservative estimate
    }
    
    # Map midpoints
    income_long['midpoint'] = income_long['Statistics'].map(bracket_midpoints)
    income_long['percentage'] = income_long['percentage'] / 100  # Convert to proportion
    
    # Calculate distribution measures
    income_stats = income_long.groupby(['Geography', 'year']).apply(
        lambda x: pd.Series({
            # Central tendency
            'mean_income': np.average(x['midpoint'], weights=x['percentage']),
            'median_income_approx': np.average(x['midpoint'], weights=x['percentage']),  # Simplified
            
            # Quintiles (20% groups) - cumulative approach
            'income_q1': calculate_percentile(x, 0.20),
            'income_q2': calculate_percentile(x, 0.40),
            'income_q3': calculate_percentile(x, 0.60),
            'income_q4': calculate_percentile(x, 0.80),
            'income_q5': calculate_percentile(x, 0.95),
            
            # Deciles (10% groups) - for more granular analysis
            'income_d1': calculate_percentile(x, 0.10),
            'income_d5': calculate_percentile(x, 0.50),  # Median
            'income_d9': calculate_percentile(x, 0.90),
            
            # Inequality measures
            'income_variance': np.average((x['midpoint'] - np.average(x['midpoint'], weights=x['percentage']))**2, 
                                         weights=x['percentage']),
            'income_std': np.sqrt(np.average((x['midpoint'] - np.average(x['midpoint'], weights=x['percentage']))**2, 
                                            weights=x['percentage'])),
            
            # Share in low/high income
            'share_under_30k': x[x['midpoint'] < 30000]['percentage'].sum(),
            'share_over_100k': x[x['midpoint'] >= 100000]['percentage'].sum()
        })
    ).reset_index()
    
    # Calculate Gini coefficient approximation
    income_stats['gini_approx'] = income_stats.apply(
        lambda row: row['income_std'] / row['mean_income'] if row['mean_income'] > 0 else 0, 
        axis=1
    )
    
    print(f"✓ Income distribution data: {len(income_stats)} rows")
    print(f"  Mean income range: ${income_stats['mean_income'].min():,.0f} to ${income_stats['mean_income'].max():,.0f}")
    print(f"  Income inequality (Gini approx): {income_stats['gini_approx'].min():.3f} to {income_stats['gini_approx'].max():.3f}")
    
    return income_stats


def calculate_percentile(group, percentile):
    """
    Calculate percentile from grouped income data
    
    Simple linear interpolation approach
    """
    sorted_data = group.sort_values('midpoint')
    cumsum = sorted_data['percentage'].cumsum()
    
    # Find bracket containing percentile
    idx = cumsum.searchsorted(percentile)
    if idx >= len(sorted_data):
        return sorted_data['midpoint'].iloc[-1]
    
    return sorted_data['midpoint'].iloc[idx]


# ==============================================================================
# 5. CONSTRUCT PRICE VARIABLES (IMPROVED)
# ==============================================================================

def prepare_prices(cpi_df, basket_df):
    """
    Construct REAL alcohol price and relevant CPI components
    
    Key insight: Only keep CPI categories that matter for analysis:
    1. Alcohol (our main variable)
    2. Overall/All-items (for deflating)
    3. Food (possible substitute)
    4. Recreation (possible complement/substitute)
    5. Cannabis (direct substitute - if available)
    
    Drop irrelevant: gasoline, clothing, etc.
    """
    print("\nPreparing price data...")
    
    # === Extract relevant CPI categories ===
    cpi_long = cpi_df.melt(
        id_vars=['Geography', 'Products_and_product_groups'],
        var_name='year_month',
        value_name='cpi'
    )
    
    # Extract year
    cpi_long['year'] = cpi_long['year_month'].str.extract('(\d{4})').astype(int)
    
    # Calculate annual average CPI
    cpi_annual = cpi_long.groupby(
        ['Geography', 'year', 'Products_and_product_groups']
    )['cpi'].mean().reset_index()
    
    # Filter for relevant categories only
    relevant_categories = [
        'Alcoholic_beverages',
        'All-items',
        'Food',
        'Recreation_education_and_reading',
        'Recreational_cannabis'
    ]
    
    # Try to find "All-items" or similar
    all_items_variants = ['All-items', 'All items', 'CPI', 'Total']
    available_categories = cpi_annual['Products_and_product_groups'].unique()
    
    all_items_col = None
    for variant in all_items_variants:
        if variant in available_categories:
            all_items_col = variant
            break
    
    if all_items_col is None:
        print(f"  ⚠ Warning: Could not find overall CPI column")
        print(f"  Available: {list(available_categories)}")
        all_items_col = available_categories[0]  # Use first as fallback
    
    # Filter for available relevant categories
    available_relevant = [cat for cat in relevant_categories if cat in available_categories]
    if all_items_col not in available_relevant:
        available_relevant.append(all_items_col)
    
    cpi_filtered = cpi_annual[
        cpi_annual['Products_and_product_groups'].isin(available_relevant)
    ].copy()
    
    # Pivot to wide format
    prices = cpi_filtered.pivot_table(
        index=['Geography', 'year'],
        columns='Products_and_product_groups',
        values='cpi',
        aggfunc='first'
    ).reset_index()
    
    prices.columns.name = None
    
    # Rename key columns for clarity
    rename_dict = {
        all_items_col: 'overall_cpi',
        'Alcoholic_beverages': 'alcohol_cpi',
        'Food': 'food_cpi',
        'Recreation_education_and_reading': 'recreation_cpi',
        'Recreational_cannabis': 'cannabis_cpi'
    }
    
    prices = prices.rename(columns=rename_dict)
    
    # Calculate REAL prices (relative to overall CPI)
    if 'alcohol_cpi' in prices.columns and 'overall_cpi' in prices.columns:
        prices['real_alcohol_price'] = (prices['alcohol_cpi'] / prices['overall_cpi']) * 100
        
        # Index to base year
        base_prices = prices[prices['year'] == BASE_YEAR].groupby('Geography')['real_alcohol_price'].first()
        prices = prices.merge(
            base_prices.rename('base_real_price'),
            left_on='Geography',
            right_index=True,
            how='left'
        )
        prices['real_alcohol_price_index'] = (prices['real_alcohol_price'] / prices['base_real_price']) * 100
        
        # Log transformations for regression
        prices['log_real_alcohol_price'] = np.log(prices['real_alcohol_price'])
    
    # Calculate relative prices for substitutes/complements
    if 'food_cpi' in prices.columns and 'overall_cpi' in prices.columns:
        prices['relative_food_price'] = (prices['food_cpi'] / prices['overall_cpi']) * 100
    
    if 'cannabis_cpi' in prices.columns and 'overall_cpi' in prices.columns:
        prices['relative_cannabis_price'] = (prices['cannabis_cpi'] / prices['overall_cpi']) * 100
    
    print(f"✓ Price data: {len(prices)} rows")
    print(f"  Available CPI categories: {[col for col in prices.columns if 'cpi' in col.lower()]}")
    
    if 'real_alcohol_price_index' in prices.columns:
        print(f"  Real price change ({BASE_YEAR}-latest):")
        for geo in prices['Geography'].unique():
            geo_data = prices[prices['Geography'] == geo]
            if len(geo_data) > 0 and BASE_YEAR in geo_data['year'].values:
                latest_year = geo_data['year'].max()
                base = 100
                latest = geo_data[geo_data['year'] == latest_year]['real_alcohol_price_index'].values[0]
                print(f"    {geo}: {base:.1f} → {latest:.1f} ({latest-base:+.1f}%)")
    
    return prices


# ==============================================================================
# 6. INTERPOLATE BASKET WEIGHTS 
# ==============================================================================

def interpolate_basket_weights(basket_df):
    """
    Interpolate basket weights for missing years
    
    Basket weight data: 2015, 2017, 2020, 2021, 2022, 2023, 2024
    Need: Annual data for 2015-2024
    
    Method: Linear interpolation between observed points
    """
    print("\nInterpolating basket weights...")
    
    # Pivot to long format
    basket_long = basket_df.melt(
        id_vars=['Geography', 'Products_and_product_groups'],
        var_name='year',
        value_name='basket_weight'
    )
    
    basket_long['year'] = basket_long['year'].astype(int)
    
    # Get unique years and fill range
    min_year = basket_long['year'].min()
    max_year = basket_long['year'].max()
    all_years = range(min_year, max_year + 1)
    
    # For each geography and product, interpolate
    interpolated_list = []
    
    for geo in basket_long['Geography'].unique():
        for product in basket_long['Products_and_product_groups'].unique():
            # Get data for this combination
            subset = basket_long[
                (basket_long['Geography'] == geo) & 
                (basket_long['Products_and_product_groups'] == product)
            ].sort_values('year')
            
            if len(subset) < 2:
                continue
            
            # Create full year range
            full_range = pd.DataFrame({'year': list(all_years)})
            
            # Merge with observed data
            merged = full_range.merge(subset, on='year', how='left')
            merged['Geography'] = geo
            merged['Products_and_product_groups'] = product
            
            # Interpolate missing values
            merged['basket_weight'] = merged['basket_weight'].interpolate(method='linear')
            
            # Forward fill for years after last observation
            merged['basket_weight'] = merged['basket_weight'].ffill()
            
            # Backward fill for years before first observation
            merged['basket_weight'] = merged['basket_weight'].bfill()
            
            interpolated_list.append(merged)
    
    basket_interpolated = pd.concat(interpolated_list, ignore_index=True)
    
    # Pivot to wide for easier merging
    basket_wide = basket_interpolated.pivot_table(
        index=['Geography', 'year'],
        columns='Products_and_product_groups',
        values='basket_weight',
        aggfunc='first'
    ).reset_index()
    
    basket_wide.columns.name = None
    
    # Focus on alcohol basket weight (main interest)
    if 'Alcoholic_beverages' in basket_wide.columns:
        basket_wide = basket_wide.rename(columns={'Alcoholic_beverages': 'alcohol_basket_weight'})
        # Keep only geography, year, and alcohol basket weight
        basket_wide = basket_wide[['Geography', 'year', 'alcohol_basket_weight']]
    
    print(f"✓ Basket weights: {len(basket_wide)} rows")
    print(f"  Years covered: {basket_wide['year'].min()} to {basket_wide['year'].max()}")
    
    return basket_wide


# ==============================================================================
# 7. MERGE AND CREATE FINAL DATASET
# ==============================================================================

def create_analysis_dataset(data):
    """Merge all components and create final analysis dataset"""
    print("\nCreating analysis dataset...")
    
    # Prepare each component
    consumption = prepare_consumption(data['consumption'])
    demographics = prepare_demographics(data['population'])
    income_dist = prepare_income_distribution(data['income'])
    prices = prepare_prices(data['cpi'], data['basket'])
    basket_weights = interpolate_basket_weights(data['basket'])
    
    # Merge everything
    analysis = consumption.merge(demographics, on=['Geography', 'year'], how='left')
    analysis = analysis.merge(income_dist, on=['Geography', 'year'], how='left')
    analysis = analysis.merge(prices, on=['Geography', 'year'], how='left')
    
    # Debug: Check what price-related columns were merged
    price_cols = [col for col in analysis.columns if 'price' in col.lower() or 'cpi' in col.lower()]
    print(f"\nPrice-related columns after merge: {price_cols}")
    
    analysis = analysis.merge(basket_weights, on=['Geography', 'year'], how='left')
    
    # Create derived variables
    if 'overall_cpi' in analysis.columns and 'mean_income' in analysis.columns:
        analysis['real_income'] = analysis['mean_income'] / analysis['overall_cpi'] * 100
        analysis['log_real_income'] = np.log(analysis['real_income'])
    
    # Log transformation of price
    if 'real_alcohol_price' in analysis.columns:
        analysis['log_real_alcohol_price'] = np.log(analysis['real_alcohol_price'])
    
    # Log transformation of consumption
    if 'consumption_total_alcoholic_beverages' in analysis.columns:
        analysis['log_consumption_total'] = np.log(analysis['consumption_total_alcoholic_beverages'])
    
    # Time variables
    analysis['year_centered'] = analysis['year'] - BASE_YEAR
    analysis['post_covid'] = (analysis['year'] >= 2020).astype(int)
    
    # Create year dummies for robustness checks
    year_dummies = pd.get_dummies(analysis['year'], prefix='year')
    analysis = pd.concat([analysis, year_dummies], axis=1)
    
    # Filter for geographies of interest
    analysis = analysis[analysis['Geography'].isin(GEOGRAPHIES)]
    
    print(f"\n✓ Final dataset: {len(analysis)} rows")
    print(f"  Time period: {analysis['year'].min()} - {analysis['year'].max()}")
    print(f"  Geographies: {', '.join(analysis['Geography'].unique())}")
    
    # Debug: check which key columns exist
    key_cols = ['log_consumption_total', 'log_real_alcohol_price', 'log_real_income']
    existing_cols = [col for col in key_cols if col in analysis.columns]
    missing_cols = [col for col in key_cols if col not in analysis.columns]
    
    if missing_cols:
        print(f"  ⚠ Missing columns: {missing_cols}")
        print(f"  Available columns with 'log': {[c for c in analysis.columns if 'log' in c.lower()]}")
        print(f"  Available columns with 'price': {[c for c in analysis.columns if 'price' in c.lower()]}")
        print(f"  Available columns with 'income': {[c for c in analysis.columns if 'income' in c.lower()]}")
    
    if existing_cols:
        print(f"  Complete cases: {analysis.dropna(subset=existing_cols).shape[0]}")
    
    return analysis


# ==============================================================================
# 8. DATA VALIDATION
# ==============================================================================

def validate_data(df):
    """Perform data quality checks"""
    print("\n" + "="*80)
    print("DATA VALIDATION")
    print("="*80)
    
    # Key variables for analysis
    key_vars = [
        'consumption_total_alcoholic_beverages',
        'log_consumption_total',
        'real_alcohol_price',
        'log_real_alcohol_price',
        'real_income',
        'log_real_income',
        'share_65plus',
        'share_18_21'
    ]
    
    # Check missing values
    print("\nMissing values in key variables:")
    available_vars = [v for v in key_vars if v in df.columns]
    missing = df[available_vars].isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(missing)
    else:
        print("  No missing values in key variables!")
    
    # Check for outliers
    print("\nPotential outliers (> 3 SD from mean):")
    for col in available_vars:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[np.abs((df[col] - mean) / std) > 3]
            if len(outliers) > 0:
                print(f"  {col}: {len(outliers)} observations")
    
    # Summary statistics
    print("\nSummary statistics:")
    print(df[available_vars].describe().round(3))


# ==============================================================================
# 9. EXPORT
# ==============================================================================

def export_summary_stats(df):
    """Create summary statistics table"""
    print("\nExporting summary statistics...")
    
    # Define desired columns and check which exist
    summary_cols = {
        'consumption_total_alcoholic_beverages': ['mean', 'std', 'min', 'max'],
        'real_alcohol_price': ['mean', 'std', 'min', 'max'],
        'real_income': ['mean', 'std', 'min', 'max'],
        'share_65plus': ['mean', 'std', 'min', 'max'],
        'share_18_21': ['mean', 'std', 'min', 'max'],
        'year': ['min', 'max', 'count']
    }
    
    # Filter to only existing columns
    existing_summary_cols = {col: aggs for col, aggs in summary_cols.items() if col in df.columns}
    
    if len(existing_summary_cols) == 0:
        print("  ⚠ Warning: No summary columns found in dataframe")
        return
    
    # Summary by geography
    summary = df.groupby('Geography').agg(existing_summary_cols).round(3)
    
    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv")
    print(f"✓ Saved to {OUTPUT_DIR / 'summary_statistics.csv'}")
    
    # Income distribution summary
    income_cols = [col for col in df.columns if col.startswith('income_q') or col.startswith('income_d')]
    if len(income_cols) > 0:
        income_summary = df.groupby('Geography')[income_cols].mean().round(0)
        income_summary.to_csv(OUTPUT_DIR / "income_distribution_summary.csv")
        print(f"✓ Saved to {OUTPUT_DIR / 'income_distribution_summary.csv'}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("CANADIAN ALCOHOL CONSUMPTION ANALYSIS - DATA PREPARATION (IMPROVED)")
    print("="*80)
    
    # Load data
    data = load_data()
    
    # Create analysis dataset
    analysis = create_analysis_dataset(data)
    
    # Validate data
    validate_data(analysis)
    
    # Export summary statistics
    export_summary_stats(analysis)
    
    # Save prepared data
    print("\nSaving prepared data...")
    analysis.to_csv(OUTPUT_DIR / "analysis_data.csv", index=False)
    print(f"✓ Saved to {OUTPUT_DIR / 'analysis_data.csv'}")
    
    # Save metadata
    metadata = {
        'base_year': BASE_YEAR,
        'geographies': GEOGRAPHIES,
        'time_period': f"{analysis['year'].min()}-{analysis['year'].max()}",
        'n_observations': len(analysis),
        'key_variables': [
            'log_consumption_total',
            'log_real_alcohol_price',
            'log_real_income',
            'share_65plus',
            'share_18_21',
            'post_covid'
        ],
        'income_measures': [col for col in analysis.columns if 'income' in col],
        'demographic_measures': [col for col in analysis.columns if 'share' in col]
    }
    
    import json
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return analysis


if __name__ == "__main__":
    analysis = main()