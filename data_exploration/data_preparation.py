"""
Canadian Alcohol Consumption Analysis - Data Preparation with Cannabis Integration
====================================================================================

New Features:
- Integrates cannabis prevalence data from survey
- Creates variables for testing substitution/complementarity
- Interpolates cannabis data between survey years
- Adds post-legalization indicators
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

BASE_YEAR = 2015
CANNABIS_LEGALIZATION_YEAR = 2018  # Oct 17, 2018

GEOGRAPHIES = ['British_Columbia','Alberta','Ontario','Manitoba','Saskatchewan','New_Brunswick','Nova_Scotia','Newfoundland_and_Labrador','Canada']

# Geography name mapping: Survey names -> Analysis names
GEOGRAPHY_MAPPING = {
    'British Columbia': 'British_Columbia',
    'Alberta': 'Alberta',
    'Ontario': 'Ontario',
    'Manitoba': 'Manitoba',
    'Saskatchewan': 'Saskatchewan',
    'New Brunswick': 'New_Brunswick',
    'Nova Scotia': 'Nova_Scotia',
    'Newfoundland and Labrador': 'Newfoundland_and_Labrador',
    'Canada': 'Canada'
}

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
        'basket': pd.read_csv(DATA_DIR / "basket_weights.csv"),
        'cannabis': pd.read_csv(DATA_DIR / "cannabisMapData.csv")  # NEW
    }
    
    print(f"✓ Loaded {len(data)} datasets")
    return data


# ==============================================================================
# 2. PREPARE CANNABIS DATA (NEW)
# ==============================================================================

def prepare_cannabis_data(df):
    """
    Prepare cannabis prevalence data
    
    Key features:
    - Map province names to analysis format
    - Interpolate between survey years for complete time series
    - Create intensity indicators (daily vs. occasional use)
    - Add post-legalization indicators
    """
    print("\nPreparing cannabis prevalence data...")
    
    # Rename columns for clarity
    cannabis = df.rename(columns={
        'Year': 'year',
        'PRNAME_EN': 'province_name',
        'use_12months': 'cannabis_use_pct',
        'daily_12months': 'cannabis_daily_pct'
    }).copy()
    
    # Map province names
    cannabis['Geography'] = cannabis['province_name'].map(GEOGRAPHY_MAPPING)
    
    # Drop provinces not in our analysis
    cannabis = cannabis[cannabis['Geography'].isin(GEOGRAPHIES + ['Saskatchewan', 
                                                                   'New_Brunswick', 
                                                                   'Nova_Scotia',
                                                                   'Newfoundland_and_Labrador'])].copy()
    
    # Convert percentages to proportions
    cannabis['cannabis_use_rate'] = cannabis['cannabis_use_pct'] / 100
    cannabis['cannabis_daily_rate'] = cannabis['cannabis_daily_pct'] / 100
    
    # Calculate occasional use (used but not daily)
    cannabis['cannabis_occasional_rate'] = (
        cannabis['cannabis_use_rate'] - cannabis['cannabis_daily_rate']
    )
    
    # Interpolate for missing years (2015-2017, if needed)
    cannabis_complete = []
    
    for geo in cannabis['Geography'].unique():
        geo_data = cannabis[cannabis['Geography'] == geo].sort_values('year')
        
        # Get year range
        min_year = geo_data['year'].min()
        max_year = geo_data['year'].max()
        
        # Create complete year range (2015-2024 if possible, else use survey range)
        year_range = range(min(2015, min_year), max_year + 1)
        
        # Create full year dataframe
        full_years = pd.DataFrame({'year': list(year_range), 'Geography': geo})
        
        # Merge with actual data
        merged = full_years.merge(geo_data, on=['year', 'Geography'], how='left')
        
        # Interpolate missing values
        for col in ['cannabis_use_rate', 'cannabis_daily_rate', 'cannabis_occasional_rate']:
            if col in merged.columns:
                # Linear interpolation
                merged[col] = merged[col].interpolate(method='linear', limit_direction='both')
                
                # For years before first survey, use first observed value (conservative)
                merged[col] = merged[col].bfill()
        
        cannabis_complete.append(merged)
    
    cannabis_final = pd.concat(cannabis_complete, ignore_index=True)
    
    # Create legalization indicators
    cannabis_final['post_legalization'] = (cannabis_final['year'] >= CANNABIS_LEGALIZATION_YEAR).astype(int)
    cannabis_final['years_since_legalization'] = np.maximum(
        cannabis_final['year'] - CANNABIS_LEGALIZATION_YEAR, 0
    )
    
    # Calculate growth rates (for dynamic analysis)
    cannabis_final = cannabis_final.sort_values(['Geography', 'year'])
    for col in ['cannabis_use_rate', 'cannabis_daily_rate']:
        cannabis_final[f'{col}_growth'] = cannabis_final.groupby('Geography')[col].pct_change()
    
    # Keep only relevant columns
    cannabis_final = cannabis_final[[
        'Geography', 'year', 
        'cannabis_use_rate', 'cannabis_daily_rate', 'cannabis_occasional_rate',
        'post_legalization', 'years_since_legalization',
        'cannabis_use_rate_growth', 'cannabis_daily_rate_growth'
    ]]
    
    print(f"✓ Cannabis data: {len(cannabis_final)} rows")
    print(f"  Years covered: {cannabis_final['year'].min()} to {cannabis_final['year'].max()}")
    print(f"  Average use rate: {cannabis_final['cannabis_use_rate'].mean()*100:.1f}%")
    print(f"  Average daily use: {cannabis_final['cannabis_daily_rate'].mean()*100:.1f}%")
    
    return cannabis_final


# ==============================================================================
# 3. CREATE CANNABIS PRICE VARIABLE (FROM CPI)
# ==============================================================================

def extract_cannabis_price(cpi_df):
    """
    Extract cannabis CPI as a price proxy
    
    Note: Cannabis CPI started in 2018-2019 for most provinces
    """
    print("\nExtracting cannabis price data from CPI...")
    
    # Filter for cannabis CPI
    cannabis_cpi = cpi_df[cpi_df['Products_and_product_groups'] == 'Recreational_cannabis'].copy()
    
    if len(cannabis_cpi) == 0:
        print("  ⚠ Warning: No cannabis CPI data found")
        return None
    
    # Melt to long format
    cannabis_price = cannabis_cpi.melt(
        id_vars=['Geography'],
        var_name='year',
        value_name='cannabis_cpi'
    )
    
    cannabis_price['year'] = cannabis_price['year'].astype(int)
    
    # Calculate real price (relative to overall CPI)
    # This will be merged with main data later
    
    print(f"✓ Cannabis price data: {len(cannabis_price)} rows")
    print(f"  Available from year: {cannabis_price['year'].min()}")
    
    return cannabis_price


def prepare_consumption(df):
    """Extract per capita consumption by beverage type"""
    print("\nPreparing consumption data...")
    
    consumption_long = df.melt(
        id_vars=['Geography', 'type_of_beverage', 'Metric'],
        var_name='year',
        value_name='value'
    )
    
    consumption_long['year'] = consumption_long['year'].str.extract('(\d{4})').astype(int)
    
    consumption = consumption_long[
        consumption_long['Metric'] == 'Absolute_volume_for_total_per_capita_sales'
    ].copy()
    
    consumption_wide = consumption.pivot_table(
        index=['Geography', 'year'],
        columns='type_of_beverage',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    consumption_wide.columns.name = None
    consumption_wide.columns = ['Geography', 'year'] + [
        f'consumption_{col.lower().replace(" ", "_")}' 
        for col in consumption_wide.columns[2:]
    ]
    
    print(f"✓ Consumption data: {len(consumption_wide)} rows")
    return consumption_wide


def prepare_demographics(df):
    """Calculate demographic shares"""
    print("\nPreparing demographic data...")
    
    pop_long = df.melt(
        id_vars=['Geography', 'age_group'],
        var_name='year',
        value_name='population'
    )
    
    pop_long['year'] = pop_long['year'].astype(int)
    
    demographics = pop_long.groupby(['Geography', 'year']).apply(
        lambda x: pd.Series({
            'total_population': x['population'].sum(),
            'population_18_21': x[x['age_group'] == '18_to_21_years']['population'].sum(),
            'population_15_64': x[x['age_group'] == '15_to_64_years']['population'].sum(),
            'population_65plus': x[x['age_group'] == '65_years_and_older']['population'].sum()
        })
    ).reset_index()
    
    demographics['share_18_21'] = demographics['population_18_21'] / demographics['total_population']
    demographics['share_15_64'] = demographics['population_15_64'] / demographics['total_population']
    demographics['share_65plus'] = demographics['population_65plus'] / demographics['total_population']
    
    demographics['adult_population'] = (
        demographics['population_15_64'] + demographics['population_65plus']
    )
    
    demographics = demographics.sort_values(['Geography', 'year'])
    for col in ['share_18_21', 'share_65plus']:
        demographics[f'{col}_change'] = demographics.groupby('Geography')[col].diff()
    
    print(f"✓ Demographics data: {len(demographics)} rows")
    return demographics


import pandas as pd
import numpy as np

def prepare_income_distribution(df):
    """
    Extract mean/median income and calculate income distribution measures 
    using vectorized aggregation.
    """
    print("\nPreparing income distribution data...")
    
    # 1. Standardize column name (handle Statistic vs Statistics)
    if 'Statistics' in df.columns:
        df = df.rename(columns={'Statistics': 'Statistic'})
    
    # 2. Identify year columns (numeric headers) and id columns
    year_cols = [col for col in df.columns if col.isdigit()]
    id_cols = [col for col in df.columns if col not in year_cols]
    
    # 3. Melt to long format
    income_long = df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name='year',
        value_name='value'
    )
    income_long['year'] = income_long['year'].astype(int)
    
    # 4. Pivot and Aggregate
    # Using aggfunc='sum' ensures that split brackets (like 80000_to_9999) 
    # are summed together into a single value for that geography/year.
    income_pivoted = income_long.pivot_table(
        index=['Geography', 'year'],
        columns='Statistic',
        values='value',
        aggfunc='sum'
    ).reset_index()
    
    # Fill missing brackets with 0
    income_pivoted = income_pivoted.fillna(0)
    
    # 5. Define bracket groupings based on data labels
    q1_cols = ['Percentage_under_5000', '5000_to_9999', '10000_to_9999']
    q2_cols = ['20000_to_9999', '30000_to_9999', '40000_to_9999']
    q3_cols = ['50000_to_9999', '60000_to_9999', '70000_to_9999']
    q4_cols = ['80000_to_9999', '90000_to_9999', '100000_and_over']
    
    # 6. Calculate aggregate shares and measures
    income_pivoted['income_q1_share'] = income_pivoted[q1_cols].sum(axis=1)
    income_pivoted['income_q2_share'] = income_pivoted[q2_cols].sum(axis=1)
    income_pivoted['income_q3_share'] = income_pivoted[q3_cols].sum(axis=1)
    income_pivoted['income_q4_share'] = income_pivoted[q4_cols].sum(axis=1)
    
    income_pivoted['share_under_30k'] = income_pivoted[['Percentage_under_5000', '5000_to_9999', 
                                                       '10000_to_9999', '20000_to_9999']].sum(axis=1)
    income_pivoted['share_over_100k'] = income_pivoted.get('100000_and_over', 0)
    
    # Rename income columns for clarity
    income_pivoted = income_pivoted.rename(columns={
        'Average_employment_income': 'mean_income',
        'Median_employment_income': 'median_income'
    })
    
    # 7. Calculate Inequality Ratio (Q4 / Q1)
    income_pivoted['income_inequality_ratio'] = (
        income_pivoted['income_q4_share'] / income_pivoted['income_q1_share']
    ).replace([np.inf, -np.inf], np.nan)
    
    # 8. Select final columns
    final_cols = [
        'Geography', 'year', 'mean_income', 'median_income',
        'income_q1_share', 'income_q2_share', 'income_q3_share', 'income_q4_share',
        'share_under_30k', 'share_over_100k', 'income_inequality_ratio'
    ]
    
    income_stats = income_pivoted[final_cols].copy()
    
    print(f"✓ Income distribution data processed: {len(income_stats)} rows")
    return income_stats

def prepare_prices(cpi_df, basket_df):
    """Construct REAL alcohol price and relevant CPI components"""
    print("\nPreparing price data...")
    
    cpi_long = cpi_df.melt(
        id_vars=['Geography', 'Products_and_product_groups'],
        var_name='year_month',
        value_name='cpi'
    )
    cpi_long['year'] = cpi_long['year_month'].str.extract('(\d{4})').astype(int)
    
    cpi_annual = cpi_long.groupby(
        ['Geography', 'year', 'Products_and_product_groups']
    )['cpi'].mean().reset_index()
    
    # Expanded relevant categories - match actual column names
    relevant_categories = [
        'All-items',
        'Food',
        'Food_purchased_from_stores',
        'Food_purchased_from_restaurants',
        'Shelter',
        'Rented_accommodation',
        'Owned_accommodation',
        'Water_fuel_and_electricity',
        'Alcoholic_beverages_tobacco_products_and_recreational_cannabis',
        'Alcoholic_beverages',
        'Alcoholic_beverages_served_in_licensed_establishments',
        'Alcoholic_beverages_purchased_from_stores',
        'Beer_served_in_licensed_establishments',
        'Wine_served_in_licensed_establishments',
        'Liquor_served_in_licensed_establishments',
        'Beer_purchased_from_stores',
        'Wine_purchased_from_stores',
        'Liquor_purchased_from_stores',
        'Recreational_cannabis',
        'Recreation_education_and_reading'
    ]
    
    available_categories = cpi_annual['Products_and_product_groups'].unique()
    
    # Find all-items column
    all_items_col = 'All-items' if 'All-items' in available_categories else available_categories[0]
    
    # Filter for available relevant categories
    available_relevant = [cat for cat in relevant_categories if cat in available_categories]
    if all_items_col not in available_relevant:
        available_relevant.append(all_items_col)
    
    cpi_filtered = cpi_annual[
        cpi_annual['Products_and_product_groups'].isin(available_relevant)
    ].copy()
    
    prices = cpi_filtered.pivot_table(
        index=['Geography', 'year'],
        columns='Products_and_product_groups',
        values='cpi',
        aggfunc='first'
    ).reset_index()
    prices.columns.name = None
    
    # Rename columns for consistency
    rename_dict = {
        'All-items': 'overall_cpi',
        'Alcoholic_beverages': 'alcohol_cpi',
        'Alcoholic_beverages_served_in_licensed_establishments': 'alcohol_onpremise_cpi',
        'Alcoholic_beverages_purchased_from_stores': 'alcohol_offpremise_cpi',
        'Beer_served_in_licensed_establishments': 'beer_onpremise_cpi',
        'Wine_served_in_licensed_establishments': 'wine_onpremise_cpi',
        'Liquor_served_in_licensed_establishments': 'spirits_onpremise_cpi',
        'Beer_purchased_from_stores': 'beer_offpremise_cpi',
        'Wine_purchased_from_stores': 'wine_offpremise_cpi',
        'Liquor_purchased_from_stores': 'spirits_offpremise_cpi',
        'Food': 'food_cpi',
        'Food_purchased_from_stores': 'food_stores_cpi',
        'Food_purchased_from_restaurants': 'food_restaurants_cpi',
        'Shelter': 'shelter_cpi',
        'Rented_accommodation': 'rent_cpi',
        'Owned_accommodation': 'owned_cpi',
        'Water_fuel_and_electricity': 'utilities_cpi',
        'Recreation_education_and_reading': 'recreation_cpi',
        'Recreational_cannabis': 'cannabis_cpi'
    }
    prices = prices.rename(columns=rename_dict)
    
    # Calculate real prices (deflated by overall CPI)
    if 'alcohol_cpi' in prices.columns and 'overall_cpi' in prices.columns:
        prices['real_alcohol_price'] = (prices['alcohol_cpi'] / prices['overall_cpi']) * 100
        
        base_prices = prices[prices['year'] == BASE_YEAR].groupby('Geography')['real_alcohol_price'].first()
        prices = prices.merge(
            base_prices.rename('base_real_price'),
            left_on='Geography',
            right_index=True,
            how='left'
        )
        prices['real_alcohol_price_index'] = (prices['real_alcohol_price'] / prices['base_real_price']) * 100
        prices['log_real_alcohol_price'] = np.log(prices['real_alcohol_price'])
    
    # On-premise vs off-premise real prices
    if 'alcohol_onpremise_cpi' in prices.columns and 'overall_cpi' in prices.columns:
        prices['real_onpremise_price'] = (prices['alcohol_onpremise_cpi'] / prices['overall_cpi']) * 100
        prices['log_real_onpremise_price'] = np.log(prices['real_onpremise_price'])
    
    if 'alcohol_offpremise_cpi' in prices.columns and 'overall_cpi' in prices.columns:
        prices['real_offpremise_price'] = (prices['alcohol_offpremise_cpi'] / prices['overall_cpi']) * 100
        prices['log_real_offpremise_price'] = np.log(prices['real_offpremise_price'])
    
    # Beverage-specific real prices (stores)
    for bev, col_suffix in [('beer', 'offpremise'), ('wine', 'offpremise'), ('spirits', 'offpremise')]:
        cpi_col = f'{bev}_{col_suffix}_cpi'
        if cpi_col in prices.columns and 'overall_cpi' in prices.columns:
            prices[f'real_{bev}_price'] = (prices[cpi_col] / prices['overall_cpi']) * 100
            prices[f'log_real_{bev}_price'] = np.log(prices[f'real_{bev}_price'])
    
    # Beverage-specific on-premise prices
    for bev, col_suffix in [('beer', 'onpremise'), ('wine', 'onpremise'), ('spirits', 'onpremise')]:
        cpi_col = f'{bev}_{col_suffix}_cpi'
        if cpi_col in prices.columns and 'overall_cpi' in prices.columns:
            prices[f'real_{bev}_onpremise_price'] = (prices[cpi_col] / prices['overall_cpi']) * 100
            prices[f'log_real_{bev}_onpremise_price'] = np.log(prices[f'real_{bev}_onpremise_price'])
    
    # Cannabis
    if 'cannabis_cpi' in prices.columns and 'overall_cpi' in prices.columns:
        prices['real_cannabis_price'] = (prices['cannabis_cpi'] / prices['overall_cpi']) * 100
        prices['log_real_cannabis_price'] = np.log(prices['real_cannabis_price'])
        
        # Substitution ratios
        if 'alcohol_cpi' in prices.columns:
            prices['cannabis_alcohol_price_ratio'] = prices['cannabis_cpi'] / prices['alcohol_cpi']
    
    # Food and shelter (complements/substitutes)
    if 'food_cpi' in prices.columns and 'overall_cpi' in prices.columns:
        prices['relative_food_price'] = (prices['food_cpi'] / prices['overall_cpi']) * 100
    
    if 'food_restaurants_cpi' in prices.columns and 'overall_cpi' in prices.columns:
        prices['real_food_restaurants_price'] = (prices['food_restaurants_cpi'] / prices['overall_cpi']) * 100
    
    if 'shelter_cpi' in prices.columns and 'overall_cpi' in prices.columns:
        prices['real_shelter_price'] = (prices['shelter_cpi'] / prices['overall_cpi']) * 100
    
    print(f"✓ Price data: {len(prices)} rows")
    
    return prices

def interpolate_basket_weights(basket_df):
    """Interpolate basket weights for missing years"""
    print("\nInterpolating basket weights...")
    
    basket_long = basket_df.melt(
        id_vars=['Geography', 'Products_and_product_groups'],
        var_name='year',
        value_name='basket_weight'
    )
    
    basket_long['year'] = basket_long['year'].astype(int)
    
    min_year = basket_long['year'].min()
    max_year = basket_long['year'].max()
    all_years = range(min_year, max_year + 1)
    
    interpolated_list = []
    
    for geo in basket_long['Geography'].unique():
        for product in basket_long['Products_and_product_groups'].unique():
            subset = basket_long[
                (basket_long['Geography'] == geo) & 
                (basket_long['Products_and_product_groups'] == product)
            ].sort_values('year')
            
            if len(subset) < 2:
                continue
            
            full_range = pd.DataFrame({'year': list(all_years)})
            merged = full_range.merge(subset, on='year', how='left')
            merged['Geography'] = geo
            merged['Products_and_product_groups'] = product
            
            merged['basket_weight'] = merged['basket_weight'].interpolate(method='linear')
            merged['basket_weight'] = merged['basket_weight'].ffill()
            merged['basket_weight'] = merged['basket_weight'].bfill()
            
            interpolated_list.append(merged)
    
    basket_interpolated = pd.concat(interpolated_list, ignore_index=True)
    
    basket_wide = basket_interpolated.pivot_table(
        index=['Geography', 'year'],
        columns='Products_and_product_groups',
        values='basket_weight',
        aggfunc='first'
    ).reset_index()
    
    basket_wide.columns.name = None
    
    if 'Alcoholic_beverages' in basket_wide.columns:
        basket_wide = basket_wide.rename(columns={'Alcoholic_beverages': 'alcohol_basket_weight'})
        basket_wide = basket_wide[['Geography', 'year', 'alcohol_basket_weight']]
    
    print(f"✓ Basket weights: {len(basket_wide)} rows")
    return basket_wide


# ==============================================================================
# 5. MERGE AND CREATE FINAL DATASET (UPDATED)
# ==============================================================================

def create_analysis_dataset(data):
    """Merge all components including cannabis data"""
    print("\nCreating analysis dataset...")
    
    # Prepare each component
    consumption = prepare_consumption(data['consumption'])
    demographics = prepare_demographics(data['population'])
    income_dist = prepare_income_distribution(data['income'])
    prices = prepare_prices(data['cpi'], data['basket'])
    basket_weights = interpolate_basket_weights(data['basket'])
    cannabis_data = prepare_cannabis_data(data['cannabis'])  # NEW
    
    # Merge everything
    analysis = consumption.merge(demographics, on=['Geography', 'year'], how='left')
    analysis = analysis.merge(income_dist, on=['Geography', 'year'], how='left')
    analysis = analysis.merge(prices, on=['Geography', 'year'], how='left')
    analysis = analysis.merge(basket_weights, on=['Geography', 'year'], how='left')
    
    # NEW: Merge cannabis data
    analysis = analysis.merge(cannabis_data, on=['Geography', 'year'], how='left')
    
    # Create derived variables
    if 'overall_cpi' in analysis.columns and 'mean_income' in analysis.columns:
        analysis['real_income'] = analysis['mean_income'] / analysis['overall_cpi'] * 100
        analysis['log_real_income'] = np.log(analysis['real_income'])
    
    if 'real_alcohol_price' in analysis.columns:
        analysis['log_real_alcohol_price'] = np.log(analysis['real_alcohol_price'])
    
    if 'consumption_total_alcoholic_beverages' in analysis.columns:
        analysis['log_consumption_total'] = np.log(analysis['consumption_total_alcoholic_beverages'])
    
    # Time variables
    analysis['year_centered'] = analysis['year'] - BASE_YEAR
    analysis['post_covid'] = (analysis['year'] >= 2020).astype(int)
    
    # NEW: Cannabis interaction terms for substitution analysis
    if 'cannabis_use_rate' in analysis.columns and 'log_real_alcohol_price' in analysis.columns:
        # Interaction: does cannabis prevalence moderate price sensitivity?
        analysis['cannabis_x_alcohol_price'] = (
            analysis['cannabis_use_rate'] * analysis['log_real_alcohol_price']
        )
        
        # Age-specific cannabis effects
        if 'share_18_21' in analysis.columns:
            analysis['cannabis_x_youth_share'] = (
                analysis['cannabis_use_rate'] * analysis['share_18_21']
            )
    
    # Create year dummies for robustness checks
    year_dummies = pd.get_dummies(analysis['year'], prefix='year')
    analysis = pd.concat([analysis, year_dummies], axis=1)
    
    # Filter for geographies of interest
    analysis = analysis[analysis['Geography'].isin(GEOGRAPHIES)]
    
    print(f"\n✓ Final dataset: {len(analysis)} rows")
    print(f"  Time period: {analysis['year'].min()} - {analysis['year'].max()}")
    print(f"  Geographies: {', '.join(analysis['Geography'].unique())}")
    
    # Check cannabis data coverage
    cannabis_cols = [col for col in analysis.columns if 'cannabis' in col.lower()]
    if cannabis_cols:
        print(f"\n  Cannabis variables: {len(cannabis_cols)}")
        for col in ['cannabis_use_rate', 'cannabis_daily_rate']:
            if col in analysis.columns:
                coverage = analysis[col].notna().sum() / len(analysis) * 100
                print(f"    {col}: {coverage:.1f}% coverage")
    
    return analysis


# ==============================================================================
# 6. DATA VALIDATION (UPDATED)
# ==============================================================================

def validate_data(df):
    """Perform data quality checks"""
    print("\n" + "="*80)
    print("DATA VALIDATION")
    print("="*80)
    
    key_vars = [
        'consumption_total_alcoholic_beverages',
        'log_consumption_total',
        'real_alcohol_price',
        'log_real_alcohol_price',
        'real_income',
        'log_real_income',
        'share_65plus',
        'share_18_21',
        'cannabis_use_rate',  # NEW
        'cannabis_daily_rate'  # NEW
    ]
    
    print("\nMissing values in key variables:")
    available_vars = [v for v in key_vars if v in df.columns]
    missing = df[available_vars].isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(missing)
    else:
        print("  No missing values in key variables!")
    
    print("\nPotential outliers (> 3 SD from mean):")
    for col in available_vars:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[np.abs((df[col] - mean) / std) > 3]
            if len(outliers) > 0:
                print(f"  {col}: {len(outliers)} observations")
    
    print("\nSummary statistic:")
    print(df[available_vars].describe().round(3))
    
    # NEW: Cannabis-specific validation
    if 'cannabis_use_rate' in df.columns:
        print("\n" + "="*80)
        print("CANNABIS DATA VALIDATION")
        print("="*80)
        
        print("\nCannabis use by geography (2023):")
        if 2023 in df['year'].values:
            cannabis_2023 = df[df['year'] == 2023][['Geography', 'cannabis_use_rate', 'cannabis_daily_rate']]
            for _, row in cannabis_2023.iterrows():
                print(f"  {row['Geography']}: {row['cannabis_use_rate']*100:.1f}% use, {row['cannabis_daily_rate']*100:.1f}% daily")
        
        print("\nPre vs Post-legalization (2018):")
        if 'post_legalization' in df.columns:
            pre = df[df['post_legalization'] == 0]['cannabis_use_rate'].mean()
            post = df[df['post_legalization'] == 1]['cannabis_use_rate'].mean()
            print(f"  Pre-2018: {pre*100:.1f}% average use")
            print(f"  Post-2018: {post*100:.1f}% average use")
            print(f"  Change: {(post-pre)*100:+.1f} percentage points")


# ==============================================================================
# 7. EXPORT (UPDATED)
# ==============================================================================

def export_summary_stats(df):
    """Create summary statistic table"""
    print("\nExporting summary statistic...")
    
    summary_cols = {
        'consumption_total_alcoholic_beverages': ['mean', 'std', 'min', 'max'],
        'real_alcohol_price': ['mean', 'std', 'min', 'max'],
        'real_income': ['mean', 'std', 'min', 'max'],
        'share_65plus': ['mean', 'std', 'min', 'max'],
        'share_18_21': ['mean', 'std', 'min', 'max'],
        'cannabis_use_rate': ['mean', 'std', 'min', 'max'],  # NEW
        'cannabis_daily_rate': ['mean', 'std', 'min', 'max'],  # NEW
        'year': ['min', 'max', 'count']
    }
    
    existing_summary_cols = {col: aggs for col, aggs in summary_cols.items() if col in df.columns}
    
    if len(existing_summary_cols) == 0:
        print("  ⚠ Warning: No summary columns found in dataframe")
        return
    
    summary = df.groupby('Geography').agg(existing_summary_cols).round(3)
    summary.to_csv(OUTPUT_DIR / "summary_statistic_with_cannabis.csv")
    print(f"✓ Saved to {OUTPUT_DIR / 'summary_statistic_with_cannabis.csv'}")
    
    # Income distribution summary
    income_cols = [col for col in df.columns if col.startswith('income_q') or col.startswith('income_d')]
    if len(income_cols) > 0:
        income_summary = df.groupby('Geography')[income_cols].mean().round(0)
        income_summary.to_csv(OUTPUT_DIR / "income_distribution_summary.csv")
        print(f"✓ Saved to {OUTPUT_DIR / 'income_distribution_summary.csv'}")
    
    # NEW: Cannabis summary by year
    cannabis_cols = [col for col in df.columns if 'cannabis' in col.lower() and '_rate' in col]
    if len(cannabis_cols) > 0:
        cannabis_summary = df.groupby(['Geography', 'year'])[cannabis_cols].mean().round(4)
        cannabis_summary.to_csv(OUTPUT_DIR / "cannabis_prevalence_summary.csv")
        print(f"✓ Saved to {OUTPUT_DIR / 'cannabis_prevalence_summary.csv'}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("CANADIAN ALCOHOL CONSUMPTION ANALYSIS - WITH CANNABIS INTEGRATION")
    print("="*80)
    
    # Load data
    data = load_data()
    
    # Create analysis dataset
    analysis = create_analysis_dataset(data)
    
    # Validate data
    validate_data(analysis)
    
    # Export summary statistic
    export_summary_stats(analysis)
    
    # Save prepared data
    print("\nSaving prepared data...")
    analysis.to_csv(OUTPUT_DIR / "analysis_data_with.csv", index=False)
    print(f"✓ Saved to {OUTPUT_DIR / 'analysis_data_with.csv'}")
    
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    
    return analysis


if __name__ == "__main__":
    analysis = main()