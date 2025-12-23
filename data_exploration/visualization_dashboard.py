"""
Canadian Alcohol Consumption - Interactive Dashboard
=====================================================

Creates interactive Plotly visualizations for:
1. Decomposition results (waterfall charts)
2. Elasticity trends and comparisons
3. Geographic comparisons
4. Time series trends
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR = Path("prepared_data")
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

COLORS = {
    'primary': '#1f77b4',
    'negative': '#d62728',
    'positive': '#2ca02c',
    'neutral': '#7f7f7f',
    'baseline': '#ff7f0e',
    'accent1': '#9467bd',
    'accent2': '#8c564b',
    'accent3': '#e377c2'
}

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

def load_all_data():
    """Load analysis data and results"""
    print("Loading data and results...")
    
    data = {
        'analysis': pd.read_csv(DATA_DIR / "analysis_data.csv"),
        'decomposition': None,
        'elasticity': None
    }
    
    # Load decomposition results if available
    decomp_file = RESULTS_DIR / "decomposition_full_results.json"
    if decomp_file.exists():
        with open(decomp_file, 'r') as f:
            data['decomposition'] = json.load(f)
    
    # Load elasticity results if available
    elast_file = RESULTS_DIR / "elasticity_results.json"
    if elast_file.exists():
        with open(elast_file, 'r') as f:
            data['elasticity'] = json.load(f)
    
    print(f"✓ Loaded all data")
    return data


# ==============================================================================
# 2. CONSUMPTION TRENDS VISUALIZATION
# ==============================================================================

def create_consumption_trends(data):
    """
    Create interactive time series of alcohol consumption
    """
    df = data['analysis']
    
    fig = go.Figure()
    
    # Add trace for each geography
    for geo in df['Geography'].unique():
        geo_data = df[df['Geography'] == geo].sort_values('year')
        
        fig.add_trace(go.Scatter(
            x=geo_data['year'],
            y=geo_data['consumption_total_alcoholic_beverages'],
            name=geo,
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    # Add COVID marker
    fig.add_vline(
        x=2020, 
        line_dash="dash", 
        line_color="red",
        annotation_text="COVID-19",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Alcohol Consumption Trends (2015-2023)<br><sub>Liters of absolute alcohol per capita</sub>",
        xaxis_title="Year",
        yaxis_title="Liters per Capita",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        font=dict(size=12),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig


# ==============================================================================
# 3. DECOMPOSITION WATERFALL CHART
# ==============================================================================

def create_waterfall_chart(decomp_results, geography='Canada', model='baseline'):
    """
    Create waterfall chart showing decomposition
    """
    if decomp_results is None or model not in decomp_results:
        print(f"⚠ Decomposition results not available")
        return None
    
    result = decomp_results[model][geography]
    
    # Prepare data
    categories = []
    values = []
    measures = []
    text_labels = []
    
    # Baseline
    baseline_value = result['actual_change']
    categories.append('2015<br>Baseline')
    values.append(baseline_value)
    measures.append('absolute')
    text_labels.append(f"{baseline_value:.2f}L")
    
    # Contributions
    contrib_order = [
        ('log_real_alcohol_price', 'Price<br>Effect'),
        ('log_real_income', 'Income<br>Effect'),
        ('share_65plus', 'Demographic<br>(Aging)'),
        ('post_covid', 'COVID<br>Shock'),
        ('year_centered', 'Time<br>Trend')
    ]
    
    cumulative = baseline_value
    for var_name, display_name in contrib_order:
        if var_name in result['contributions']:
            value = result['contributions'][var_name]
            categories.append(display_name)
            values.append(value)
            measures.append('relative')
            text_labels.append(f"{value:+.2f}L")
            cumulative += value
    
    # Unexplained
    if 'unexplained' in result:
        categories.append('Unexplained<br>Residual')
        values.append(result['unexplained'])
        measures.append('relative')
        text_labels.append(f"{result['unexplained']:+.2f}L")
        cumulative += result['unexplained']
    
    # Final value
    categories.append('2023<br>Actual')
    values.append(cumulative)
    measures.append('total')
    text_labels.append(f"{cumulative:.2f}L")
    
    # Create waterfall
    fig = go.Figure(go.Waterfall(
        name="Decomposition",
        orientation="v",
        measure=measures,
        x=categories,
        textposition="outside",
        text=text_labels,
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": COLORS['negative']}},
        increasing={"marker": {"color": COLORS['positive']}},
        totals={"marker": {"color": COLORS['primary']}}
    ))
    
    fig.update_layout(
        title=f"Decomposition of Alcohol Consumption Decline<br><sub>{geography}, 2015-2023</sub>",
        showlegend=False,
        height=600,
        template='plotly_white',
        font=dict(size=12),
        yaxis_title="Change in Consumption (Liters per Capita)",
        xaxis_title=""
    )
    
    # Add explanation annotation
    explained_pct = result.get('percent_explained', 0)
    fig.add_annotation(
        text=f"Model explains {explained_pct:.1f}% of decline",
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(size=11, color="gray")
    )
    
    return fig


# ==============================================================================
# 4. ELASTICITY COMPARISON CHART
# ==============================================================================

def create_elasticity_comparison(elast_results):
    """
    Create chart comparing elasticities across geographies and periods
    """
    if elast_results is None:
        print(f"⚠ Elasticity results not available")
        return None
    
    # Extract data
    data_pre = []
    data_post = []
    geographies = []
    
    for geo, results in elast_results.items():
        if 'structural_break' in results:
            sb = results['structural_break']
            geographies.append(geo)
            
            data_pre.append({
                'geography': geo,
                'elasticity': sb['elasticity_pre_covid']['estimate'],
                'ci_lower': sb['elasticity_pre_covid']['ci_lower'],
                'ci_upper': sb['elasticity_pre_covid']['ci_upper']
            })
            
            data_post.append({
                'geography': geo,
                'elasticity': sb['elasticity_post_covid']['estimate'],
                'ci_lower': sb['elasticity_post_covid']['ci_lower'],
                'ci_upper': sb['elasticity_post_covid']['ci_upper']
            })
    
    df_pre = pd.DataFrame(data_pre)
    df_post = pd.DataFrame(data_post)
    
    # Create figure
    fig = go.Figure()
    
    # Pre-COVID
    fig.add_trace(go.Scatter(
        x=df_pre['geography'],
        y=df_pre['elasticity'],
        error_y=dict(
            type='data',
            symmetric=False,
            array=df_pre['ci_upper'] - df_pre['elasticity'],
            arrayminus=df_pre['elasticity'] - df_pre['ci_lower']
        ),
        name='Pre-COVID (2015-2019)',
        mode='markers',
        marker=dict(size=12, color=COLORS['primary'])
    ))
    
    # Post-COVID
    fig.add_trace(go.Scatter(
        x=df_post['geography'],
        y=df_post['elasticity'],
        error_y=dict(
            type='data',
            symmetric=False,
            array=df_post['ci_upper'] - df_post['elasticity'],
            arrayminus=df_post['elasticity'] - df_post['ci_lower']
        ),
        name='Post-COVID (2020-2023)',
        mode='markers',
        marker=dict(size=12, color=COLORS['negative'])
    ))
    
    fig.update_layout(
        title="Price Elasticity of Demand: Pre vs Post-COVID<br><sub>95% Confidence Intervals</sub>",
        xaxis_title="Geography",
        yaxis_title="Price Elasticity",
        height=500,
        template='plotly_white',
        font=dict(size=12),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add reference line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


# ==============================================================================
# 5. BEVERAGE-SPECIFIC ELASTICITIES
# ==============================================================================

def create_beverage_elasticities(elast_results, geography='Canada'):
    """
    Create chart comparing elasticities across beverage types
    """
    if elast_results is None or geography not in elast_results:
        print(f"⚠ Elasticity results not available for {geography}")
        return None
    
    if 'beverage_elasticities' not in elast_results[geography]:
        print(f"⚠ Beverage elasticities not available for {geography}")
        return None
    
    bev_results = elast_results[geography]['beverage_elasticities']
    
    if 'beverages' not in bev_results or len(bev_results['beverages']) == 0:
        print(f"⚠ No beverage data for {geography}")
        return None
    
    # Extract data
    beverages = []
    elasticities = []
    ci_lower = []
    ci_upper = []
    
    for bev_name, bev_data in bev_results['beverages'].items():
        beverages.append(bev_name)
        elasticities.append(bev_data['elasticity'])
        ci_lower.append(bev_data['ci_lower'])
        ci_upper.append(bev_data['ci_upper'])
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=beverages,
        y=elasticities,
        error_y=dict(
            type='data',
            symmetric=False,
            array=np.array(ci_upper) - np.array(elasticities),
            arrayminus=np.array(elasticities) - np.array(ci_lower)
        ),
        marker_color=[COLORS['primary'] if e < 0 else COLORS['positive'] for e in elasticities],
        text=[f"{e:.2f}" for e in elasticities],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Price Elasticity by Beverage Type<br><sub>{geography}, 95% Confidence Intervals</sub>",
        xaxis_title="Beverage Type",
        yaxis_title="Price Elasticity",
        height=500,
        template='plotly_white',
        font=dict(size=12),
        showlegend=False
    )
    
    # Add reference line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


# ==============================================================================
# 6. REAL PRICE TRENDS
# ==============================================================================

def create_price_trends(data):
    """
    Create chart showing real alcohol price trends
    """
    df = data['analysis']
    
    fig = go.Figure()
    
    # Add trace for each geography
    for geo in df['Geography'].unique():
        geo_data = df[df['Geography'] == geo].sort_values('year')
        
        fig.add_trace(go.Scatter(
            x=geo_data['year'],
            y=geo_data['real_alcohol_price_index'],
            name=geo,
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    # Add COVID marker
    fig.add_vline(
        x=2020, 
        line_dash="dash", 
        line_color="red",
        annotation_text="COVID-19",
        annotation_position="top"
    )
    
    # Add reference line at 100
    fig.add_hline(
        y=100, 
        line_dash="dot", 
        line_color="gray",
        opacity=0.5
    )
    
    fig.update_layout(
        title="Real Alcohol Price Index (2015=100)<br><sub>Relative to overall CPI</sub>",
        xaxis_title="Year",
        yaxis_title="Real Price Index",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        font=dict(size=12),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig


# ==============================================================================
# 7. INCOME VS CONSUMPTION SCATTER
# ==============================================================================

def create_income_consumption_scatter(data):
    """
    Create scatter plot of real income vs consumption
    """
    df = data['analysis'].copy()
    
    # Add period label
    df['Period'] = df['year'].apply(lambda x: 'Post-COVID (2020+)' if x >= 2020 else 'Pre-COVID (2015-2019)')
    
    fig = px.scatter(
        df,
        x='real_income',
        y='consumption_total_alcoholic_beverages',
        color='Geography',
        symbol='Period',
        size='share_65plus',
        hover_data=['year', 'share_65plus'],
        trendline='ols'
    )
    
    fig.update_layout(
        title="Real Income vs Alcohol Consumption<br><sub>Bubble size = % population 65+</sub>",
        xaxis_title="Real Income per Capita",
        yaxis_title="Consumption (Liters per Capita)",
        height=600,
        template='plotly_white',
        font=dict(size=12)
    )
    
    return fig


# ==============================================================================
# 8. AGE STRUCTURE EVOLUTION
# ==============================================================================

def create_age_structure_chart(data):
    """
    Create stacked area chart showing age structure evolution
    """
    df = data['analysis']
    
    fig = go.Figure()
    
    # For each geography
    for geo in df['Geography'].unique():
        geo_data = df[df['Geography'] == geo].sort_values('year')
        
        fig.add_trace(go.Scatter(
            x=geo_data['year'],
            y=geo_data['share_65plus'] * 100,
            name=geo,
            mode='lines',
            stackgroup='one',
            line=dict(width=0.5),
            fillcolor=f"rgba{tuple(list(int(COLORS['primary'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.5])}"
        ))
    
    fig.update_layout(
        title="Population Aging Trends<br><sub>Share of population 65+</sub>",
        xaxis_title="Year",
        yaxis_title="Percent of Population (%)",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        font=dict(size=12)
    )
    
    return fig


# ==============================================================================
# 9. GEOGRAPHIC COMPARISON MAP
# ==============================================================================

def create_geographic_comparison(data, year=2023):
    """
    Create map comparing consumption across provinces
    """
    df = data['analysis']
    df_year = df[df['year'] == year].copy()
    
    # Create bar chart (since we don't have geographic coordinates)
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_year['Geography'],
        y=df_year['consumption_total_alcoholic_beverages'],
        marker_color=COLORS['primary'],
        text=[f"{v:.2f}L" for v in df_year['consumption_total_alcoholic_beverages']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Alcohol Consumption by Province ({year})<br><sub>Liters of absolute alcohol per capita</sub>",
        xaxis_title="Province",
        yaxis_title="Consumption (Liters per Capita)",
        height=500,
        template='plotly_white',
        font=dict(size=12),
        showlegend=False
    )
    
    return fig


# ==============================================================================
# 10. COMPREHENSIVE DASHBOARD
# ==============================================================================

def create_dashboard(data):
    """
    Create comprehensive multi-panel dashboard
    """
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Consumption Trends",
            "Real Price Trends",
            "Decomposition (Canada)",
            "Elasticity Pre vs Post-COVID",
            "Income vs Consumption",
            "Population Aging"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # This is a simplified version - in practice, you'd add all the traces
    # For now, we'll create individual charts
    
    print("Creating individual charts (dashboard mode not fully implemented)")
    
    return None


# ==============================================================================
# 11. EXPORT ALL VISUALIZATIONS
# ==============================================================================

def export_all_visualizations(data):
    """
    Create and export all visualizations
    """
    print("="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    visualizations = {}
    
    # 1. Consumption trends
    print("\n1. Creating consumption trends chart...")
    fig = create_consumption_trends(data)
    if fig:
        fig.write_html(OUTPUT_DIR / "01_consumption_trends.html")
        visualizations['consumption_trends'] = fig
        print("   ✓ Saved to 01_consumption_trends.html")
    
    # 2. Real price trends
    print("\n2. Creating real price trends chart...")
    fig = create_price_trends(data)
    if fig:
        fig.write_html(OUTPUT_DIR / "02_price_trends.html")
        visualizations['price_trends'] = fig
        print("   ✓ Saved to 02_price_trends.html")
    
    # 3. Decomposition waterfall (Canada)
    print("\n3. Creating decomposition waterfall chart...")
    if data['decomposition']:
        fig = create_waterfall_chart(data['decomposition'], 'Canada', 'baseline')
        if fig:
            fig.write_html(OUTPUT_DIR / "03_decomposition_waterfall.html")
            visualizations['decomposition'] = fig
            print("   ✓ Saved to 03_decomposition_waterfall.html")
    
    # 4. Elasticity comparison
    print("\n4. Creating elasticity comparison chart...")
    if data['elasticity']:
        fig = create_elasticity_comparison(data['elasticity'])
        if fig:
            fig.write_html(OUTPUT_DIR / "04_elasticity_comparison.html")
            visualizations['elasticity_comparison'] = fig
            print("   ✓ Saved to 04_elasticity_comparison.html")
    
    # 5. Beverage elasticities
    print("\n5. Creating beverage elasticities chart...")
    if data['elasticity']:
        fig = create_beverage_elasticities(data['elasticity'], 'Canada')
        if fig:
            fig.write_html(OUTPUT_DIR / "05_beverage_elasticities.html")
            visualizations['beverage_elasticities'] = fig
            print("   ✓ Saved to 05_beverage_elasticities.html")
    
    # 6. Income vs consumption
    print("\n6. Creating income-consumption scatter...")
    fig = create_income_consumption_scatter(data)
    if fig:
        fig.write_html(OUTPUT_DIR / "06_income_consumption_scatter.html")
        visualizations['income_scatter'] = fig
        print("   ✓ Saved to 06_income_consumption_scatter.html")
    
    # 7. Age structure
    print("\n7. Creating age structure chart...")
    fig = create_age_structure_chart(data)
    if fig:
        fig.write_html(OUTPUT_DIR / "07_age_structure.html")
        visualizations['age_structure'] = fig
        print("   ✓ Saved to 07_age_structure.html")
    
    # 8. Geographic comparison
    print("\n8. Creating geographic comparison...")
    fig = create_geographic_comparison(data, 2023)
    if fig:
        fig.write_html(OUTPUT_DIR / "08_geographic_comparison.html")
        visualizations['geographic'] = fig
        print("   ✓ Saved to 08_geographic_comparison.html")
    
    return visualizations


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main execution function
    """
    print("="*80)
    print("INTERACTIVE VISUALIZATION DASHBOARD")
    print("="*80)
    
    # Load data
    data = load_all_data()
    
    # Create visualizations
    visualizations = export_all_visualizations(data)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nVisualizations saved to: {OUTPUT_DIR}")
    print(f"\nOpen the HTML files in your browser to view interactive charts.")
    print(f"\nAvailable charts:")
    for i, (name, fig) in enumerate(visualizations.items(), 1):
        if fig:
            print(f"  {i}. {name}")
    
    return visualizations


if __name__ == "__main__":
    visualizations = main()
