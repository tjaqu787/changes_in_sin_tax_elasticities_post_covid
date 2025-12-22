import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Read the CSV file
df = pd.read_csv('./data/basket_weights.csv')

# Clean up the data
df.columns = df.columns.str.strip()

# Create the main interactive visualization
def create_interactive_cpi_chart():
    """Create an interactive line chart showing CPI basket weights over time"""
    
    # Get unique geographies and product groups
    geographies = df['Geography'].unique()
    
    # Create figure with dropdown menu for geography selection
    fig = go.Figure()
    
    # Add traces for each product group for the first geography (Canada)
    default_geo = 'Canada'
    geo_data = df[df['Geography'] == default_geo]
    
    # Get year columns (all numeric columns)
    year_columns = [col for col in df.columns if col not in ['Geography', 'Products and product groups']]
    
    # Color palette for different product categories
    colors = px.colors.qualitative.Plotly
    
    for idx, product in enumerate(geo_data['Products and product groups']):
        values = geo_data[geo_data['Products and product groups'] == product][year_columns].values[0]
        
        fig.add_trace(go.Scatter(
            x=year_columns,
            y=values,
            name=product,
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=8),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Year: %{x}<br>' +
                         'Weight: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))
    
    # Create dropdown buttons for geography selection
    buttons = []
    for geo in geographies:
        geo_data = df[df['Geography'] == geo]
        
        # Create visibility list and data updates for this geography
        button = dict(
            label=geo,
            method='update',
            args=[
                {'y': [geo_data[geo_data['Products and product groups'] == product][year_columns].values[0] 
                       for product in geo_data['Products and product groups']]},
                {'title': f'CPI Basket Weights - {geo}'}
            ]
        )
        buttons.append(button)
    
    # Update layout
    fig.update_layout(
        title=f'CPI Basket Weights - {default_geo}',
        xaxis_title='Year',
        yaxis_title='Weight (%)',
        hovermode='closest',
        template='plotly_white',
        height=700,
        updatemenus=[
            dict(
                buttons=buttons,
                direction='down',
                pad={'r': 10, 't': 10},
                showactive=True,
                x=0.11,
                xanchor='left',
                y=1.15,
                yanchor='top'
            )
        ],
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,
            font=dict(size=10)
        ),
        margin=dict(r=250)
    )
    
    return fig


# Generate all visualizations
if __name__ == '__main__':
    print("Generating CPI Basket Weights Visualizations...")
    
    # Create and save the line chart
    fig1 = create_interactive_cpi_chart()
    fig1.write_html('cpi_line_chart.html')
    print("✓ Line chart saved to cpi_line_chart.html")
    