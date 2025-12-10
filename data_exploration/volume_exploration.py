import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the CSV file
df = pd.read_csv("data/bc_ab_mb_on_consuption.csv")

print(df)
# Rename columns for convenience and clarity
df.rename(columns={
    'Type of beverage': 'Beverage Type',
    'Value, volume and absolute volume': 'Metric'
}, inplace=True)

# Identify year columns
year_cols = df.columns[3:].tolist()

# 1. Melt the DataFrame to long format
df_long = df.melt(
    id_vars=['Geography', 'Beverage Type', 'Metric'],
    value_vars=year_cols,
    var_name='Year',
    value_name='Value'
)

# Convert 'Year' to a string category, but ensure order is maintained
df_long['Year'] = pd.Categorical(df_long['Year'], categories=year_cols, ordered=True)

# 2. Calculate the indexed value (percent change from 2015 / 2016)
def calculate_index(series):
    """Calculates percent change from the first value in the series."""
    first_value = series.iloc[0]
    # To handle division by zero, especially for new categories, we'll check the first value.
    if first_value == 0 or pd.isna(first_value):
        # Return 0% change for all if the base is zero or null (though unlikely here)
        return pd.Series([0.0] * len(series), index=series.index)
    return ((series / first_value) - 1) * 100

# Group by the three categorical columns and apply the index calculation
df_long['Indexed Value'] = df_long.groupby(['Geography', 'Beverage Type', 'Metric'])['Value'].transform(calculate_index)

# 3. Create the Plotly Figure

# Get unique values for dropdowns
geography_options = sorted(df_long['Geography'].unique())
beverage_type_options = sorted(df_long['Beverage Type'].unique())
metric_options = sorted(df_long['Metric'].unique())

# Start with a full dataset (all lines visible)
fig = go.Figure()

# Add traces for all unique combinations. We'll control visibility later.
data_combinations = df_long[['Geography', 'Beverage Type', 'Metric']].drop_duplicates().to_records(index=False)

for i, (geo, bev, met) in enumerate(data_combinations):
    df_subset = df_long[(df_long['Geography'] == geo) & (df_long['Beverage Type'] == bev) & (df_long['Metric'] == met)]
    
    # Raw Value Trace
    fig.add_trace(go.Scatter(
        x=df_subset['Year'].astype(str),
        y=df_subset['Value'],
        mode='lines+markers',
        name=f'{geo} - {bev} - {met}',
        customdata=df_subset[['Indexed Value']], # Store indexed values here
        line=dict(shape='spline'),
        visible=True, # Start with all visible, filtering will hide them
        # Group to allow for easier index/value switch
        meta=f'{geo}|{bev}|{met}',
        hovertemplate='<b>%{meta}</b><br>Year: %{x}<br>Value: %{y:,.2f}<extra></extra>',
        # Set a unique key for the trace, helpful for updatemenus
        uid=f'raw_trace_{i}'
    ))

# Helper function to generate a trace-visibility-setting list for a given set of filters
def get_visibility_settings(geo_filter, bev_filter, met_filter):
    """Generates the visibility list for traces based on filters."""
    visibility = []
    
    for trace in fig.data:
        # Parse the unique identifier from the trace's meta data
        geo, bev, met = trace.meta.split('|')
        
        # Check if the trace matches the current filter settings
        geo_match = (geo_filter == 'All') or (geo == geo_filter)
        bev_match = (bev_filter == 'All') or (bev == bev_filter)
        met_match = (met_filter == 'All') or (met == met_filter)
        
        visibility.append(geo_match and bev_match and met_match)
    
    return visibility

# 4. Implement Dropdown Filters
def create_dropdown(options, label, axis_id, x_pos):
    # Add an 'All' option for the default state
    menu_options = [{'label': 'All', 'method': 'restyle', 'args': ['visible', get_visibility_settings('All', 'All', 'All')]}]
    
    for option in options:
        # For simplicity, each dropdown independently filters (you'd need JS for combined filtering)
        if axis_id == 'geo':
            visibility = get_visibility_settings(option, 'All', 'All')
        elif axis_id == 'bev':
            visibility = get_visibility_settings('All', option, 'All')
        else:  # met
            visibility = get_visibility_settings('All', 'All', option)
        
        menu_options.append({
            'label': option,
            'method': 'restyle',
            'args': ['visible', visibility]
        })
        
    return dict(
        buttons=menu_options,
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=x_pos,
        y=1.15,
        xanchor="left",
        yanchor="top",
        bgcolor="white",
        bordercolor="lightgray",
        borderwidth=1,
        font=dict(size=10)
    )

# 5. Create the mode switch button (Raw Value vs. Indexed Value)
value_button = dict(
    method='update',
    label='Raw Value',
    args=[
        {
            'y': [[trace.y] for trace in fig.data],
            'hovertemplate': ['<b>%{meta}</b><br>Year: %{x}<br>Value: %{y:,.2f}<extra></extra>'] * len(fig.data)
        },
        {'yaxis.title.text': 'Value', 'yaxis.tickformat': ',', 'yaxis.zeroline': False}
    ]
)

index_button = dict(
    method='update',
    label='Percent Change (Index)',
    args=[
        {
            'y': [[trace.customdata[:, 0]] for trace in fig.data],
            'hovertemplate': ['<b>%{meta}</b><br>Year: %{x}<br>Change: %{y:.2f}%<extra></extra>'] * len(fig.data)
        },
        {'yaxis.title.text': 'Indexed Value (Percent Change from 2015/2016)', 
         'yaxis.tickformat': '.1f', 
         'yaxis.zeroline': True, 
         'yaxis.zerolinewidth': 1, 
         'yaxis.zerolinecolor': 'lightgray'}
    ]
)

# --- Configure Layout with Dropdowns and Buttons ---
fig.update_layout(
    title_text='Consumption Data: Raw Value',
    yaxis=dict(title='Value', tickformat=',', zeroline=False),
    xaxis=dict(title='Year'),
    hovermode='closest',
    updatemenus=[
        # Mode Switch Buttons (Raw Value / Indexed Value)
        dict(
            type="buttons",
            direction="left",
            buttons=[value_button, index_button],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.05,
            y=1.25,
            xanchor="right",
            yanchor="top",
            bgcolor="white",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=11, weight='bold')
        ),
        # Geography Dropdown
        create_dropdown(geography_options, 'Geography', 'geo', 0.0),
        # Beverage Type Dropdown
        create_dropdown(beverage_type_options, 'Beverage Type', 'bev', 0.25),
        # Metric Dropdown
        create_dropdown(metric_options, 'Metric', 'met', 0.55),
    ],
    autosize=True,
    legend_title_text='Geography - Type - Metric',
    margin=dict(t=150, r=200),
    height=700
)

# Save as interactive HTML file
fig.write_html('consumption_data_plot.html', include_plotlyjs='cdn')

print("Plotly HTML file 'consumption_data_plot.html' created successfully!")
print("Open this file in your web browser to view the interactive visualization.")