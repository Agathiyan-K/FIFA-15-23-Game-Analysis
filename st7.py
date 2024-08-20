import streamlit as st
import plotly.express as px
import pandas as pd
import os
import matplotlib.pyplot as plt
import altair as a

# Setting the Page configuration
st.set_page_config(page_title="FIFA Player Stats Dashboard", page_icon=":bar_chart:", layout="wide")

# Setting the dashboard title
st.title(":bar_chart: FIFA Player Stats Dashboard")

# Make the title appear above
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

data_url="final.csv"
df = pd.read_csv(data_url)

# Display the DataFrame
st.subheader("Data Preview:")
st.dataframe(df)






# Sample DataFrame for demonstration (use your df here)
# df = pd.read_csv('your_data.csv')  # Ensure to use your actual data

# Define wage ranges
wage_ranges = [
    (0, 100000), (100001, 200000), (200001, 300000), (300001, 400000),
    (400001, 500000), (500001, 600000), (600001, 700000), (700001, 800000),
    (800001, 900000), (900001, 1000000)
]
wage_range_labels = [f"{low:,} - {high:,}" for low, high in wage_ranges]

# Create options for player tags
player_tags_columns = ['player_tags_1', 'player_tags_2', 'player_tags_3',
                       'player_tags_4', 'player_tags_5', 'player_tags_6', 'player_tags_7']
player_tags = sorted(set(tag for col in player_tags_columns for tag in df[col].dropna().unique()))
player_tags_options = [tag for tag in player_tags]

# Sidebar for FIFA version selection
st.sidebar.header("Filters")
fifa_version = st.sidebar.selectbox(
    "Select FIFA Version",
    sorted(df['fifa_version'].unique())
)

# Filtered DataFrame based on FIFA version
df_fifa = df[df['fifa_version'] == fifa_version] if fifa_version else df

# Player position selection
player_positions = sorted(df_fifa['player_positions_1'].dropna().unique())
player_position = st.sidebar.selectbox(
    "Select Player Position",
    player_positions
)

# League name selection
leagues = sorted(df_fifa[df_fifa['player_positions_1'] == player_position]['league_name'].dropna().unique()) if player_position else []
league_name = st.sidebar.selectbox(
    "Select League Name",
    leagues
)

# Club name selection
clubs = sorted(df_fifa[(df_fifa['player_positions_1'] == player_position) & (df_fifa['league_name'] == league_name)]['club_name'].dropna().unique()) if league_name else []
club_name = st.sidebar.selectbox(
    "Select Club Name",
    clubs
)

# First Visualization: Player Position Bar Chart
if club_name:
    filtered_df = df_fifa[(df_fifa['player_positions_1'] == player_position) &
                          (df_fifa['league_name'] == league_name) &
                          (df_fifa['club_name'] == club_name)]
    
    if not filtered_df.empty:
        fig = px.bar(filtered_df, x='short_name', y=['overall', 'potential', 'value_eur', 'age'],
                     barmode='group', labels={'value': 'Value (€)'})
        st.subheader("Player Stats by Position")
        st.plotly_chart(fig)

# Second Visualization: Wage Range Treemap
wage_range_index = st.sidebar.selectbox(
    "Select Wage Range",
    list(range(len(wage_range_labels))),
    format_func=lambda x: wage_range_labels[x]
)

low, high = wage_ranges[wage_range_index]
filtered_df_wage = df_fifa[(df_fifa['wage_eur'] >= low) & (df_fifa['wage_eur'] <= high)]

if not filtered_df_wage.empty:
    fig = px.treemap(
        filtered_df_wage,
        path=['player_positions_1'],
        values='wage_eur',
        title='Player Positions Treemap'
    )
    st.subheader("Player Stats by Wage Range")
    st.plotly_chart(fig)

    # Display player details
    selected_position = st.selectbox("Select Player Position from Treemap", sorted(filtered_df_wage['player_positions_1'].unique()))
    filtered_details = filtered_df_wage[filtered_df_wage['player_positions_1'] == selected_position]
    
    if not filtered_details.empty:
        st.write("Player Details:")
        st.dataframe(filtered_details[['short_name', 'player_positions_1', 'nationality_name', 'club_name', 'wage_eur', 'value_eur', 'overall', 'potential', 'age']])

# Third Visualization: Player Tags Bar Chart
selected_tags = st.sidebar.multiselect(
    "Select Player Tags",
    options=player_tags_options
)

if selected_tags:
    filtered_df_tags = df[df[player_tags_columns].isin(selected_tags).any(axis=1)]
    players = sorted(filtered_df_tags['short_name'].unique())
    selected_players = st.sidebar.multiselect(
        "Select Player Names",
        options=players
    )
    
    if selected_players:
        filtered_df_players = filtered_df_tags[filtered_df_tags['short_name'].isin(selected_players)]
        fig = px.bar(filtered_df_players, x='short_name', y=['overall', 'potential', 'value_eur', 'age'],
                     barmode='group', labels={'value': 'Value (€)'})
        st.subheader("Player Stats by Tags")
        st.plotly_chart(fig)


import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import altair as alt

# Clean DataFrame (assuming df is defined elsewhere in your code)
df_clean = df[['value_eur', 'overall', 'potential', 'age', 'wage_eur', 'international_reputation', 'player_positions_1', 'skill_moves']].dropna()

# One-hot encoding
df_clean = pd.get_dummies(df_clean, columns=['player_positions_1'], drop_first=True)

# Define the regression model
model = smf.ols('value_eur ~ overall + potential + age + wage_eur + international_reputation + skill_moves + ' + 
                ' + '.join(df_clean.columns[df_clean.columns.str.startswith('player_positions_1')]), data=df_clean).fit()

# Streamlit app
st.title('Player Value Prediction')

# Input fields for the player's details
overall = st.number_input('Overall', min_value=0, max_value=100, value=75)
potential = st.number_input('Potential', min_value=0, max_value=100, value=80)
age = st.number_input('Age', min_value=16, max_value=50, value=25)
wage_eur = st.number_input('Wage (EUR)', min_value=0, value=50000)
international_reputation = st.selectbox('International Reputation', options=[1, 2, 3, 4, 5])
skill_moves = st.selectbox('Skill Moves', options=[1, 2, 3, 4, 5])

# Dropdown for player position (one-hot encoded)
player_position = st.selectbox('Player Position', options=df['player_positions_1'].unique())

# Create a DataFrame for prediction based on the input values
input_data = pd.DataFrame({
    'overall': [overall],
    'potential': [potential],
    'age': [age],
    'wage_eur': [wage_eur],
    'international_reputation': [international_reputation],
    'skill_moves': [skill_moves],
    **{f'player_positions_1_{pos}': [1 if player_position == pos else 0] for pos in df['player_positions_1'].unique() if pos != df['player_positions_1'].unique()[0]}
})

# Align input_data columns with df_clean columns (fill missing columns with 0)
input_data = input_data.reindex(columns=model.model.exog_names, fill_value=0)

# Button to trigger prediction
if st.button('Predict Value (EUR)'):
    predicted_value = model.predict(input_data)[0]
    st.write(f"Predicted Value (EUR): {predicted_value:,.2f}")
    
    # Prepare data for chart
    df_chart = pd.DataFrame({
        'Type': ['Actual Values', 'Predicted Value'],
        'Value (EUR)': [df_clean['value_eur'].mean(), predicted_value]
    })

    # Create a bar chart to compare actual vs. predicted value
    chart = alt.Chart(df_chart).mark_bar().encode(
        x='Type',
        y='Value (EUR)',
        color='Type'
    ).properties(
        width=alt.Step(80)  # Controls the width of the bars
    )

    st.altair_chart(chart)
