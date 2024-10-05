# Required imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering as agclus, KMeans as kmclus
from sklearn.metrics import silhouette_score as sscore, davies_bouldin_score as dbscore
import scipy.stats as sps
from unidecode import unidecode
import streamlit as st
import plotly.express as px

# Load the dataset (replace with the actual file path)
file_id = '1ZV-5OFM6glnv9Eji_0mb7CyvPGjR_EPg'
file_path = f'https://drive.google.com/uc?id={file_id}'
df = pd.read_csv(file_path)

# Drop unnecessary columns
columns_to_drop = [
    'player_id', 'player_url', 'fifa_update', 'fifa_update_date', 'long_name', 'dob', 'league_id', 'league_level',
    'club_team_id', 'club_position', 'club_jersey_number', 'club_loaned_from', 'club_joined_date',
    'club_contract_valid_until_year', 'nationality_id', 'nation_team_id', 'nation_jersey_number', 'player_face_url'
]
data = df.drop(columns=columns_to_drop)

# Create a sample of the data
df1 = data.sample(n=25000, random_state=45005)
df2 = df1.copy()

# Function to split a column into multiple columns
def split_column(df, column_name):
    if column_name in df.columns:
        split_df = df[column_name].str.split(', ', expand=True)
        split_df.columns = [f'{column_name}_{i+1}' for i in range(split_df.shape[1])]
        return pd.concat([df, split_df], axis=1).drop(columns=[column_name])

# Apply the split function
df2 = split_column(df2, 'player_positions')
df2 = split_column(df2, 'player_tags')
df2 = split_column(df2, 'player_traits')

# Clean text function
def clean_text(text):
    if pd.isna(text):
        return text
    return unidecode(text)

# Apply cleaning
df2['short_name'] = df2['short_name'].apply(clean_text)
df2['club_name'] = df2['club_name'].apply(clean_text)
df2['league_name'] = df2['league_name'].apply(clean_text)
df2['nationality_name'] = df2['nationality_name'].apply(clean_text)

# Sum values in specific columns
def sum_values(value):
    if pd.isna(value):
        return value
    value = str(value)
    parts = value.split('+')
    if len(parts) == 2:
        return int(parts[0]) + int(parts[1])
    return value

columns_to_sum = [
    'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm',
    'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk'
]

for column in columns_to_sum:
    df2[f'{column}_sum'] = df2[column].apply(sum_values)

# Reordering columns
df2['index'] = range(1, len(df2) + 1)
cols = ['index'] + [col for col in df2.columns if col != 'index']
df2 = df2[cols]

# Drop unnecessary columns
columns_to_drop1 = [
    'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 
    'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk'
]
df2 = df2.drop(columns=columns_to_drop1)

# Convert to numeric and correct column types
numeric_columns = [
    'ls_sum', 'st_sum', 'rs_sum', 'lw_sum', 'lf_sum', 'cf_sum', 'rf_sum', 'rw_sum',
    'lam_sum', 'cam_sum', 'ram_sum', 'lm_sum', 'lcm_sum', 'cm_sum', 'rcm_sum', 'rm_sum',
    'lwb_sum', 'ldm_sum', 'cdm_sum', 'rdm_sum', 'rwb_sum', 'lb_sum', 'lcb_sum', 'cb_sum',
    'rcb_sum', 'rb_sum', 'gk_sum'
]

for column in numeric_columns:
    df2[column] = pd.to_numeric(df2[column], errors='coerce')

df2 = df2.astype({
    'ls_sum': 'float64', 'st_sum': 'float64', 'rs_sum': 'float64', 'lw_sum': 'float64',
    'lf_sum': 'float64', 'cf_sum': 'float64', 'rf_sum': 'float64', 'rw_sum': 'float64',
    'lam_sum': 'float64', 'cam_sum': 'float64', 'ram_sum': 'float64', 'lm_sum': 'float64',
    'lcm_sum': 'float64', 'cm_sum': 'float64', 'rcm_sum': 'float64', 'rm_sum': 'float64',
    'lwb_sum': 'float64', 'ldm_sum': 'float64', 'cdm_sum': 'float64', 'rdm_sum': 'float64',
    'rwb_sum': 'float64', 'lb_sum': 'float64', 'lcb_sum': 'float64', 'cb_sum': 'float64',
    'rcb_sum': 'float64', 'rb_sum': 'float64', 'gk_sum': 'int64'
})

# Streamlit Dashboard code
st.title("FIFA Player Stats Dashboard")

# FIFA version dropdown
fifa_version = st.selectbox('Select FIFA Version', sorted(df2['fifa_version'].unique()))

# Player positions dropdown
if fifa_version:
    player_positions = sorted(df2[df2['fifa_version'] == fifa_version]['player_positions_1'].dropna().unique())
    position = st.selectbox('Select Player Position', player_positions)

# League dropdown
if position:
    leagues = sorted(df2[(df2['fifa_version'] == fifa_version) & (df2['player_positions_1'] == position)]['league_name'].dropna().unique())
    league = st.selectbox('Select League Name', leagues)

# Club dropdown
if league:
    clubs = sorted(df2[(df2['fifa_version'] == fifa_version) & (df2['player_positions_1'] == position) & (df2['league_name'] == league)]['club_name'].dropna().unique())
    club = st.selectbox('Select Club Name', clubs)

# Player bar chart
if club:
    filtered_df = df2[(df2['fifa_version'] == fifa_version) & 
                      (df2['player_positions_1'] == position) & 
                      (df2['league_name'] == league) & 
                      (df2['club_name'] == club)]
    
    if not filtered_df.empty:
        fig = px.bar(filtered_df, x='short_name', y=['overall', 'potential', 'value_eur', 'age'], barmode='group', labels={'value': 'Value (€)'})
        st.plotly_chart(fig)

# Wage range treemap
wage_ranges = [(0, 100000), (100001, 200000), (200001, 300000), (300001, 400000), (400001, 500000)]
wage_labels = [f"{low:,} - {high:,}" for low, high in wage_ranges]
wage_range = st.selectbox('Select Wage Range', wage_labels)

if wage_range:
    low, high = wage_ranges[wage_labels.index(wage_range)]
    wage_filtered_df = df2[(df2['fifa_version'] == fifa_version) & (df2['wage_eur'] >= low) & (df2['wage_eur'] <= high)]
    
    if not wage_filtered_df.empty:
        treemap_fig = px.treemap(wage_filtered_df, path=['player_positions_1'], values='wage_eur', title='Player Positions Treemap')
        st.plotly_chart(treemap_fig)
