import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Data Science with Python\global_inflation_post_covid.csv")

print(df.head())
print(df.info())

# Check missing values
print(df.isnull().sum())

# Drop duplicates
df = df.drop_duplicates()

# Fill missing values (numerical)
df = df.fillna(df.mean(numeric_only=True))

# If categorical present
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Inflation change year-wise
df['Inflation_Change'] = df.groupby('country')['inflation_rate'].diff()

# Volatility (std per country)
volatility = df.groupby('country')['inflation_rate'].std().reset_index()
volatility.rename(columns={'inflation_rate': 'Volatility'}, inplace=True)

df = df.merge(volatility, on='country', how='left')

plt.figure(figsize=(12,6))
sns.boxplot(x='inflation_rate', data=df)
plt.title("Inflation Distribution")
plt.show()


import plotly.express as px

# Create an interactive histogram using Plotly
fig = px.histogram(
    df,
    x='inflation_rate',
    title='Inflation Distribution',
    labels={'inflation_rate': 'Inflation Rate'},
    color_discrete_sequence=['steelblue'],
    template='ggplot2',
    marginal='box', # Added marginal box plot
    nbins=250# Adjusted number of bins for more detail
)

fig.show()

# 1. Inflation Trends Over Time
# We'll aggregate by date to see the global average trend
df_trend = df.groupby('date')['inflation_rate'].mean().reset_index()

fig_line = px.line(
    df_trend,
    x='date',
    y='inflation_rate',
    title='Global Average Inflation Trend (Post-COVID)',
    markers=True,
    template='plotly_white'
)
fig_line.show()

# 2. Correlation Heatmap
# This helps understand which factors are most closely linked to inflation
corr = df.select_dtypes(include=[np.number]).corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu_r',
    title='Correlation Heatmap of Economic Indicators'
)
fig_corr.show()

# 3. Oil Price vs Inflation
# Taking a sample of the data to keep the interactive plot responsive
fig_scatter = px.scatter(
    df.sample(1000),
    x='oil_price',
    y='inflation_rate',
    color='gdp_growth',
    hover_data=['country'],
    title='Impact of Oil Prices on Inflation',
)
fig_scatter.show()

# 4. Country-wise Average Inflation
# Aggregating by country to see the mean inflation rate per nation
df_country = df.groupby('country')['inflation_rate'].mean().sort_values(ascending=False).reset_index()

fig_bar = px.bar(
    df_country,
    x='country',
    y='inflation_rate',
    title='Average Inflation Rate by Country',
    color='inflation_rate',
    color_continuous_scale='Viridis',
    template='plotly_white'
)

fig_bar.update_layout(xaxis_tickangle=-45)
fig_bar.show()

# 5. Interactive Box Plot for Outliers
fig_box = px.box(
    df,
    x='inflation_rate',
    title='Distribution and Outliers of Inflation Rate',
    points='outliers',
    template='plotly_white',
    labels={'inflation_rate': 'Inflation Rate'}
)

fig_box.show()

from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
le = LabelEncoder()

try:
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
except NameError:
    print("Error: DataFrame 'df' is not defined. Please ensure all previous cells that load and preprocess the data have been executed.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

    X = df.drop('inflation_rate', axis=1)
y = df['inflation_rate']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fix: Handle NaNs introduced by the .diff() operation in previous cells
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 6. Improved Actual vs Predicted Comparison
# Creating a DataFrame for visualization
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Sample data if it's too large to maintain responsiveness and reduce overlap
plot_df = results_df.sample(n=min(2000, len(results_df)), random_state=42)

# Create the base scatter plot
fig_compare = px.scatter(
    plot_df,
    x='Actual',
    y='Predicted',
    opacity=0.6,
    trendline="ols",
    title='Model Performance: How Close are Predictions to Reality?',
    labels={'Actual': 'Actual Inflation Rate (%)', 'Predicted': 'Predicted Inflation Rate (%)'},
    template='plotly_white',
    hover_data={'Actual': ':.2f', 'Predicted': ':.2f'}
)

# Add a 45-degree line representing the 'Perfect Prediction' line
# If a point falls on this line, the prediction was 100% accurate
min_val = min(plot_df['Actual'].min(), plot_df['Predicted'].min())
max_val = max(plot_df['Actual'].max(), plot_df['Predicted'].max())

fig_compare.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(dash='dash', color='red', width=2),
        name='Perfect Prediction Line'
    )
)

fig_compare.update_layout(
    annotations=[
        dict(
            x=max_val, y=max_val, 
            xref="x", yref="y",
            text="Ideal Match",
            showarrow=True,
            arrowhead=2,
            ax=-40, ay=-40
        )
    ],
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig_compare.show()


