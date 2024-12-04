# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 2: Load the dataset
url = "https://raw.githubusercontent.com/chendaniely/scipy-2017-tutorial-pandas/refs/heads/master/data/gapminder.tsv"
gapminder_df = pd.read_csv(url, sep='\t')

# Step 3: Display rows 1550 to 1560
subset_rows_1550_1560 = gapminder_df.iloc[1550:1561]
print("Rows 1550 to 1560:")
print(subset_rows_1550_1560)

# Step 4: Filter rows for the country of birth
country_of_birth = "India"  # Example: Change to your country
country_df = gapminder_df[gapminder_df['country'] == country_of_birth]
print(f"Rows for {country_of_birth}:")
print(country_df)

# Step 5: Filter rows for the year closest to the year of birth
year_of_birth = 1990  # Example: Change to your year of birth
closest_year = gapminder_df['year'].iloc[(gapminder_df['year'] - year_of_birth).abs().argsort()].iloc[0]
year_df = gapminder_df[gapminder_df['year'] == closest_year]
print(f"Rows for the year closest to {year_of_birth} ({closest_year}):")
print(year_df)

# Step 6: Create scatterplot of lifeExp vs gdpPerCap
plt.figure(figsize=(10, 6))
plt.scatter(year_df['gdpPercap'], year_df['lifeExp'], color='blue', alpha=0.7)
plt.title(f'Life Expectancy vs GDP Per Capita ({closest_year})')
plt.xlabel('GDP Per Capita')
plt.ylabel('Life Expectancy')
plt.grid(True)
plt.show()

# Step 7: Regression analysis
# Prepare data for regression
X = year_df[['gdpPercap']]
y = year_df['lifeExp']

# Perform linear regression
regressor = LinearRegression()
regressor.fit(X, y)

# Plot regression line
plt.figure(figsize=(10, 6))
plt.scatter(year_df['gdpPercap'], year_df['lifeExp'], color='blue', alpha=0.7, label='Data')
plt.plot(year_df['gdpPercap'], regressor.predict(X), color='red', label='Regression Line')
plt.title(f'Regression of Life Expectancy on GDP Per Capita ({closest_year})')
plt.xlabel('GDP Per Capita')
plt.ylabel('Life Expectancy')
plt.legend()
plt.grid(True)
plt.show()

# Regression results
print("Regression Coefficients:")
print(f"Slope (Effect of GDP Per Capita): {regressor.coef_[0]:.2f}")
print(f"Intercept (Base Life Expectancy): {regressor.intercept_:.2f}")
