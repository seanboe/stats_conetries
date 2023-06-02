import pandas as pd
import numpy as np  
import scipy.stats as st
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

plt.style.use('ggplot')

df = pd.read_csv('~/Documents/School/Stats/dataproject/test1/dataframes/countries-of-the-world.csv')

def main():
  # Define a dictionary containing employee data
  # data = {'Name':['Jai', 'Ben', 'Gaurav', 'Ben'], 
  #         'Age':[1, 12, 3, 14]}

  # df = pd.DataFrame(data)

  # df.iloc[0, 1] = 50
  # print(df)

  # print(type(df.query('Name == \'Ben\'')))
  # print(df.query('Name == \'Ben\''))

  dropped_cols = ['Arable (%)', 
                  'Other (%)', 
                  'Climate', 
                  'Agriculture', 
                  'Industry', 
                  'Service']
  df.drop(dropped_cols, inplace=True, axis=1)
  df.dropna(inplace=True)

  for column in df.columns:
    if isinstance(df.loc[5, column], str) and df.loc[5, column].find(',') > 0:
      df[column] = df[column].replace(',', '.', regex=True)
      df[column] = pd.to_numeric(df[column])

  print(df['Phones (per 1000)'])

  # analyze('GDP ($ per capita)', 'Infant mortality (per 1000 births)', regression_line=True)
  analyze('GDP ($ per capita)', 'Phones (per 1000)', regression_line=True)

  # summarize('GDP ($ per capita)')

def summarize(series):
  print(series, df[series].describe())
  df.boxplot(column=[series])
  plt.show()


def analyze(series_x, series_y, regression_line=True):
  new_df = pd.DataFrame({series_x : df[series_x],
                         series_y : df[series_y]})

  # use numpy to create a regression line
  d = np.polyfit(new_df[series_x], new_df[series_y], 1)
  f = np.poly1d(d)

  slope, intercept, rvalue, pvalue, stderr = st.linregress(new_df[series_x], new_df[series_y])

  # Technically, statsmodels should be able to do everything but I'm too lazy to redo it
  # Renaming is required because statsmodels gets angry at all special characters in the series names - stupid, but ok
  new_df.rename(columns={series_x : 'x', series_y : 'y'}, inplace=True)
  regression_table_model = smf.ols('y ~ x', data=new_df)
  results = regression_table_model.fit()
  new_df.rename(columns={'x' : series_x, 'y' : series_y}, inplace=True)

  new_df.insert(2, 'Treg', f(new_df[series_x]))
  ax = new_df.plot.scatter(x=series_x, y=series_y)
  if regression_line:
    new_df.plot(x=series_x, y='Treg', color='Red', ax=ax)
    plt.legend([series_x, f'{f}'])

  # Plot!
  plt.show()

  print(f"""Regression line for: {series_y} vs {series_x}: {str(f)}; \nR^2: {rvalue**2}; \np-value: {pvalue}""")
  print()
  print(interpret_data(series_x, series_y, slope, rvalue, pvalue, stderr, 0.05))
  print()
  print("Regression table: ")
  print(results.summary())



def interpret_data(series_x, series_y, slope, rvalue, pvalue, stderr, alpha):
  output = 'There is a ' 
  output += ('positive') if slope > 0 else ('negative') 
  output += ', linear, and '
  output += 'moderately strong to strong' if rvalue > 0.6 else 'weak'
  output += f' relationship between {series_x} and {series_y}. '
  if pvalue < alpha:
    output += f'Since the p-value is less than {alpha*100}%, reject H0, and there is statistically significant evidence of a linear relationship. '
  else:
    output += f'Since the p-value is greater than {alpha*100}%, fail to reject H0, and there is no statistically significant evidence of a linear relationship.'
  # Interpret R^2
  output += f'{rvalue**2 * 100}% of the variation in {series_y} is explained by the linear relationship between {series_x} and {series_y}.'
  output += f'The linear regression has a standard error of {stderr}.'
  return output

  # Return a formatted version for a regression table




if __name__ == "__main__":
  main()