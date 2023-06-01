import pandas as pd
import numpy as np  
import scipy.stats as st
import matplotlib.pyplot as plt

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

  dropped_cols = ['Phones (per 1000)', 
                  'Arable (%)', 
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

  plot_data('GDP ($ per capita)', 'Pop. Density (per sq. mi.)', regression_line=True)

  # summarize('GDP ($ per capita)')

def summarize(series):
  print(series, df[series].describe())
  df.boxplot(column=[series])
  plt.show()


def plot_data(series_x, series_y, regression_line=True):
  new_df = pd.DataFrame({series_x : df[series_x],
                         series_y : df[series_y]})

  # use numpy to create a regression line
  d = np.polyfit(new_df[series_x], new_df[series_y], 1)
  f = np.poly1d(d)

  slope, intercept, rvalue, pvalue, stderr = st.linregress(new_df[series_x], new_df[series_y])

  # print out the regression line
  print(f"""Regression line for: {series_y} vs {series_x}: {str(f)}; \nR^2: {rvalue**2}; \np-value: {pvalue}""")

  new_df.insert(2, 'Treg', f(new_df[series_x]))
  ax = new_df.plot.scatter(x=series_x, y=series_y)
  if regression_line:
    new_df.plot(x=series_x, y='Treg', color='Red', ax=ax)
    plt.legend([series_x, f'{f}'])

  # plot
  plt.show()


if __name__ == "__main__":
  main()