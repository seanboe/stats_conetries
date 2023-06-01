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
  df.set_index('Country', inplace=True)
  df.dropna(inplace=True)

  plot_data('GDP ($ per capita)', 'Pop. Density (per sq. mi.)', regression_line=True)


def plot_data(series_x, series_y, regression_line=True):
  new_df = pd.DataFrame({series_x : df[series_x],
                         series_y : df[series_y]})
  
  # verify that there are no commas in the data - this is probably better to do in the cleaning stage, but whatever
  if isinstance(new_df[series_x][0], str):
    new_df[series_x] = new_df[series_x].replace(',', '.', regex=True)
    new_df[series_x] = pd.to_numeric(new_df[series_x])
  if isinstance(new_df[series_y][0], str):
    new_df[series_y] = new_df[series_y].replace(',', '.', regex=True)
    new_df[series_y] = pd.to_numeric(new_df[series_y])

  # use numpy to create a regression line
  d = np.polyfit(new_df[series_x], new_df[series_y], 1)
  f = np.poly1d(d)

  # print out the regression line
  print(f'Regression line for: {series_y} vs {series_x}: {str(f)}')

  new_df.insert(2, 'Treg', f(new_df[series_x]))
  ax = new_df.plot.scatter(x=series_x, y=series_y)
  if regression_line:
    new_df.plot(x=series_x, y='Treg', color='Red', ax=ax)
    plt.legend([series_x, f'{f}'])

  # plot
  plt.show()

if __name__ == "__main__":
  main()