
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pandas import to_datetime

# if using a Jupyter notebook, kindly uncomment the following line:
# %matplotlib inline


def plot_stock_prices(df1, df2, sname, intvl, stocks_type):
  x = df1['Date'].to_list()
  y1 = df1[sname].to_list()
  y2 = df2[sname].to_list()

  plt.rcParams["figure.figsize"] = (20, 5)
  if intvl == 'daily':
    plt.subplot(1, 3, 1)
  elif intvl == 'weekly':
    plt.subplot(1, 3, 2)
  else:
    plt.subplot(1, 3, 3)
#   print(x)
  plt.plot(x, y2, color = 'orange', label = 'Predicted Price')
  plt.plot(x, y1, color = 'blue', label = 'Original Price')
#   plt.xticks(np.arange(0, len(x), int(len(x)/4)), df1['Date'][0:len(x):int(len(x)/4)])
  plt.title('Plot for Stock prices for {} on {} basis'.format(sname, intvl))
  plt.xlabel('Time')
  plt.ylabel('Price')
  plt.grid(True)
  plt.legend()

  if intvl == 'monthly':
    # plt.savefig('./Estimations/' + stocks_type + '/' + sname + '.png')
    plt.show()
  

def generate_path(df, stocks_name, stocks_type):
  df = df.fillna(method ='bfill')
  interval = ['daily', 'weekly', 'monthly']
  initial_df = df.copy()
  
  for sname in stocks_name:
    df = initial_df.copy()

    for intvl in interval:
      delta_t = 1/252
      df = initial_df.copy()
    #   print(df['Date'])
      df.reset_index(inplace=True)
      if intvl == 'weekly':
        df['Day'] = (to_datetime(df['Date'])).dt.day_name()
        df = df.loc[df['Day'] == 'Monday']
        del df['Day']
        delta_t = 7/252
      elif intvl == 'monthly':
        df = df.groupby(pd.DatetimeIndex(df['Date']).to_period('M')).nth(0)
        delta_t = 30/252

      df_original = df.copy()
      df_training = df.loc[ df['Date'] <= '2021-12-31']
      df_predicted = df.loc[ df['Date'] > '2021-12-31']
      df_predicted.set_index('Date', inplace = True)

      x = np.log(df_predicted[sname]/df_predicted[sname].shift(1))

      factor = 0
      if intvl == 'daily':
        factor = 252
      elif intvl == 'weekly':
        factor = 52
      else:
        factor = 12
    
      mu = np.nanmean(np.array(x)) 
      sigma_2 = np.nanvar(np.array(x))

      
      
      mu *= factor
      sigma_2 *= (len(x) * factor) / (len(x) - 1)
      mu += 0.5 * sigma_2
      np.random.seed(40)
      
      S0 = df_training.iloc[len(df_training) - 1][sname]
      for idx, row in df_predicted.iterrows():
        S = S0 * math.exp((mu - 0.5 * sigma_2) * delta_t + math.sqrt(sigma_2) * math.sqrt(delta_t) * np.random.normal(0, 1))
        S0 = S
        row[sname] = S

      df_predicted = df_training.append(df_predicted, ignore_index=True)
      plot_stock_prices(df_original, df_predicted, sname, intvl, stocks_type)


def main():
  stocks_type = ['bsedata1.xlsx', 'nsedata1.xlsx']
  stocks_name = []

  for fname in stocks_type:
    df = pd.read_excel(fname)
    df.set_index('Date', inplace=True)
    stocks_name = df.columns
    generate_path(df, stocks_name, fname)

main()