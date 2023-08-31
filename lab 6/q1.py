import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

if os.path.exists('./plot')==False:
    os.mkdir('./plot')
intervals = ['daily', 'weekly', 'monthly']

def plot_data(x, y, dir, file_name, interval):
    print(f"Plotting {intervals[interval]} data for {file_name} of market {dir}")
    plt.plot(x, y)
    # plt.xlabel(x_title)
    # plt.ylabel(y_title)
    plt.title(f'Plot of company {file_name[:-5]} over {intervals[interval]} interval')
    # print(f'./plot/{dir}/{file_name[:-5]}')
    if os.path.exists(f'./plot/{dir}/{file_name[:-5]}') == False:
        os.mkdir(f'./plot/{dir}/{file_name[:-5]}')
    
    plt.savefig(f'./plot/{dir}/{file_name[:-5]}/{intervals[interval]}.png')
    plt.clf()

def plot_company_data(file_name,dir, df):
    new_dfs = []
    df["Date"] = pd.to_datetime(df["Date"])
    new_dfs.append(df["Date"])
    plot_data(df["Date"], df[file_name],dir, file_name, 0)
    df["Day"] = df["Date"].dt.dayofweek
    max_c, max_day = 0, 0
    for i in range(7):
        if df[df["Day"] == i].shape[0] > max_c:
            max_c = df[df["Day"] == i].shape[0]
            max_day = i
    
    df1 = df[df["Day"] == max_day]
    plot_data(df1["Date"], df1[file_name],dir, file_name, 1)
    new_dfs.append(df1["Date"])
    new_data_date = []
    new_data_price = []
    df.drop(['Day'], axis=1, inplace=True)
    # df["Date"] = pd.to_datetime(df["Date"])
    for index, row in df.iterrows():
        datestring = str(row["Date"])
        # print(datestring)
        m = int(datestring[5:7])
        if len(new_data_date) == 0:
            new_data_date.append(datestring)
            new_data_price.append(row[file_name])
        else:
            # print(type(k))
            if int(new_data_date[-1][5:7]) == m:
                continue
            else:
                new_data_date.append(datestring)
                new_data_price.append(row[file_name])
    # print(new_data_date[0])
    format = "%Y-%m-%d %H:%M:%S"
    for i in range(len(new_data_date)):
        new_data_date[i] = datetime.strptime(new_data_date[i], format)
    # df_monthly = pd.DataFrame({'Date': new_data_date, file_name:new_data_price})
    new_dfs.append(new_data_date)
    plot_data(new_data_date, new_data_price,dir, file_name, 2 )
    return new_dfs


def main():
    bsedata1 = pd.read_excel('bsedata1.xlsx')
    nsedata1 = pd.read_excel('nsedata1.xlsx')
    if os.path.exists('./plot/BSE') == False:
        os.mkdir('./plot/BSE')
    if os.path.exists('./plot/NSE') == False:
        os.mkdir('./plot/NSE')
    print("\n############### Plotting for BSE data ###############")
    bse_dfs = []
    for file in bsedata1.columns:
        if file == "Date":
            continue
        imp_dates = plot_company_data(file, "BSE", bsedata1)
    bsedata2 = bsedata1.reset_index()
    print(bsedata2["Date"])
    for dates in imp_dates:
        print(len(dates))
        # print(dates)
        bse_dfs.append(bsedata2[bsedata2["Date"].isin(dates)])

    if os.path.exists('./data') == False:
        os.mkdir('./data')
    if os.path.exists('./data/BSE') == False:
        os.mkdir('./data/BSE')
    if os.path.exists('./data/BSE_log') == False:
        os.mkdir('./data/BSE_log')
    for index,df in enumerate(bse_dfs):
        df.set_index("Date",inplace=True)
        # print(df.columns)
        df = df.pct_change()
        df.to_csv(f'./data/BSE/{intervals[index]}.csv')

    for index,df in enumerate(bse_dfs):
        # df.set_index("Date",inplace=True)
        #
        for col in df.columns:
            if col == "Date":
                continue
            df[col] = np.log(df[col])-np.log(df[col].shift(1))
        df.to_csv(f'./data/BSE_log/{intervals[index]}.csv')
    
    print("\n############### Plotting for NSE data ###############")
    for file in nsedata1.columns:
        if file == "Date":
            continue
        imp_dates = plot_company_data(file, "NSE", nsedata1)
    nse_dfs = []
    nsedata2 = nsedata1.reset_index()
    print(nsedata2["Date"])
    for dates in imp_dates:
        print(len(dates))
        # print(dates)
        nse_dfs.append(nsedata2[nsedata2["Date"].isin(dates)])

    if os.path.exists('./data') == False:
        os.mkdir('./data')
    if os.path.exists('./data/NSE') == False:
        os.mkdir('./data/NSE')
    if os.path.exists('./data/NSE_log') == False:
        os.mkdir('./data/NSE_log')
    for index,df in enumerate(nse_dfs):
        df.set_index("Date",inplace=True)
        # print(df.columns)
        df = df.pct_change()
        df.to_csv(f'./data/NSE/{intervals[index]}.csv')

    for index,df in enumerate(nse_dfs):
        # df.set_index("Date",inplace=True)
        #
        for col in df.columns:
            if col == "Date" or col=="index":
                continue
            df[col] = np.log(df[col])-np.log(df[col].shift(1))
        df.to_csv(f'./data/NSE_log/{intervals[index]}.csv')

main()

