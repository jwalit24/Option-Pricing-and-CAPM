import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os

intervals = ["daily","weekly","monthly"]

def load_data():
    bsedata = []
    nsedata = []
    for interval in intervals:
        df1 = pd.read_csv(f'./data/BSE_log/{interval}.csv')
        df2 = pd.read_csv(f'./data/NSE_log/{interval}.csv')
        df1.drop(0, axis = 0, inplace=True)
        df1.drop('index', axis=1, inplace=True)
        df2.drop(0, axis = 0, inplace=True)
        df2.drop('index', axis=1, inplace=True)
        bsedata.append(df1)
        nsedata.append(df2)

    return bsedata, nsedata

def f(x):
    return (1/math.sqrt(2*math.pi))*math.exp(-(x*x)/2)

def normalize(bsedata, nsedata):
    if os.path.exists('./histograms') == False:
        os.mkdir('./histograms')

    if os.path.exists('./histograms/BSE_log') == False:
        os.mkdir('./histograms/BSE_log')
    if os.path.exists('./histograms/NSE_log') == False:
        os.mkdir('./histograms/NSE_log')

    for index, df in enumerate(bsedata):
        if os.path.exists(f'./histograms/BSE_log/{intervals[index]}') == False:
            os.mkdir(f'./histograms/BSE_log/{intervals[index]}')
        for col in df.columns:
            if col == "Date":
                continue
            data = np.asarray(df[col],dtype = float)
            m = np.mean(data)
            s = math.sqrt(np.var(data))
            data = [(x-m)/s for x in data]
            # print(df[col])
            x = np.arange(np.min(data), np.max(data) , (np.max(data)-np.min(data))/1000)
            # x = np.arange(-2, 2, 0.0001)
            y = [f(x_val) for x_val in x]
            # print(y)
            plt.hist(data, density=True)
            plt.plot()
            plt.plot(x, y, color='orange')
            plt.title(f"{intervals[index]} data For company {col[:-5]}")
            print(f"Plotted histogram for {intervals[index]} data For company {col[:-5]}")
            plt.savefig(f'./histograms/BSE_log_{col[:-5]}_{intervals[index]}.png')
            plt.clf()
    
    for index, df in enumerate(nsedata):
        if os.path.exists(f'./histograms/NSE_log/{intervals[index]}') == False:
            os.mkdir(f'./histograms/NSE_log/{intervals[index]}')
        for col in df.columns:
            if col == "Date":
                continue
            data = np.asarray(df[col],dtype = float)
            m = np.mean(data)
            s = math.sqrt(np.var(data))
            data = [(x-m)/s for x in data]
            # print(df[col])
            x = np.arange(np.min(data), np.max(data) , (np.max(data)-np.min(data))/1000)
            # x = np.arange(-2, 2, 0.0001)
            y = [f(x_val) for x_val in x]
            # print(y)
            plt.hist(data, density=True)
            plt.plot()
            plt.plot(x, y, color='orange')
            plt.title(f"{intervals[index]} data For company {col[:-5]}")
            print(f"Plotted histogram for {intervals[index]} data For company {col[:-5]}")
            plt.savefig(f'./histograms/NSE_log_{col[:-5]}_{intervals[index]}.png')
            plt.clf()

def boxplots(bsedata, nsedata):
    if os.path.exists('./boxplots') == False:
        os.mkdir('./boxplots')

    if os.path.exists('./boxplots/BSE_log') == False:
        os.mkdir('./boxplots/BSE_log')
    if os.path.exists('./boxplots/NSE_log') == False:
        os.mkdir('./boxplots/NSE_log')

    for index, df in enumerate(bsedata):
        print(df.columns[1:])
        # for i in range(len(df.columns)):
        #     df.columns[i] = df.columns[i][:-5]
        plt.rcParams['font.size'] = 9
        boxplot = df.plot.box(figsize=(22,6),column = list(df.columns[1:]))
        plt.title(f'Box plot for BSE data over {intervals[index]} data')
        plt.savefig(f'./boxplots/BSE_log_{intervals[index]}.png')
        plt.clf()

    for index, df in enumerate(nsedata):
        print(df.columns[1:])
        # for i in range(len(df.columns)):
        #     df.columns[i] = df.columns[i][:-5]
        plt.rcParams['font.size'] = 9
        boxplot = df.plot.box(figsize=(22,15),column = list(df.columns[1:]))
        plt.title(f'Box plot for NSE data over {intervals[index]} data')
        plt.savefig(f'./boxplots/NSE_log_{intervals[index]}.png')
        plt.clf()



def main():
    bsedata, nsedata = load_data()
    normalize(bsedata, nsedata)
    boxplots(bsedata, nsedata)

main()
