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
        df1 = pd.read_csv(f'./data/BSE/{interval}.csv')
        df2 = pd.read_csv(f'./data/NSE/{interval}.csv')
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

    if os.path.exists('./histograms/BSE') == False:
        os.mkdir('./histograms/BSE')
    if os.path.exists('./histograms/NSE') == False:
        os.mkdir('./histograms/NSE')

    for index, df in enumerate(bsedata):
        if os.path.exists(f'./histograms/BSE/{intervals[index]}') == False:
            os.mkdir(f'./histograms/BSE/{intervals[index]}')
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
            plt.savefig(f'./histograms/BSE_{col[:-5]}_{intervals[index]}.png')
            plt.clf()
    
    for index, df in enumerate(nsedata):
        if os.path.exists(f'./histograms/NSE/{intervals[index]}') == False:
            os.mkdir(f'./histograms/NSE/{intervals[index]}')
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
            plt.savefig(f'./histograms/NSE_{col[:-5]}_{intervals[index]}.png')
            plt.clf()

def boxplots(bsedata, nsedata):
    if os.path.exists('./boxplots') == False:
        os.mkdir('./boxplots')

    if os.path.exists('./boxplots/BSE') == False:
        os.mkdir('./boxplots/BSE')
    if os.path.exists('./boxplots/NSE') == False:
        os.mkdir('./boxplots/NSE')

    for index, df in enumerate(bsedata):
        print(df.columns[1:])
        # for i in range(len(df.columns)):
        #     df.columns[i] = df.columns[i][:-5]
        plt.rcParams['font.size'] = 9
        boxplot = df.plot.box(figsize=(22,6),column = list(df.columns[1:]))
        plt.title(f'Boxplot for BSE for {intervals[index]} data')
        plt.savefig(f'./boxplots/BSE_{intervals[index]}.png')
        plt.clf()

    for index, df in enumerate(nsedata):
        print(df.columns[1:])
        # for i in range(len(df.columns)):
        #     df.columns[i] = df.columns[i][:-5]
        plt.rcParams['font.size'] = 9
        boxplot = df.plot.box(figsize=(22,15),column = list(df.columns[1:]))
        plt.title(f'Boxplot for NSE for {intervals[index]} data')
        plt.savefig(f'./boxplots/NSE_{intervals[index]}.png')
        plt.clf()



def main():
    bsedata, nsedata = load_data()
    normalize(bsedata, nsedata)
    boxplots(bsedata, nsedata)

main()
