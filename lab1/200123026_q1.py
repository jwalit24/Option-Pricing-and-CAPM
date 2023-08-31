
import math
import pandas as pd
from IPython.display import display

# Function to get Option Price for a given M
def optionpricecalculator(M, u, d, p):
    putoptionlist = []
    calloptionlist =[]
    for i in range(M+1):
       calloptionlist.append(0)
       putoptionlist.append(0)

    
    for i in range(M+1):
        putoptionlist[i] = max(0, K-S0*((pow(u,i))*(pow(d,M-i))))
        calloptionlist[i] = max(S0*((pow(u,i))*(pow(d,M-i))) - K, 0)
        
    for i in range(M):
        for j in range(M-i):
            putoptionlist[j] = ((1-p)*putoptionlist[j] + p*putoptionlist[j+1])*math.exp(-r*T/M)
            calloptionlist[j] = ((1-p)*calloptionlist[j] + p*calloptionlist[j+1])*math.exp(-r*T/M)
           
    return calloptionlist[0], putoptionlist[0]

# Given data
S0=100
T=5
K=105
sig=0.3
r=0.05
MList=[1, 5, 10, 20, 50, 100, 200, 400]

# Lists to store the option prices
putPrices = []
callPrices = []



for M in MList:
    dt = T/M
    d = math.exp(-sig*math.sqrt(dt)+(r-sig*sig/2)*dt)
    u = math.exp(sig*math.sqrt(dt)+(r-sig*sig/2)*dt)
    p = (math.exp(r*dt)-d)/(u-d)
    
    # Check if No Arbitrage Principle has got violated
    if p < 0 or p > 1:
        print("No Arbitrage Principle has Violated")
        CallPrices.append('-')
        PutPrices.append('-')
        continue
    
    call, put = optionpricecalculator(M, u, d, p)
    callPrices.append(call)
    putPrices.append(put)

# Display the data using Pandas Dataframe
df = pd.DataFrame({'Step Size':MList,'Call Option Price': callPrices, 'Put Option Price': putPrices},)
display(df)
