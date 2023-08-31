

import math
import numpy as np
import matplotlib.pyplot as plt

# Given data
S0=100
K=105
T=5
r=0.05
sig=0.3

# Function to get Option Price for a given M
def optionpricecalculator(M):
    dt = T/M
    d = math.exp(-sig*math.sqrt(dt)+(r-sig*sig/2)*dt)
    u = math.exp(sig*math.sqrt(dt)+(r-sig*sig/2)*dt)
    p = (math.exp(r*dt)-d)/(u-d)
    
    # Check if No Arbitrage Principle has got violated
    if p < 0 or p > 1:
        print("No Arbitrage Principle has Violated")
        return '-','-'
    
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

# Lists to store the option prices
callPrices = []
putPrices = []
M=0
# Compute initial option prices in steps of 1
while M < 400:
    M += 1
    call, put = optionpricecalculator(M)
    callPrices.append(call)
    putPrices.append(put)
MList = np.linspace(1, 400, 400)

print(MList)

plt.plot(MList, callPrices)
plt.xlabel('Value of M')
plt.ylabel('Call Option Price')
plt.title('Varying Price of Call Option with Value of M (Step Size 1)')
plt.show()

plt.plot(MList, putPrices)
plt.xlabel('Value of M')
plt.ylabel('Price of Put Option')
plt.title('Varying Price of Put Option with Value of M (Step Size 1)')
plt.show()

# Lists to store the option prices
callPrices = []
putPrices = []

# Compute initial option prices in steps of 5
M=0
while M < 400:
    M += 5
    call, put = optionpricecalculator(M)
    callPrices.append(call)
    putPrices.append(put)
MList = np.linspace(5, 400, 80)

print(MList)

plt.plot(MList, callPrices)
plt.xlabel('Value of M')
plt.ylabel('Call Option Price')
plt.title('Varying Call Option Price with Value of M (Step Size 5)')
plt.show()

plt.plot(MList, putPrices)
plt.xlabel('Value of M')
plt.ylabel('Price of Put Option')
plt.title('Varying Put Option Price with Value of M (Step Size 5)')
plt.show()
