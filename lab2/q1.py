
# imports
from math import exp, sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# functions ---------------------------------------------------------------------------

def plot_fixed_3d(x, y, z, x_axis, y_axis, z_axis, title):
  ax = plt.axes(projection='3d')
  # print(len(x), len(y), len(z))
  ax.scatter3D(x, y, z, cmap='Greens')
  plt.title(title)
  ax.set_xlabel(x_axis) 
  ax.set_ylabel(y_axis) 
  ax.set_zlabel(z_axis)
  plt.show()

def getOptionPrice(S0, K, T, r, sig, M, p, u, d, dt):
    # callList = [0]*(M+1)
    # putList = [0]*(M+1)
    callList=[]
    putList=[]
    for i in range(M+1):
        callList.append(0)
        putList.append(0)

    for i in range(M+1):
        putList[i] = max(0, K - S0*(u**i)*(d**(M-i)))
        callList[i] = max(S0*(u**i)*(d**(M-i)) - K, 0)
        

    for i in range(M):
        for j in range(M-i):
            putList[j] = ((1-p)*putList[j] + p*putList[j+1])*exp(-r*dt)
            callList[j] = ((1-p)*callList[j] + p*callList[j+1])*exp(-r*dt)
    return callList[0], putList[0]

# --------------------------------------------------------------------------------------


# Find the Initial prices of European Put and Call Options
# Given Initial Values
S0, K, T, M, r, sig = 100, 100, 1, 100, 0.08, 0.2

dt = T/M

d1 = exp(-sig*sqrt(dt))
u1 = exp(sig*sqrt(dt))
p1 = (exp(r*dt)-d1)/(u1-d1)

d2 = exp(-sig*sqrt(dt)+(r-sig*sig/2)*dt)
u2 = exp(sig*sqrt(dt)+(r-sig*sig/2)*dt)
p2 = (exp(r*dt)-d2)/(u2-d2)

callp1, putp1 = getOptionPrice(S0, K, T, r, sig, M, p1, u1, d1, dt)
callp2, putp2 = getOptionPrice(S0, K, T, r, sig, M, p2, u2, d2, dt)

print('The Call price for set 1 is: {}'.format(callp1))
print('The Put price for set 1 is: {}'.format(putp1))
print('The Call price for set 2 is: {}'.format(callp2))
print('The Put price for set 2 is: {}'.format(putp2))

# -------------------------------------------------------------------------------------------

# Draw 2-D Plots
# (a) S0 from 50 - 150 in steps of 5
S0 = np.arange(50, 155, 5)

callp1, putp1, callp2, putp2 = np.zeros(S0.shape[0]), np.zeros(
    S0.shape[0]), np.zeros(S0.shape[0]), np.zeros(S0.shape[0])

for i in range(S0.shape[0]):
    K, T, M, r, sig = 100, 1, 100, 0.08, 0.2
    dt = T/M

    u1 = exp(sig*sqrt(dt))
    d1 = exp(-sig*sqrt(dt))
    p1 = (exp(r*dt)-d1)/(u1-d1)

    u2 = exp(sig*sqrt(dt)+(r-sig*sig/2)*dt)
    d2 = exp(-sig*sqrt(dt)+(r-sig*sig/2)*dt)
    p2 = (exp(r*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0[i], K, T, r, sig, M, p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0[i], K, T, r, sig, M, p2, u2, d2, dt)

plt.plot(S0, callp1, color='red')
plt.title('Plot of Call option at t=0 vs S(0) (50-150) for Set 1 ')
plt.xlabel('Stock Price at t = 0')
plt.ylabel('Price of Call option')
plt.show()

plt.plot(S0, putp1, color='green')
plt.title('Plot of Put option at t=0 vs S(0) (50-150) for Set 1 ')
plt.xlabel('Stock Price at t = 0')
plt.ylabel('Price of Put option')
plt.show()

plt.plot(S0, callp2, color='red')
plt.title('Plot of Call option at t=0 vs S(0) (50-150) for Set 2 ')
plt.xlabel('Stock Price at t = 0')
plt.ylabel('Price of Call option')
plt.show()

plt.plot(S0, putp2, color='green')
plt.title('Plot of Put option at t=0 vs S(0) (50-150) for Set 2 ')
plt.xlabel('Stock Price at t = 0')
plt.ylabel('Price of Put option')
plt.show()

# (b) K from 50 - 150 in steps of 5
K = np.arange(50, 155, 5)
callp1, putp1, callp2, putp2 = np.zeros(K.shape[0]), np.zeros(
    K.shape[0]), np.zeros(K.shape[0]), np.zeros(K.shape[0])

for i in range(K.shape[0]):
    S0, T, M, r, sig = 100, 1, 100, 0.08, 0.2
    dt = T/M

    u1 = exp(sig*sqrt(dt))
    d1 = exp(-sig*sqrt(dt))
    p1 = (exp(r*dt)-d1)/(u1-d1)

    u2 = exp(sig*sqrt(dt)+(r-sig*sig/2)*dt)
    d2 = exp(-sig*sqrt(dt)+(r-sig*sig/2)*dt)
    p2 = (exp(r*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0, K[i], T, r, sig, M, p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0, K[i], T, r, sig, M, p2, u2, d2, dt)

plt.plot(K, callp1, color='red')
plt.title('Plot of Call option at t=0 vs K (50-150) for Set 1 ')
plt.xlabel('Strike Price (K)')
plt.ylabel('Price of Call option')
plt.show()

plt.plot(K, putp1, color='green')
plt.title('Plot of Put option at t=0 vs K (50-150) for Set 1 ')
plt.xlabel('Strike Price (K)')
plt.ylabel('Price of Put option')
plt.show()

plt.plot(K, callp2, color='red')
plt.title('Plot of Call option at t=0 vs K (50-150) for Set 2 ')
plt.xlabel('Strike Price (K)')
plt.ylabel('Price of Call option')
plt.show()

plt.plot(K, putp2, color='green')
plt.title('Plot of Put option at t=0 vs K (50-150) for Set 2 ')
plt.xlabel('Strike Price (K)')
plt.ylabel('Price of Put option')
plt.show()


# (c) r from 0 to 0.2 in steps of 0.01
r = np.arange(0, 0.21, 0.01)
callp1, putp1, callp2, putp2 = np.zeros(r.shape[0]), np.zeros(
    r.shape[0]), np.zeros(r.shape[0]), np.zeros(r.shape[0])

for i in range(r.shape[0]):
    S0, K, T, M, sig = 100, 100, 1, 100, 0.2
    dt = T/M

    u1 = exp(sig*sqrt(dt))
    d1 = exp(-sig*sqrt(dt))
    p1 = (exp(r[i]*dt)-d1)/(u1-d1)

    u2 = exp(sig*sqrt(dt)+(r[i]-sig*sig/2)*dt)
    d2 = exp(-sig*sqrt(dt)+(r[i]-sig*sig/2)*dt)
    p2 = (exp(r[i]*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0, K, T, r[i], sig, M, p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0, K, T, r[i], sig, M, p2, u2, d2, dt)

plt.plot(r, callp1, color='red')
plt.title('Plot of Call option at t=0 vs r (0-0.2) for Set 1 ')
plt.xlabel('Interest Rate (r)')
plt.ylabel('Price of Call option')
plt.show()

plt.plot(r, putp1, color='green')
plt.title('Plot of Put option at t=0 vs r (0-0.2) for Set 1 ')
plt.xlabel('Interest Rate (r)')
plt.ylabel('Price of Put option')
plt.show()

plt.plot(r, callp2, color='red')
plt.title('Plot of Call option at t=0 vs r (0-0.2) for Set 2 ')
plt.xlabel('Interest Rate (r)')
plt.ylabel('Price of Call option')
plt.show()

plt.plot(r, putp2, color='green')
plt.title('Plot of Put option at t=0 vs r (0-0.2) for Set 2 ')
plt.xlabel('Interest Rate (r)')
plt.ylabel('Price of Put option')
plt.show()


# (d) sig from 0 to 0.3 in steps of 0.01
sig = np.arange(0.01, 0.31, 0.01)  # avoid getting sigma = 0 exactly as it'd lead to div by 0
callp1, putp1, callp2, putp2 = np.zeros(sig.shape[0]), np.zeros(
    sig.shape[0]), np.zeros(sig.shape[0]), np.zeros(sig.shape[0])

for i in range(sig.shape[0]):
    S0, K, T, M, r = 100, 100, 1, 100, 0.08
    dt = T/M

    u1 = exp(sig[i]*sqrt(dt))
    d1 = exp(-sig[i]*sqrt(dt))
    p1 = (exp(r*dt)-d1)/(u1-d1)

    u2 = exp(sig[i]*sqrt(dt)+(r-sig[i]*sig[i]/2)*dt)
    d2 = exp(-sig[i]*sqrt(dt)+(r-sig[i]*sig[i]/2)*dt)
    p2 = (exp(r*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0, K, T, r, sig[i], M, p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0, K, T, r, sig[i], M, p2, u2, d2, dt)

plt.plot(sig, callp1, color='red')
plt.title('Plot of Call option at t=0 vs sig (0-0.3) for Set 1 ')
plt.xlabel('Value of Sigma')
plt.ylabel('Price of Call option')
plt.show()

plt.plot(sig, putp1, color='green')
plt.title('Plot of Put option at t=0 vs sig (0-0.3) for Set 1 ')
plt.xlabel('Value of Sigma')
plt.ylabel('Price of Put option')
plt.show()

plt.plot(sig, callp2, color='red')
plt.title('Plot of Call option at t=0 vs sig (0-0.3) for Set 2 ')
plt.xlabel('Value of Sigma')
plt.ylabel('Price of Call option')
plt.show()

plt.plot(sig, putp2, color='green')
plt.title('Plot of Put option at t=0 vs sig (0-0.3) for Set 2 ')
plt.xlabel('Value of Sigma')
plt.ylabel('Price of Put option')
plt.show()


# (e) M from 50-150 in steps of 1, and K for 3 values 95, 100 and 105
K = np.arange(95, 110, 5)
M = np.arange(50, 151, 1)

for j in range(K.shape[0]):
    callp1, putp1, callp2, putp2 = np.zeros(M.shape[0]), np.zeros(
        M.shape[0]), np.zeros(M.shape[0]), np.zeros(M.shape[0])

    for i in range(M.shape[0]):
        S0, T, r, sig = 100, 1, 0.08, 0.2
        dt = T/M[i]

        u1 = exp(sig*sqrt(dt))
        d1 = exp(-sig*sqrt(dt))
        p1 = (exp(r*dt)-d1)/(u1-d1)

        u2 = exp(sig*sqrt(dt)+(r-sig*sig/2)*dt)
        d2 = exp(-sig*sqrt(dt)+(r-sig*sig/2)*dt)
        p2 = (exp(r*dt)-d2)/(u2-d2)

        callp1[i], putp1[i] = getOptionPrice(S0, K[j], T, r, sig, M[i], p1, u1, d1, dt)
        callp2[i], putp2[i] = getOptionPrice(S0, K[j], T, r, sig, M[i], p2, u2, d2, dt)

    plt.plot(M, callp1, color='red')
    plt.title('Plot of Call option at t=0 vs M (50-150) for Set 1 and K = {} '.format(K[j]))
    plt.xlabel('No. of Steps (M)')
    plt.ylabel('Price of Call option')
    plt.show()

    plt.plot(M, putp1, color='green')
    plt.title('Plot of Put option at t=0 vs M (50-150) for Set 1 and K = {} '.format(K[j]))
    plt.xlabel('No. of Steps (M)')
    plt.ylabel('Price of Put option')
    plt.show()

    plt.plot(M, callp2, color='red')
    plt.title('Plot of Call option at t=0 vs M (50-150) for Set 2 and K = {} '.format(K[j]))
    plt.xlabel('No. of Steps (M)')
    plt.ylabel('Price of Call option')
    plt.show()

    plt.plot(M, putp2, color='green')
    plt.title('Plot of Put option at t=0 vs M (50-150) for Set 2 and K = {} '.format(K[j]))
    plt.xlabel('No. of Steps (M)')
    plt.ylabel('Price of Put option')
    plt.show()

# -------------------------------------------------------------------------------------------

# Draw 3-D Plots
# We'll have C(5,2) = 10 cases taking 2 parameters at a time and then plotting 4 graphs for each case.

# (a) S0 and K taken together
# Generate 600 uniform discrete random numbers in range 1 - 300

S0 = np.random.randint(1, 300+1, 600)
K = np.random.randint(1, 300+1, 600)

callp1, putp1, callp2, putp2 = np.zeros(S0.shape[0]), np.zeros(
    S0.shape[0]), np.zeros(S0.shape[0]), np.zeros(S0.shape[0])

for i in range(S0.shape[0]):
    T, M, r, sig = 1, 100, 0.08, 0.2
    dt = T/M

    u1 = exp(sig*sqrt(dt))
    d1 = exp(-sig*sqrt(dt))
    p1 = (exp(r*dt)-d1)/(u1-d1)

    u2 = exp(sig*sqrt(dt)+(r-sig*sig/2)*dt)
    d2 = exp(-sig*sqrt(dt)+(r-sig*sig/2)*dt)
    p2 = (exp(r*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0[i], K[i], T, r, sig, M, p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0[i], K[i], T, r, sig, M, p2, u2, d2, dt)





plot_fixed_3d(S0, K, callp1, "S0", "K", "Prices of Call option at t = 0", "Initial Call Option Price vs S0 and K for the set = 1")
plot_fixed_3d(S0, K, putp1,"S0", "K", "Prices of Put option at t = 0", "Initial Put Option Price vs S0 and K for the set = 1")




plot_fixed_3d(S0, K, callp2, "S0", "K", "Prices of Call option at t = 0", "Initial Call Option Price vs S0 and K for the set = 2")
plot_fixed_3d(S0, K, putp2,"S0", "K", "Prices of Put option at t = 0", "Initial Put Option Price vs S0 and K for the set = 2")


# (b) S0 and r taken together
# Generate 600 uniform discrete random numbers in range 1 - 300 and in range 0 to 0.2

S0 = np.random.randint(1, 300+1, 600)
r = np.random.uniform(0, 0.2, 600)

callp1, putp1, callp2, putp2 = np.zeros(S0.shape[0]), np.zeros(
    S0.shape[0]), np.zeros(S0.shape[0]), np.zeros(S0.shape[0])

for i in range(S0.shape[0]):
    K, T, M, sig = 100, 1, 100, 0.2
    dt = T/M

    u1 = exp(sig*sqrt(dt))
    d1 = exp(-sig*sqrt(dt))
    p1 = (exp(r[i]*dt)-d1)/(u1-d1)

    u2 = exp(sig*sqrt(dt)+(r[i]-sig*sig/2)*dt)
    d2 = exp(-sig*sqrt(dt)+(r[i]-sig*sig/2)*dt)
    p2 = (exp(r[i]*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0[i], K, T, r[i], sig, M, p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0[i], K, T, r[i], sig, M, p2, u2, d2, dt)



plot_fixed_3d(S0, r, callp1, "S0", "r", "Prices of Call option at t = 0", "Initial Call Option Price vs S0 and r for the set = 1" )
plot_fixed_3d(S0, r, putp1, "S0", "r", "Prices of Put option at t = 0", "Initial Put Option Price vs S0 and r for the set = 1")


plot_fixed_3d(S0, r, callp2, "S0", "r", "Prices of Call option at t = 0", "Initial Call Option Price vs S0 and r for the set = 2" )
plot_fixed_3d(S0, r, putp2, "S0", "r", "Prices of Put option at t = 0", "Initial Put Option Price vs S0 and r for the set = 2")


# (c) S0 and sigma taken together
# Generate 600 uniform discrete random numbers in range 1 - 300 and in range 0 to 0.2

S0 = np.random.randint(1, 300+1, 600)
sig = np.random.uniform(0.1, 0.3, 600)

callp1, putp1, callp2, putp2 = np.zeros(S0.shape[0]), np.zeros(
    S0.shape[0]), np.zeros(S0.shape[0]), np.zeros(S0.shape[0])

for i in range(S0.shape[0]):
    K, T, M, r = 100, 1, 100, 0.08
    dt = T/M

    u1 = exp(sig[i]*sqrt(dt))
    d1 = exp(-sig[i]*sqrt(dt))
    p1 = (exp(r*dt)-d1)/(u1-d1)

    u2 = exp(sig[i]*sqrt(dt)+(r-sig[i]*sig[i]/2)*dt)
    d2 = exp(-sig[i]*sqrt(dt)+(r-sig[i]*sig[i]/2)*dt)
    p2 = (exp(r*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0[i], K, T, r, sig[i], M, p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0[i], K, T, r, sig[i], M, p2, u2, d2, dt)



plot_fixed_3d(S0,sig,callp1, "S0", "sigma", "Prices of Call option at t = 0", "Initial Call Option Price vs S0 and sigma for the set = 1")
plot_fixed_3d(S0,sig,putp1, "S0", "sigma", "Prices of Put option at t = 0", "Initial Put Option Price vs S0 and sigma for the set = 1")



plot_fixed_3d(S0,sig,callp2, "S0", "sigma", "Prices of Call option at t = 0", "Initial Call Option Price vs S0 and sigma for the set = 2")
plot_fixed_3d(S0,sig,putp2, "S0", "sigma", "Prices of Put option at t = 0", "Initial Put Option Price vs S0 and sigma for the set = 2")


# (d) S0 and M taken together
# Generate 600 uniform discrete random numbers in range 1 - 300
S0 = np.random.randint(1, 300+1, 600)
M = np.random.randint(1, 300+1, 600)

callp1, putp1, callp2, putp2 = np.zeros(S0.shape[0]), np.zeros(
    S0.shape[0]), np.zeros(S0.shape[0]), np.zeros(S0.shape[0])

for i in range(S0.shape[0]):
    T, K, r, sig = 1, 100, 0.08, 0.2
    dt = T/M[i]

    u1 = exp(sig*sqrt(dt))
    d1 = exp(-sig*sqrt(dt))
    p1 = (exp(r*dt)-d1)/(u1-d1)

    u2 = exp(sig*sqrt(dt)+(r-sig*sig/2)*dt)
    d2 = exp(-sig*sqrt(dt)+(r-sig*sig/2)*dt)
    p2 = (exp(r*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0[i], K, T, r, sig, M[i], p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0[i], K, T, r, sig, M[i], p2, u2, d2, dt)



plot_fixed_3d(S0,M,callp1, "S0", "M", "Prices of Call option at t = 0", "Initial Call Option Price vs S0 and M for the set = 1")
plot_fixed_3d(S0,M,putp1, "S0", "M", "Prices of Put option at t = 0", "Initial Put Option Price vs S0 and M for the set = 1")


plot_fixed_3d(S0,M,callp2, "S0", "M", "Prices of Call option at t = 0", "Initial Call Option Price vs S0 and M for the set = 2")
plot_fixed_3d(S0,M,putp2, "S0", "M", "Prices of Put option at t = 0", "Initial Put Option Price vs S0 and M for the set = 2")


# (e) K and r taken together
# Generate 600 uniform discrete random numbers in range 1 - 300 and in range 0 to 0.2

K = np.random.randint(1, 300+1, 600)
r = np.random.uniform(0, 0.2, 600)

callp1, putp1, callp2, putp2 = np.zeros(K.shape[0]), np.zeros(
    K.shape[0]), np.zeros(K.shape[0]), np.zeros(K.shape[0])

for i in range(K.shape[0]):
    S0, T, M, sig = 100, 1, 100, 0.2
    dt = T/M

    u1 = exp(sig*sqrt(dt))
    d1 = exp(-sig*sqrt(dt))
    p1 = (exp(r[i]*dt)-d1)/(u1-d1)

    u2 = exp(sig*sqrt(dt)+(r[i]-sig*sig/2)*dt)
    d2 = exp(-sig*sqrt(dt)+(r[i]-sig*sig/2)*dt)
    p2 = (exp(r[i]*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0, K[i], T, r[i], sig, M, p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0, K[i], T, r[i], sig, M, p2, u2, d2, dt)



plot_fixed_3d(K,r, callp1, "K", "r", "Prices of Call option at t = 0", "Initial Call Option Price vs K and r for the set = 1" )
plot_fixed_3d(K,r, putp1, "K", "r", "Prices of Put option at t = 0", "Initial Put Option Price vs K and r for the set = 1")


plot_fixed_3d(K,r, callp2, "K", "r", "Prices of Call option at t = 0", "Initial Call Option Price vs K and r for the set = 2" )
plot_fixed_3d(K,r, putp2, "K", "r", "Prices of Put option at t = 0", "Initial Put Option Price vs K and r for the set = 2")



# (f) K and sig taken together
# Generate 600 uniform discrete random numbers in range 1 - 300 and in range 0 to 0.3

K = np.random.randint(1, 300+1, 600)
sig = np.random.uniform(0.1, 0.3, 600)

callp1, putp1, callp2, putp2 = np.zeros(K.shape[0]), np.zeros(
    K.shape[0]), np.zeros(K.shape[0]), np.zeros(K.shape[0])

for i in range(K.shape[0]):
    S0, T, M, r = 100, 1, 100, 0.08
    dt = T/M

    u1 = exp(sig[i]*sqrt(dt))
    d1 = exp(-sig[i]*sqrt(dt))
    p1 = (exp(r*dt)-d1)/(u1-d1)

    u2 = exp(sig[i]*sqrt(dt)+(r-sig[i]*sig[i]/2)*dt)
    d2 = exp(-sig[i]*sqrt(dt)+(r-sig[i]*sig[i]/2)*dt)
    p2 = (exp(r*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0, K[i], T, r, sig[i], M, p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0, K[i], T, r, sig[i], M, p2, u2, d2, dt)




plot_fixed_3d(K, sig, callp1, "K", "sigma", "Prices of Call option at t = 0", "Initial Call Option Price vs K and sigma for the set = 1")
plot_fixed_3d(K, sig, putp1, "K", "sigma", "Prices of Put option at t = 0", "Initial Put Option Price vs K and sigma for the set = 1")


plot_fixed_3d(K, sig, callp2, "K", "sigma", "Prices of Call option at t = 0", "Initial Call Option Price vs K and sigma for the set = 2")
plot_fixed_3d(K, sig, putp2, "K", "sigma", "Prices of Put option at t = 0", "Initial Put Option Price vs K and sigma for the set = 2")


# (g) K and M taken together
# Generate 600 uniform discrete random numbers in range 1 - 300
K = np.random.randint(1, 300+1, 600)
M = np.random.randint(1, 300+1, 600)

callp1, putp1, callp2, putp2 = np.zeros(K.shape[0]), np.zeros(
    K.shape[0]), np.zeros(K.shape[0]), np.zeros(K.shape[0])

for i in range(K.shape[0]):
    T, S0, r, sig = 1, 100, 0.08, 0.2
    dt = T/M[i]

    u1 = exp(sig*sqrt(dt))
    d1 = exp(-sig*sqrt(dt))
    p1 = (exp(r*dt)-d1)/(u1-d1)

    u2 = exp(sig*sqrt(dt)+(r-sig*sig/2)*dt)
    d2 = exp(-sig*sqrt(dt)+(r-sig*sig/2)*dt)
    p2 = (exp(r*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0, K[i], T, r, sig, M[i], p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0, K[i], T, r, sig, M[i], p2, u2, d2, dt)



plot_fixed_3d(K,M, callp1, "K", "M", "Prices of Call option at t = 0", "Initial Call Option Price vs K and M for the set = 1")
plot_fixed_3d(K,M,putp1, "K", "M", "Prices of Put option at t = 0", "Initial Put Option Price vs K and M for the set = 1")



plot_fixed_3d(K,M, callp2, "K", "M", "Prices of Call option at t = 0", "Initial Call Option Price vs K and M for the set = 2")
plot_fixed_3d(K,M,putp2, "K", "M", "Prices of Put option at t = 0", "Initial Put Option Price vs K and M for the set = 2")


# (h) r and sig taken together
# Generate 600 uniform discrete random numbers in range 0 to 0.2 and in range 0 to 0.3

r = np.random.uniform(0, 0.2, 600)
sig = np.random.uniform(0.1, 0.3, 600)

callp1, putp1, callp2, putp2 = np.zeros(r.shape[0]), np.zeros(
    r.shape[0]), np.zeros(r.shape[0]), np.zeros(r.shape[0])

for i in range(r.shape[0]):
    S0, K, T, M = 100, 100, 1, 100
    dt = T/M

    u1 = exp(sig[i]*sqrt(dt))
    d1 = exp(-sig[i]*sqrt(dt))
    p1 = (exp(r[i]*dt)-d1)/(u1-d1)

    u2 = exp(sig[i]*sqrt(dt)+(r[i]-sig[i]*sig[i]/2)*dt)
    d2 = exp(-sig[i]*sqrt(dt)+(r[i]-sig[i]*sig[i]/2)*dt)
    p2 = (exp(r[i]*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0, K, T, r[i], sig[i], M, p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0, K, T, r[i], sig[i], M, p2, u2, d2, dt)



plot_fixed_3d(r, sig, callp1, "r", "sigma", "Prices of Call option at t = 0", "Initial Call Option Price vs r and sigma for the set = 1")
plot_fixed_3d(r, sig, putp1, "r", "sigma", "Prices of Put option at t = 0", "Initial Put Option Price vs r and sigma for the set = 1")


plot_fixed_3d(r, sig, callp2, "r", "sigma", "Prices of Call option at t = 0", "Initial Call Option Price vs r and sigma for the set =2")
plot_fixed_3d(r, sig, putp2, "r", "sigma", "Prices of Put option at t = 0", "Initial Put Option Price vs r and sigma for the set = 2")

# (i) r and M taken together
# Generate 600 uniform discrete random numbers in range 0 to 0.2 and in range 1 to 300

r = np.random.uniform(0, 0.2, 600)
M = np.random.randint(1, 301, 600)

callp1, putp1, callp2, putp2 = np.zeros(r.shape[0]), np.zeros(
    r.shape[0]), np.zeros(r.shape[0]), np.zeros(r.shape[0])

for i in range(r.shape[0]):
    S0, K, T, sig = 100, 100, 1, 0.2
    dt = T/M[i]

    u1 = exp(sig*sqrt(dt))
    d1 = exp(-sig*sqrt(dt))
    p1 = (exp(r[i]*dt)-d1)/(u1-d1)

    u2 = exp(sig*sqrt(dt)+(r[i]-sig*sig/2)*dt)
    d2 = exp(-sig*sqrt(dt)+(r[i]-sig*sig/2)*dt)
    p2 = (exp(r[i]*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0, K, T, r[i], sig, M[i], p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0, K, T, r[i], sig, M[i], p2, u2, d2, dt)


plot_fixed_3d(r,M,callp1, "r", "M", "Prices of Call option at t = 0", "Initial Call Option Price vs r and M for the set = 1")
plot_fixed_3d(r, M, putp1, "r", "M", "Prices of Put option at t = 0", "Initial Put Option Price vs r and M for the set = 1")



plot_fixed_3d(r,M,callp2, "r", "M", "Prices of Call option at t = 0", "Initial Call Option Price vs r and M for the set = 2")
plot_fixed_3d(r, M, putp2, "r", "M", "Prices of Put option at t = 0", "Initial Put Option Price vs r and M for the set = 2")


# (j) sig and M taken together
# Generate 600 uniform discrete random numbers in range 0 to 0.3 and in range 1 to 300

sig = np.random.uniform(0.1, 0.3, 600)
M = np.random.randint(1, 301, 600)

callp1, putp1, callp2, putp2 = np.zeros(sig.shape[0]), np.zeros(
    sig.shape[0]), np.zeros(sig.shape[0]), np.zeros(sig.shape[0])

for i in range(sig.shape[0]):
    S0, K, T, r = 100, 100, 1, 0.08
    dt = T/M[i]

    u1 = exp(sig[i]*sqrt(dt))
    d1 = exp(-sig[i]*sqrt(dt))
    p1 = (exp(r*dt)-d1)/(u1-d1)

    u2 = exp(sig[i]*sqrt(dt)+(r-sig[i]*sig[i]/2)*dt)
    d2 = exp(-sig[i]*sqrt(dt)+(r-sig[i]*sig[i]/2)*dt)
    p2 = (exp(r*dt)-d2)/(u2-d2)

    callp1[i], putp1[i] = getOptionPrice(S0, K, T, r, sig[i], M[i], p1, u1, d1, dt)
    callp2[i], putp2[i] = getOptionPrice(S0, K, T, r, sig[i], M[i], p2, u2, d2, dt)



plot_fixed_3d(M, sig, callp1, "sigma", "M", "Prices of Call option at t = 0", "Initial Call Option Price vs sigma and M for the set = 1")
plot_fixed_3d(M, sig, putp1, "sigma", "M", "Prices of Put option at t = 0", "Initial Put Option Price vs sigma and M for the set = 1")



plot_fixed_3d(M, sig, callp2, "sigma", "M", "Prices of Call option at t = 0", "Initial Call Option Price vs sigma and M for the set = 2")
plot_fixed_3d(M, sig, putp2, "sigma", "M", "Prices of Put option at t = 0", "Initial Put Option Price vs sigma and M for the set = 2")


# -------------------------------------------------------------------------------------------
# Question 1 complete
