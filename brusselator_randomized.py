import numpy as np
import random as rd
import scipy as sci
import itertools as it
from scipy.optimize import fsolve
from collections import Counter
from itertools import islice, combinations
from scipy.interpolate import interp1d
from scipy import stats, integrate

def smartlog(a,b):
    if(a != 0 and b != 0):
        return np.log(a/b)
    return 0

def simplified_Brusselator(A, B, V, rates, tmax):
    [k1,km1,k2,km2,k3,km3] = rates
    X = 10
    Y = 10
    t = 0
    traj = []
    trajA = []
    trajAB = []
    times = []
    timesA = []
    timesAB = []
    Σ = 0

    # burner
    while t < 10:

        r1 = A*k1
        rm1 = X*km1
        r2 = B*k2
        rm2 = Y*km2
        r3 = X*(X-1)*Y*k3/(V**2)
        rm3 = X*(X-1)*(X-2)*km3/(V**2)
        tot = r1 + rm1 + r2 + rm2 + r3 + rm3

        r = tot*rd.random()

        if r <= r1:
            X += 1
        elif r <= r1 + rm1:
            X -= 1
        elif r <= r1 + rm1 + r2:
            Y += 1
        elif r <= r1 + rm1 + r2 + rm2:
            Y -= 1
        elif r <= r1 + rm1 + r2 + rm2 + r3:
            X += 1
            Y -= 1
        else:
            X -= 1
            Y += 1
        t += -np.log(rd.random())/tot

    t = 0
    while t < tmax:

        r1 = A*k1
        rm1 = X*km1
        r2 = B*k2
        rm2 = Y*km2
        #r3 = (X**2)*Y*k3
        #rm3 = (X**3)*km3
        r3 = X*(X-1)*Y*k3/(V**2)
        rm3 = X*(X-1)*(X-2)*km3/(V**2)
        tot = r1 + rm1 + r2 + rm2 + r3 + rm3

        r = tot*rd.random()
        t += -np.log(rd.random())/tot

        if r <= r1:
            X += 1
            traj.append('pA')
            times.append(t)
            trajA.append('pA')
            timesA.append(t)
            trajAB.append('pA')
            timesAB.append(t)
            Σ += smartlog(r1,X*km1)
        elif r <= r1 + rm1:
            X -= 1
            traj.append('mA')
            times.append(t)
            trajA.append('mA')
            timesA.append(t)
            trajAB.append('mA')
            timesAB.append(t)
            Σ += smartlog(rm1,r1)
        elif r <= r1 + rm1 + r2:
            Y += 1
            traj.append('pB')
            times.append(t)
            trajAB.append('pB')
            timesAB.append(t)
            Σ += smartlog(r2,Y*km2)
        elif r <= r1 + rm1 + r2 + rm2:
            Y -= 1
            traj.append('mB')
            times.append(t)
            trajAB.append('mB')
            timesAB.append(t)
            Σ += smartlog(rm2,r2)
        elif r <= r1 + rm1 + r2 + rm2 + r3:
            X += 1
            Y -= 1
            traj.append('pC')
            times.append(t)
            Σ += smartlog(r3,X*(X-1)*(X-2)*km3/(V**2))
        else:
            X -= 1
            Y += 1
            traj.append('mC')
            times.append(t)
            Σ += smartlog(rm3,X*(X-1)*Y*k3/(V**2))


    return traj, times, Σ/tmax, trajA, timesA, trajAB, timesAB

def count_pairs(array,n=2):
    c1 = Counter(zip(*[islice(array,i,None) for i in range(n)]))
    total = 1.0 * (len(array) - n + 1)
    for k in c1:
        c1[k] /= total
    
    return c1

def cumcount_reduced(arr):
    '''Returns the step function value at each increment of the CDF'''
    sorted_arr = np.array(sorted(arr))
    counts = np.zeros(len(arr))
    
    rolling_count = 0
    for idx, elem in enumerate(sorted_arr):
        rolling_count += 1
        counts[idx] = rolling_count

    counts /= len(counts)
    counts -= (1 / (2 * len(counts)))

    return (sorted_arr, counts)


def KLD_PerezCruz(P, Q, eps=1e-11):
    '''takes two datasets to estimate the relative entropy between their PDFs
    we use eps=10^-11, but it could be defined as < the minimal interval between data points'''
    P = sorted(P)
    Q = sorted(Q)
    
    P_positions, P_counts = cumcount_reduced(P)
    Q_positions, Q_counts = cumcount_reduced(Q)
    
    #definition of x_0 and x_{n+1}
    x_0 = np.min([P_positions[0], Q_positions[0]]) - 1
    P_positions = np.insert(P_positions, 0, [x_0])
    P_counts = np.insert(P_counts, 0, [0])
    Q_positions = np.insert(Q_positions, 0, [x_0])
    Q_counts = np.insert(Q_counts, 0, [0])
    
    x_np1 = np.max([P_positions[-1], Q_positions[-1]]) + 1
    P_positions = np.append(P_positions, [x_np1])
    P_counts = np.append(P_counts, [1])
    Q_positions = np.append(Q_positions, [x_np1])
    Q_counts = np.append(Q_counts, [1])
    
    f_P = interp1d(P_positions, P_counts)
    f_Q = interp1d(Q_positions, Q_counts) 
    
    X = P_positions[1:-2]
    eps = 0.5 * np.min( np.diff( X ) )
    values = (f_P(X) - f_P(X - eps)) / (f_Q(X) - f_Q(X - eps))
    filt = ((values != 0.) & ~(np.isinf(values)) & ~(np.isnan(values)))
    values_filter = values[filt]
    out = (np.sum(np.log(values_filter)) / len(values_filter)) - 1.

    return out

def KLD_PerezCruz_2(P, Q):
    '''takes two datasets to estimate the relative entropy between their PDFs
    we use eps=10^-11, but it could be defined as < the minimal interval between data points'''
    P = sorted(P)
    Q = sorted(Q)
    
    P_positions, P_counts = cumcount_reduced(P)
    Q_positions, Q_counts = cumcount_reduced(Q)
    
    #definition of x_0 and x_{n+1}
    x_0 = np.min([P_positions[0], Q_positions[0]]) - 1
    P_positions = np.insert(P_positions, 0, [x_0])
    P_counts = np.insert(P_counts, 0, [0])
    Q_positions = np.insert(Q_positions, 0, [x_0])
    Q_counts = np.insert(Q_counts, 0, [0])
    
    x_np1 = np.max([P_positions[-1], Q_positions[-1]]) + 1
    P_positions = np.append(P_positions, [x_np1])
    P_counts = np.append(P_counts, [1])
    Q_positions = np.append(Q_positions, [x_np1])
    Q_counts = np.append(Q_counts, [1])
    
    f_P = interp1d(P_positions, P_counts)
    f_Q = interp1d(Q_positions, Q_counts) 
    
    X = P_positions[1:-2]
    eps = 0.5 * np.min( np.diff( X ) )
    values = [smartlog(f_P(Xi) - f_P(Xi - eps), f_Q(Xi) - f_Q(Xi - eps)) for Xi in X]
    out = (np.sum(values) / len(values)) - 1.

    return out

x_dict = {
    "pA" : 0,
    "mA" : 1,
    "pB" : 2,
    "mB" : 3
}

rev = {
    "pA" : "mA",
    "mA" : "pA",
    "pB" : "mB",
    "mB" : "pB",
    "pC" : "mC",
    "mC" : "pC"
}

[k1, km1, k2, km2, k3, km3] = np.array([2*rd.random() for x in range(6)]) + 1
B = int( 5 + 5 * rd.random() )
V = 5
A = int( 5 + 5 * rd.random() )
tmax = 10**3
reps = 10

vec = []

for rep in range(reps):

    traj, times, epr, trajA, timesA, trajAB, timesAB = simplified_Brusselator(A, B, V, [k1, km1, k2, km2, k3, km3], tmax)
    Δμ = np.log( (B * k2 * km1 * k3)/(A * km2 * k1 * km3) )

    # σ_zk_AB
    K = len(trajAB)/tmax
    count = Counter(trajAB)
    PpA = count['pA']/len(trajAB)
    PmA = count['mA']/len(trajAB)
    PpB = count['pB']/len(trajAB)
    PmB = count['mB']/len(trajAB)
    σzk = K * (PpA - PmA) * smartlog( PpA, PmA ) + K * (PpB - PmB) * smartlog( PpB, PmB )

    # σ_zk_A
    K = len(trajA)/tmax
    count = Counter(trajA)
    PpA = count['pA']/len(trajA)
    PmA = count['mA']/len(trajA)
    σzkA =  K * (PpA - PmA) * smartlog( PpA, PmA )

    # σti
    K = len(trajA)/tmax
    counts3 = count_pairs(traj, 3)
    A1 = smartlog( counts3[ ('mA', 'pB', 'pC') ], counts3[ ('mC', 'mB', 'pA') ])
    A2 = smartlog( counts3[ ('pC', 'mA', 'pB') ], counts3[ ('mB', 'pA', 'mC') ])
    A3 = smartlog( counts3[ ('pB', 'pC', 'mA') ], counts3[ ('pA', 'mC', 'mB') ])
    affinity = np.mean([A1, A2, A3])
    J = - K * (PpA - PmA)


    # σwt_A
    σwt_A = 0
    Dx_A = 0
    Dt_A = 0
    K_A = len(trajA)/tmax
    Xspace_A = ['pA', 'mA']
    counts2 = count_pairs(trajA, 2)
    Pt = [ [[],[]],
           [[],[]] ]
    for i in range(len(timesA)-1):
        x1 = x_dict[ trajA[i] ]
        x2 = x_dict[ trajA[i+1] ]
        Pt[x1][x2].append( timesA[i+1] - timesA[i] )
    for x1 in Xspace_A:
        for x2 in Xspace_A:
            if x1 != rev[x2]:
                inc1 = counts2[x1,x2] * smartlog( counts2[x1,x2] , counts2[rev[x2],rev[x1]] )
                σwt_A += inc1
                Dx_A += inc1
                kld = KLD_PerezCruz_2(Pt[ x_dict[x1] ][ x_dict[x2] ], Pt[ x_dict[rev[x2]] ][ x_dict[rev[x1]] ])
                σwt_A += counts2[x1,x2] * kld
                Dt_A += counts2[x1,x2] * kld
    σwt_A *= K_A/2
    
    # σwt
    σwt = 0
    Dx = 0
    Dt = 0
    K = len(trajAB)/tmax
    Xspace = ['pA', 'mA', 'pB', 'mB']
    counts2 = count_pairs(trajAB, 2)
    Pt = [ [[],[],[],[]],
           [[],[],[],[]],
           [[],[],[],[]],
           [[],[],[],[]] ]
    for i in range(len(timesAB)-1):
        x1 = x_dict[ trajAB[i] ]
        x2 = x_dict[ trajAB[i+1] ]
        Pt[x1][x2].append( timesAB[i+1] - timesAB[i] )
    for x1 in Xspace:
        for x2 in Xspace:
            if x1 != rev[x2]:
                inc1 = counts2[x1,x2] * smartlog( counts2[x1,x2] , counts2[rev[x2],rev[x1]] )
                σwt += inc1
                Dx += inc1
                kld = KLD_PerezCruz_2(Pt[ x_dict[x1] ][ x_dict[x2] ], Pt[ x_dict[rev[x2]] ][ x_dict[rev[x1]] ])
                σwt += counts2[x1,x2] * kld
                Dt += counts2[x1,x2] * kld
    σwt *= K/2 

    vec.append( [Δμ, epr, σzk, σzkA, J, A1, A2, A3, σwt_A, Dx_A, Dt_A, σwt, Dx, Dt] )


# exact EPR
nX_sol = sci.optimize.fsolve(lambda nX: nX**2 * ( k1*k3*A + k2*k3*B - (k3*km1 + km2*km3)*nX ) + km2 * (k1*A - km1*nX)*V**2, 10)
J_exact = km1*nX_sol[0] - k1*A
epr_exact = J_exact * Δμ

epr_mean = np.mean([x[1] for x in vec])
epr_std = np.std([x[1] for x in vec])

σzk_mean = np.mean([x[2] for x in vec])
σzk_std = np.std([x[2] for x in vec])

σzkA_mean = np.mean([x[3] for x in vec])
σzkA_std = np.std([x[3] for x in vec])

σti1s = [ x[4]*x[5] for x in vec]
σti1s_mean = np.mean(σti1s)
σti1s_std = np.std(σti1s)

σti2s = [ x[4]*x[6] for x in vec]
σti2s_mean = np.mean(σti2s)
σti2s_std = np.std(σti2s)

σti3s = [ x[4]*x[7] for x in vec]
σti3s_mean = np.mean(σti3s)
σti3s_std = np.std(σti3s)

σwtA_mean = np.mean([x[8] for x in vec])
σwtA_std = np.std([x[8] for x in vec])

DxA_mean = np.mean([x[9] for x in vec])
DxA_std = np.std([x[9] for x in vec])

DtA_mean = np.mean([x[10] for x in vec])
DtA_std = np.std([x[10] for x in vec])

σwt_mean = np.mean([x[11] for x in vec])
σwt_std = np.std([x[11] for x in vec])

Dx_mean = np.mean([x[12] for x in vec])
Dx_std = np.std([x[12] for x in vec])

Dt_mean = np.mean([x[13] for x in vec])
Dt_std = np.std([x[13] for x in vec])

res = [vec[0][0], epr_mean, epr_std, σzk_mean, σzk_std, σzkA_mean, σzkA_std, σti1s_mean, σti1s_std, σti2s_mean, σti2s_std, σti3s_mean, σti3s_std, σwtA_mean, σwtA_std, DxA_mean, DxA_std, DtA_mean, DtA_std, σwt_mean, σwt_std, Dx_mean, Dx_std, Dt_mean, Dt_std, epr_exact ]

with open("bruss_simulations_2_test.txt", "ab") as f:
    f.write(b"\n")
    np.savetxt(f, res, newline=", ", fmt='%5.12f')