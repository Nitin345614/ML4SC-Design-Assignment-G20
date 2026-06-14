from matplotlib import pyplot as plt # plotting
import math
import sklearn
#import GPy
#np.random.seed(101)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel, DotProduct, Matern
from sklearn.cluster import KMeans


import scipy
from tqdm import tqdm
#import pytorch as torch
import numpy as np
import gymnasium as gym
import gym_unbalanced_disk, time


def rbf(x1,x2,width):
    return np.exp(-0.5* (x1-x2)**2/width)

def demonstrate(full_range,y1,f,var,indices,name):
    plt.figure(figsize=(16,4.8))
    plt.title(name)
    plt.plot(full_range,y1,label="Data",marker=".")
    plt.plot(full_range[indices],f,label="Prediction")
    plt.fill_between(full_range[indices],f+var,f-var,alpha=0.3,label='est std out')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()
    return

def demonstrate_alternate(part_range, y1,f,var, name):
    plt.figure(figsize=(16,4.8))
    plt.title(name)
    plt.plot(part_range,y1,label="Data")
    plt.plot(part_range,f,label="Prediction")
    plt.fill_between(part_range,f+var,f-var,alpha=0.3,label='est std out')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()
    return

def quick_plot(full_range,f,var,name):
    plt.figure(figsize=(16,4.8))
    plt.title(name)
    plt.plot(full_range,f,label="Prediction")
    plt.fill_between(full_range,f+var,f-var,alpha=0.3,label='est std out')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.grid()
    plt.show()
    return

def quick_plot_alternative(full_range,f,var,index,name):
    plt.figure(figsize=(16,4.8))
    plt.title(name)
    plt.plot(full_range,f,label="Prediction")
    plt.fill_between(full_range,f+var,f-var,alpha=0.3,label='est std out')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.grid()
    plt.show()
    return

def state_prep(data, time, order):
        time += 1
        entries = np.min([time,order])
        state = np.zeros((order,2))
        state[(order-entries):order,:] = data[(time-entries):time,:]
        return np.concat((state[:,0].reshape(1,order),state[:,1].reshape(1,order)),axis=1)

def state_prep_loop(data, order):
    states = np.zeros((len(data),2*order))
    for time in range(len(data)): states[time] = state_prep(data, time, order)
    return states

N=28000
V=7000

full_range = range(N+V)

readout = np.loadtxt('/home/sodium-nitrate/aur/gym-unbalanced-disk/disc-benchmark-files/training-val-test-data.csv',delimiter=',',dtype=float)

readout_u = readout[:,0]
readout_y = readout[:,1]


#norm_u = (np.max(readout_u ) - np.min(readout_u ))
#norm_y = (np.max(readout_y ) - np.min(readout_y ))

norm_u = np.std(readout_u)
norm_y = np.std(readout_y)

mean_u = np.mean(readout_u)
mean_y = np.mean(readout_y)

#readout_u = (readout_u-mean_u)/norm_u
#readout_y = (readout_y-mean_y)/norm_y

readout_u_tra = readout_u[:N]
readout_y_tra = readout_y[:N]

readout_u_val = readout_u[N:]
readout_y_val = readout_y[N:]

order = 5


x_tra = state_prep_loop(np.concat((readout_u_tra[:,None],readout_y_tra[:,None]),axis=1), order)
y_tra = readout_y_tra[:,None]

x_val = state_prep_loop(np.concat((readout_u_val[:,None],readout_y_val[:,None]),axis=1), order)
y_val = readout_y_val[:,None]


range_tra = np.array(range(N))
range_val = np.array(range(V))

kernel = rbf

#print(readout_u_val)
#print(readout_y_val)
#print(x_val)

# Setup

kernel = rbf
width_bcm = 0.5
noise_bcm = 0.1

p_bcm = 190 #no of partitions
M_bcm = 350 #partition block size

N_div = 40
V_div = 1

testpoints_ind_bcm = (np.mod(range(N),N_div)==0)
valipoints_ind_bcm = (np.mod(range(V),V_div)==0)


x_test = x_tra[testpoints_ind_bcm,:]
y_test = y_tra[testpoints_ind_bcm,:]

x_vali = x_val[valipoints_ind_bcm,:]
y_vali = y_val[valipoints_ind_bcm,:]


#[p*M_bcm:((p+1)*M_bcm),:]

#Try to combine SD with BCM?

#hyperinclude_bcm = np.ones([N,p_bcm],dtype='bool')

# + *ExpSineSquared(length_scale=0.5,periodicity=0.5)


initial_length_scales = np.ones(2*order)*0.5

ker_bcm =  WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)) + Matern(length_scale=initial_length_scales, nu=2.5)  + ConstantKernel(constant_value_bounds=(0.01,10))*ExpSineSquared(length_scale=0.5,periodicity=0.5) #DotProduct()   #a=)

def BCM_predict(x22,reg_bcm, use_norm=False, norm=None, supress_nan=True):#, norm_y = None):
    t_bcm = x22.shape[0]
    prediction_blocks = len(reg_bcm)

    #Initialisation
    f_bcm = np.array(np.zeros([t_bcm, prediction_blocks]))
    f_bcm_run = np.array(np.zeros(t_bcm))
    var_bcm = f_bcm
    precision = np.array(np.zeros([t_bcm, prediction_blocks]))
    precision_run = np.array(np.zeros([t_bcm]))

    #accumulating and combining predictions
    for i in range(prediction_blocks):
        f_bcm[:,i], precision[:,i] = reg_bcm[i].predict(x22,return_std=True)
        
        #sanity checker ignores the results from experts giving NaN or divide by 0 outputs
        if (precision[:,i].all()>0) and (f_bcm[:,i].all() == f_bcm[:,i].all()): 
            precision[:,i] = precision[:,i]**-1
        else: 
            f_bcm[:,i], precision[:,i] = 0, 0
            prediction_blocks += -1             #This accounts for the reduced number of experts for confidence calculations
        
        precision_run = precision_run + precision[:,i]
        f_bcm_run = f_bcm_run + precision[:,i]*f_bcm[:,i]
    
    #precision_run = precision_run**(1/beta)
    #print(precision_run)
    #K22_factor = np.diag(np.linalg.inv(reg_bcm[0].kernel_(x22)))
    
    K22_factor = np.ones(t_bcm)

    cov_bcm = ( -(prediction_blocks-1)*(K22_factor) + precision_run)**-1

    var_bcm = cov_bcm
    f_bcm = (cov_bcm*f_bcm_run)
    #f_bcm = f_bcm/np.std(f_bcm)

    #mod f_bcm
    if use_norm: f_bcm = f_bcm%(np.sign(f_bcm)*math.pi)

    #if task_mode: return f_bcm#*norm_y
    return f_bcm, var_bcm#/np.std(f_bcm)

def BCM_train(x1,y1,ker_bcm,p_bcm,N,n_restarts_optimizer=10):
    #yeah it's training idk
    reg_bcm = np.zeros([p_bcm],dtype='object')

    #labels = quick_cluster(x1,y1,p_bcm)
    
    for p in tqdm(range(p_bcm)):
        #x11 = x1[np.mod(np.array(range(N))+p,p_bcm)==0]
        #y11 = y1[np.mod(np.array(range(N))+p,p_bcm)==0]
        
        x11 = x1[int(p*N/p_bcm):int((p+1)*N/p_bcm)]
        y11 = y1[int(p*N/p_bcm):int((p+1)*N/p_bcm)]

        #x11 = x1[labels == p]
        #y11 = y1[labels == p]
        
        reg_bcm[p] = GaussianProcessRegressor(ker_bcm,alpha=10**-6, n_restarts_optimizer=n_restarts_optimizer, normalize_y=True) #a=)
        reg_bcm[p].fit(x11,y11) #a)
    return reg_bcm

def quick_cluster(x,y, num):
    kmeans = KMeans(n_clusters=num, random_state=0, n_init="auto")
    kmeans.fit(x)
    labels = kmeans.labels_
    return labels

reg_bcm_pred = BCM_train(x_tra,y_tra,ker_bcm,p_bcm,N)


data = np.load('/home/sodium-nitrate/aur/gym-unbalanced-disk/disc-benchmark-files/hidden-test-prediction-submission-file.npz')
upast_test = data['upast'] #N by u[k-15],u[k-14],...,u[k-1]
thpast_test = data['thpast'] #N by y[k-15],y[k-14],...,y[k-1]
Xtest = np.concatenate([upast_test[:,15-order:15], thpast_test[:,15-order:15]],axis=1)

#Ypredict, Varpredict = reg_bcm[0].predict(Xtest,return_std=True)
Ypredict, Varpredict = BCM_predict(Xtest,reg_bcm_pred,kernel,width_bcm)

np.savez('NITIN-hidden-test-prediction-example-submission-file.npz', upast=upast_test, thpast=thpast_test, thnow=Ypredict)

quick_plot(range(len(Xtest)),Ypredict,Varpredict,'predictions idk')
