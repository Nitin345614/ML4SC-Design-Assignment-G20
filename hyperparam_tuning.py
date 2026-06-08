# Import the relevant libraries for this exercise session
import time, tqdm
import numpy as np
from scipy.io import loadmat
import pandas as pd
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared
from sklearn.preprocessing import StandardScaler
from wakepy import keep # To prevent the computer from sleeping during long hyperparameter tuning runs
import GPy

# To supress some of the GPy warnings later on.
import warnings
warnings.filterwarnings("ignore")

out = np.load('../gym-unbalanced-disk/disc-benchmark-files/training-val-test-data.npz')

data_amount = len(out['th']) # Use all available data
filename = f"SK_bcm_results_{data_amount}_datapoints.txt"

u_data = out['u'][:data_amount] #u[0],u[1],u[2],u[3],...
th_data = out['th'][:data_amount] #th[0],th[1],th[2],th[3],...

def create_IO_data(u,y,na,nb) -> tuple:
    X = []
    Y = []
    for k in range(max(na,nb), len(y)):
        X.append(np.concatenate([u[k-nb:k],y[k-na:k]]))
        Y.append(y[k])
    return np.array(X), np.array(Y)

def create_split_datasets(u,y,na,nb,split) -> tuple:
    split_index = int(len(u)*split) 
    Xtrain, Ytrain = create_IO_data(u[:split_index],y[:split_index], na, nb) 
    Xval,   Yval   = create_IO_data(u[split_index:],y[split_index:], na, nb)
    return Xtrain, Ytrain, Xval, Yval

def simulation_IO_model(f, ulist, ylist, na, nb, skip=50):
    upast = ulist[skip-nb:skip].tolist() #good initialization
    ypast = ylist[skip-na:skip].tolist()
    Y = ylist[:skip].tolist()
    for u in tqdm.tqdm(ulist[skip:]):
        x = np.concatenate([upast,ypast],axis=0)
        y_pred_mu = f(x)
        Y.append(y_pred_mu)
        upast.append(u)
        upast.pop(0)
        ypast.append(y_pred_mu)
        ypast.pop(0)
    return np.array(Y)

def dataset_factory(x, y, na, nb, split):
    split_index = int(len(x)*split) 
    Xtrain, Ytrain = create_IO_data(x[:split_index],y[:split_index], na, nb) 
    Xval,   Yval   = create_IO_data(x[split_index:],y[split_index:], na, nb)
    x_sim, y_sim = x[split_index:], y[split_index:]
    return Xtrain, Ytrain, Xval, Yval, x_sim, y_sim

def custom_kernel_factory(na, nb):
    k_rbf = GPy.kern.RBF(input_dim=(na+nb), lengthscale=0.1, ARD=True)
    k_white = GPy.kern.White(input_dim=(na+nb), variance=0.01)
    th_dim = np.arange(na+nb)[-na:]  # The last 'na' dimensions correspond to the output history, I think those are periodic
    k_periodic = GPy.kern.StdPeriodic(input_dim=len(th_dim), active_dims=th_dim, lengthscale=0.1, period=4.2)
    return (k_rbf * k_periodic) + k_white
    
def GP_prediction(model, X, Y):
    N_show = 500

    Y_pred_mu, Y_pred_var = model.predict(X)
    Y_pred_mu = Y_pred_mu.squeeze()
    Y_pred_std = np.sqrt(Y_pred_var.squeeze())
    Y = Y.squeeze()

    plt.figure(figsize=(12,5))

    plt.plot(Y_pred_mu[:N_show], label="GP mean")
    plt.plot(Y[:N_show], label="True output")

    plt.fill_between(
        np.arange(len(Y_pred_mu[:N_show])),
        Y_pred_mu[:N_show] - 2*Y_pred_std[:N_show],
        Y_pred_mu[:N_show] + 2*Y_pred_std[:N_show],
        color='blue',
        alpha=0.3,
        label="95% confidence"
    )

    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Output")
    plt.title("NARX GP prediction with uncertainty on the validation dataset")
    plt.show()

    print('RMS:', np.mean((Y_pred_mu-Y)**2)**0.5,'radians')
    print('RMS:', np.mean((Y_pred_mu-Y)**2)**0.5/(2*np.pi)*360,'degrees')

def GP_simulation(model, u, th, na, nb):
    N_show = 500
    skip = max(na,nb)
    th_sim = simulation_IO_model(lambda x: model.predict(x[None,:])[0][0][0], u, th, na, nb, skip=skip)

    plt.figure(figsize=(12,5))
    plt.title('Simulation on the validation dataset')
    plt.plot(th[:N_show], label='measured')
    plt.plot(th_sim[:N_show], label='simulated')
    plt.grid()
    plt.xlabel('sample')
    plt.ylabel('th')
    plt.legend()
    plt.show() 


    print('Simulation errors:')
    print('RMS:', np.mean((th_sim[skip:]-th[skip:])**2)**0.5,'radians')
    print('RMS:', np.mean((th_sim[skip:]-th[skip:])**2)**0.5/(2*np.pi)*360,'degrees')
    # print('NRMS:', np.mean((th_sim[skip:]-th[skip:])**2)**0.5/th.std()*100,'%')

class BCM(BaseEstimator, RegressorMixin):

    def __init__(self, n_experts=5, na=3, nb=2, kernel=None, random_state=None):
        self.n_experts = n_experts
        self.na = na
        self.nb = nb
        if kernel is None:
            self.kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2)
        else:
            self.kernel = kernel
        self.random_state = random_state
        self.experts = []

        # Initialize tracking scalers as class attributes
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X, y):
        # 1. Fit and transform the training partitions
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        X_chunks = np.array_split(X_scaled, self.n_experts)
        y_chunks = np.array_split(y_scaled, self.n_experts)

        self.experts = []
        print(
            f"Training {self.n_experts} expert GPs on {X.shape[0]} samples, chunk size ~{X_chunks[0].shape[0]}x({self.na}+{self.nb}) = {X_chunks[0].shape[0] * (self.na + self.nb)} samples each."
        )
        for i in tqdm.tqdm(range(self.n_experts)):
            gp = GaussianProcessRegressor(
                kernel=self.kernel,
                normalize_y=False,  # Set to False because we are manually handling target normalization
                alpha=0.0,
                random_state=self.random_state,
            )
            gp.fit(X_chunks[i], y_chunks[i])
            self.experts.append(gp)

        return self

    def predict(self, X, return_std=False):
        # Handle 1D inputs (like a single step from simulation loop) by shaping to 2D
        if X.ndim == 1:
            X = X[None, :]

        # 2. Automatically scale the incoming raw feature matrices
        X_scaled = self.scaler_X.transform(X)
        p = self.n_experts
        N_test = X_scaled.shape[0]

        sum_prec_times_mean = np.zeros(N_test)
        sum_precision = np.zeros(N_test)

        # 3. Gather predictions in the scaled space
        for gp in self.experts:
            mean_scaled, std_scaled = gp.predict(X_scaled, return_std=True)
            std_scaled = np.clip(std_scaled, 1e-8, None)

            precision_scaled = 1.0 / (std_scaled**2)
            sum_precision += precision_scaled
            sum_prec_times_mean += precision_scaled * mean_scaled

        # 4. Compute prior covariance using the scaled coordinate grid
        K_xx = self.experts[0].kernel_(X_scaled)
        K_star_star_diag = np.diag(K_xx)
        K_prior_precision = 1.0 / np.clip(K_star_star_diag, 1e-8, None)

        # 5. Apply BCM Aggregation (still in scaled space)
        bcm_precision_scaled = -(p - 1) * K_prior_precision + sum_precision
        bcm_precision_scaled = np.clip(bcm_precision_scaled, 1e-8, None)

        bcm_var_scaled = 1.0 / bcm_precision_scaled
        bcm_mean_scaled = bcm_var_scaled * sum_prec_times_mean

        # 6. Inverse Transform everything back to physical target units (Radians)
        # Reshape to 2D for scikit-learn's scaler, then flatten back
        bcm_mean_raw = self.scaler_y.inverse_transform(
            bcm_mean_scaled.reshape(-1, 1)
        ).flatten()

        if return_std:
            # Standard deviation scales linearly with the target scaling factor
            # (i.e., standard deviation multiplied by the target's original standard deviation scale)
            bcm_std_scaled = np.sqrt(bcm_var_scaled)
            bcm_std_raw = bcm_std_scaled * np.sqrt(self.scaler_y.var_[0])
            return bcm_mean_raw, bcm_std_raw

        return bcm_mean_raw

    def prediction_error(self, Xval, Yval):
        # Takes unscaled raw inputs, evaluates, and returns true error in radians
        Y_pred = self.predict(Xval, return_std=False)
        rms_error = np.mean((Y_pred - Yval) ** 2) ** 0.5
        return rms_error

    def simulation_error(self, u_val, th_val):
        print(f"[SIMULATION]")
        skip = max(self.na, self.nb)

        # Because predict() automatically handles scaling and inverse scaling,
        # our simulation lambda can just interact with pure raw arrays directly!
        th_train_sim = simulation_IO_model(
            lambda x: self.predict(x[None, :])[0],
            u_val,
            th_val,
            self.na,
            self.nb,
            skip=skip,
        )

        rms_error = np.mean((th_train_sim[skip:] - th_val[skip:]) ** 2) ** 0.5
        return rms_error

class GPy_BCM:

    def __init__(self, n_experts=5, na=2, nb=2, kernel=None, M=None, random_state=None):
        self.n_experts = n_experts
        self.na = na
        self.nb = nb
        self.M = M
        self.kernel = kernel  # Expecting the composite kernel passed from outside
        self.random_state = random_state
        self.experts = []
        if self.M is None:
            print("[FULL GP]")
        else:
            print(f"[SPARSE GP with {self.M} inducing points]")

    def fit(self, X, y):
        X_chunks = np.array_split(X, self.n_experts)
        y_chunks = np.array_split(y, self.n_experts)
        print(f"Training {self.n_experts} expert GPs on {X.shape[0]} samples, chunk size ~{X_chunks[0].shape[0]}x({self.na}+{self.nb}) = {X_chunks[0].shape[0] * (self.na + self.nb)} samples each.")

        self.experts = []
        for i in tqdm.tqdm(range(self.n_experts)):
            # Crucial: Deep copy or recreate the kernel so experts don't overwrite each other's weights
            expert_kernel = self.kernel.copy()

            if self.M is None or X_chunks[i].shape[0] < self.M:
                # Using full GP for each expert (only feasible for smaller datasets)
                gp = GPy.models.GPRegression(
                    X_chunks[i],
                    y_chunks[i].reshape(-1, 1),
                    kernel=expert_kernel,
                )
            else:
                # Using sparse GP for each expert to handle larger datasets efficiently
                gp = GPy.models.SparseGPRegression(
                    X_chunks[i],
                    y_chunks[i].reshape(-1, 1),
                    num_inducing=self.M,
                    kernel=expert_kernel,
                )

            if self.M is None or X_chunks[i].shape[0] < self.M:
                gp.optimize("bfgs", messages=False)
            else:
                # 'scg' (Scaled Conjugate Gradient) is often more stable for SparseGPs
                gp.optimize("scg", messages=False)
                
            self.experts.append(gp)

        return self

    def predict(self, X, return_std=False):
        p = self.n_experts
        N_test = X.shape[0]

        # FIX 1: Initialize as 2D column vectors (shape: 8748, 1) to match GPy outputs
        sum_prec_times_mean = np.zeros((N_test, 1))
        sum_precision = np.zeros((N_test, 1))

        # 1. Gather predictions and precisions from all experts
        for gp in self.experts:
            # FIX 2: GPy returns (mean, variance) -> NOT standard deviation!
            mean, var = gp.predict(X)
            var = np.clip(var, 1e-8, None)

            # Since 'var' is already variance, do not square it!
            precision = 1.0 / var  # Shape: (8748, 1)
            sum_precision += precision
            sum_prec_times_mean += precision * mean

        # 2. Compute the Prior Covariance matrix K_** at the test points
        # FIX 3: In GPy, the syntax is .K(X) instead of .kernel_(X)
        K_xx_diag = self.experts[0].kern.Kdiag(X).reshape(-1, 1)
        K_prior_precision = 1.0 / np.clip(K_xx_diag, 1e-8, None)

        # 3. Apply Equation (8.30): Subtract the overcounted prior precision
        bcm_precision = -(p - 1) * K_prior_precision + sum_precision
        bcm_precision = np.clip(
            bcm_precision, 1e-8, None
        )  # Prevent negative variance

        # Calculate final predictive variance
        bcm_var = 1.0 / bcm_precision

        # 4. Apply Equation (8.29): Weighted predictive mean
        bcm_mean = bcm_var * sum_prec_times_mean

        # Flatten outputs back to 1D arrays for easy plotting/compatibility
        if return_std:
            return bcm_mean.flatten(), np.sqrt(bcm_var).flatten()
        return bcm_mean.flatten()

    def prediction_error(self, Xval, Yval):
        Y_pred, Y_pred_std = self.predict(Xval, return_std=True)
        rms_error = np.mean((Y_pred - Yval)**2)**0.5
        return rms_error

    def simulation_error(self, u_val, th_val):
        print(f"[SIMULATION]")
        skip = max(self.na, self.nb)
        th_train_sim = simulation_IO_model(lambda x: self.predict(x[None,:])[0], u_val, th_val, self.na, self.nb, skip=skip)
        rms_error = np.mean((th_train_sim[skip:]-th_val[skip:])**2)**0.5
        return rms_error


df = pd.DataFrame(columns=['na', 'nb', 'n_experts', 'PRED', 'SIM', 'time_taken'])

na_list = [2,3,4,5,6]
nb_list = [2,3,4,5,6]
nr_expert_list = [40,20,10]
M_list = [500,1000]

with open(filename, 'w') as f:
    f.write("na, nb, nr_expert, PRED, SIM, time_taken\n")

# for M in M_list:
# This context manager prevents the computer from sleeping while inside the block
with keep.presenting():
    for nr_expert in nr_expert_list:
        for na in na_list:
            for nb in nb_list:
                # custom_kernel = custom_kernel_factory(na, nb)
                custom_kernel = (RBF(length_scale=0.1) * ExpSineSquared(length_scale=4.0)) + WhiteKernel(noise_level=0.01)
                bcm = BCM(n_experts=nr_expert, na=na, nb=nb, kernel=custom_kernel, random_state=42)
                Xtrain, Ytrain, Xval, Yval, x_sim, y_sim = dataset_factory(u_data, th_data, na, nb, split=0.8)
                start_time = time.time()
                bcm.fit(Xtrain, Ytrain)
                error_pred = bcm.prediction_error(Xval, Yval)
                error_sim = bcm.simulation_error(x_sim, y_sim)
                end_time = time.time()
                print(f"na={bcm.na}, nb={bcm.nb}, nr_expert={nr_expert:04d}, PRED: {error_pred:.4f} radians, SIM: {error_sim:.4f} radians")
                new_row = {
                    'na': bcm.na,
                    'nb': bcm.nb,
                    'n_experts': bcm.n_experts,
                    # 'M': bcm.M,
                    'PRED': error_pred,
                    'SIM': error_sim,
                    'time_taken': end_time - start_time
                }
                df.loc[len(df)] = new_row
                with open(filename, 'a') as f:
                    f.write(f"{bcm.na}, {bcm.nb}, {bcm.n_experts}, {error_pred:.4f}, {error_sim:.4f}, {end_time - start_time:.4f}\n")

df.to_csv(filename.replace('.txt', '.csv'), index=False)