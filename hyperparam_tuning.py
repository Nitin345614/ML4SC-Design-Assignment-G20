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
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from wakepy import keep # To prevent the computer from sleeping during long hyperparameter tuning runs
import GPy

# To supress some of the GPy warnings later on.
import warnings
warnings.filterwarnings("ignore")

out = np.load('../gym-unbalanced-disk/disc-benchmark-files/training-val-test-data.npz')


u_data = out['u'] #u[0],u[1],u[2],u[3],...
th_data = out['th'] #th[0],th[1],th[2],th[3],...

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
    

def GPy_prediction_error(model, u, th, na, nb):
    Y_pred_mu, Y_pred_var = model.predict(X)
    Y_pred_mu = Y_pred_mu.squeeze()
    Y = Y.squeeze()
    rms_error = np.mean((Y_pred_mu-Y)**2)**0.5
    return rms_error

def GPy_simulation_error(model, u, th, na, nb):
    skip = max(na,nb)
    th_sim = simulation_IO_model(lambda x: model.predict(x[None,:])[0][0][0], u, th, na, nb, skip=skip)
    rms_error = np.mean((th_sim[skip:]-th[skip:])**2)**0.5
    return rms_error


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



if __name__ == "__main__":

    na_list = [2,3,4,5,6]
    nb_list = [2,3,4,5,6]
    nr_expert_list = [40,20,10]
    M_list = [500,1000,1500]

    df_sparse = pd.DataFrame(columns=['M', 'na', 'nb', 'PRED', 'SIM', 'time_taken'])

    filename = f"Sparse_GP_results_Matern.txt"

    with open(filename, 'w') as f:
        f.write("M, na, nb, PRED, SIM, time_taken\n")
    # This context manager prevents the computer from sleeping while inside the block
    with keep.presenting():
        for M in M_list:
            for na in na_list:
                for nb in nb_list:
                    # Data splitting
                    Xtrain, Ytrain, Xval, Yval, x_sim, y_sim = dataset_factory(u_data, th_data, na, nb, split=0.8)

                    # Kernel construction
                    custom_kernel = GPy.kern.Matern52(input_dim=(na+nb), ARD=True) + GPy.kern.White(input_dim=(na+nb))

                    # Model initialization, training, prediction error evaluation, and simulation error evaluation
                    model = GPy.models.SparseGPRegression(Xtrain,Ytrain.reshape(-1,1),num_inducing=M, kernel=custom_kernel)
                    
                    start_time = time.time()
                    
                    model.optimize('scg', messages=True, max_iters=100)

                    optimized_inducing_points = Sparse_GP_model.inducing_inputs.values.copy()

                    # Save the array to a file on your hard drive
                    np.save('saved-inducing-points/optimized_inducing_points_' + str(M) + '_na' + str(na) + '_nb' + str(nb) + '.npy', optimized_inducing_points)
                    print(f"Saved {M} inducing points to disk. Shape: {optimized_inducing_points.shape}")


                    error_pred = GPy_prediction_error(model, Xval, Yval, na, nb)
                    error_sim = GPy_simulation_error(model, x_sim, y_sim, na, nb)
                    
                    end_time = time.time()

                    # Save results to DataFrame and text file
                    print(f"M={M}, na={na}, nb={nb}, PRED: {error_pred:.4f} radians, SIM: {error_sim:.4f} radians")
                    new_row = {
                        'M': M,
                        'na': na,
                        'nb': nb,
                        'PRED': error_pred,
                        'SIM': error_sim,
                        'time_taken': end_time - start_time
                    }
                    df_sparse.loc[len(df_sparse)] = new_row
                    with open(filename, 'a') as f:
                        f.write(f"{M}, {na}, {nb}, {error_pred:.4f}, {error_sim:.4f}, {end_time - start_time:.4f}\n")


    df_sparse.to_csv(filename.replace('.txt', '.csv'), index=False)




    filename = f"SK_bcm_results_Matern.txt"

    df_bcm = pd.DataFrame(columns=['na', 'nb', 'n_experts', 'PRED', 'SIM', 'time_taken'])
    with open(filename, 'w') as f:
        f.write("na, nb, n_experts, PRED, SIM, time_taken\n")

    # This context manager prevents the computer from sleeping while inside the block
    with keep.presenting():
        for nr_expert in nr_expert_list:
            for na in na_list:
                for nb in nb_list:
                    # Data splitting
                    Xtrain, Ytrain, Xval, Yval, x_sim, y_sim = dataset_factory(u_data, th_data, na, nb, split=0.8)

                    # Kernel construction
                    # k_rbf = RBF(length_scale=0.1, length_scale_bounds=(1e-3, 1e3))
                    k_matern = Matern(length_scale=0.1, nu=2.5)
                    k_white = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 1e2)) # the BCM scales the data
                    custom_kernel = k_matern + k_white    

                    # Model initialization
                    bcm = BCM(n_experts=nr_expert, na=na, nb=nb, kernel=custom_kernel, random_state=42)

                    start_time = time.time()

                    # Model training, prediction error evaluation, and simulation error evaluation
                    bcm.fit(Xtrain, Ytrain)
                    error_pred = bcm.prediction_error(Xval, Yval)
                    error_sim = bcm.simulation_error(x_sim, y_sim)

                    end_time = time.time()

                    print(f"na={bcm.na}, nb={bcm.nb}, nr_expert={nr_expert:04d}, PRED: {error_pred:.4f} radians, SIM: {error_sim:.4f} radians")
                    new_row = {
                        'na': bcm.na,
                        'nb': bcm.nb,
                        'n_experts': bcm.n_experts,
                        'PRED': error_pred,
                        'SIM': error_sim,
                        'time_taken': end_time - start_time
                    }
                    df_bcm.loc[len(df_bcm)] = new_row
                    with open(filename, 'a') as f:
                        f.write(f"{bcm.na}, {bcm.nb}, {bcm.n_experts}, {error_pred:.4f}, {error_sim:.4f}, {end_time - start_time:.4f}\n")

    df_bcm.to_csv(filename.replace('.txt', '.csv'), index=False)



