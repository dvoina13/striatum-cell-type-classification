import numpy as np
import matplotlib.pyplot as plt

def heaviside(x):
    return 0.5 * (np.sign(x) + 1)

def cosbump(x,alpha):
    ee = .0000001;
    y = heaviside(x-ee)*heaviside(alpha*np.log(x+ee)+np.pi)*heaviside(-alpha*np.log(x+ee)+np.pi)*(1 + np.cos(alpha*np.log(x+ee)))/2
    return y

def create_basis_functions(n_basis, hist_bins, bas_type = 'exponential', tau_scale = 'logarithmic', delay = 0, alpha = 1.5):
    """
    Create exponential basis functions to model filters
    Inputs:
    n_basis: the number of basis functions to create
    hist_bins: the number of bins to use from the stimulus history, determines length of filter basis functions
    tau_scale: linear or logarithmic, how to determine time scales of exponential basis funcs

    Outputs:
    K: the basis functions - 2D ndarray of shape (hist_bins, n_basis)
    """
    K = np.zeros((hist_bins, n_basis))
    if tau_scale == 'logarithmic':
        bas = np.power((hist_bins),1.0/(n_basis-1))
        bas = np.repeat(bas, n_basis)
        tau_k = np.power(bas, range(n_basis))
    elif tau_scale == 'linear':
        tau_k = np.linspace(1, hist_bins, n_basis)
    ts = np.arange(hist_bins).reshape((hist_bins,1))
    if bas_type == 'exponential':
        K = np.tile(-ts+delay, n_basis)*1.0/tau_k
        K = np.exp(K)
        K[:delay, :] = 0
    elif bas_type == 'cosine_bump':
        K_b = np.tile(ts-delay, n_basis)*1.0/tau_k
        K_b[K_b<0] = 0
        K = cosbump(K_b, alpha)
    return K

def construct_design_matrix_basis_funcs(running_speed, binned_spikes, n_basis, num_bins, bas_type='exponential', tau_scale = 'logarithmic', delay = 0):
    """
    Construct the design matrix spiking response to running by creating basis functions to fit to running response
    Inputs:
    running_speed: a (T,) array
    binned_spikes: spikes binned over same time period as running
    n_basis: number of basis functions to fit to each filter
    num_bins: integer - how many time bins of history to use as predictors (will use past and future running speed)

    Outputs:
    y: the dependent variable to be predicted (spiking response) - (T-2d,) ndarray
    X_dsn: the design matrix of independent variables - 2D ndarray
    K: the basis functions being fit to the data
    """
    T = len(running_speed)
    y = binned_spikes[num_bins:]
    X_dsn = np.ones((T-num_bins,1))
    K = []
    
    # design matrix for running
    K_run = create_basis_functions(n_basis, num_bins, bas_type = bas_type, tau_scale = tau_scale, delay = delay)
    X_run = np.zeros((T-num_bins, n_basis))
    for n in range(n_basis):
        conv = np.convolve(running_speed, K_run[:,n])
        X_run[:,n] = conv[num_bins:T]
    X_dsn = np.hstack((X_run, X_dsn))
    K.append(K_run)
    
    # design matric for spike history
    K_spike = create_basis_functions(n_basis, num_bins, tau_scale = tau_scale)
    X_spike = np.zeros((T-num_bins, n_basis))
    for n in range(n_basis):
        conv = np.convolve(binned_spikes, K_spike[:,n])
        X_spike[:,n] = conv[num_bins:T]
    X_dsn = np.hstack((X_spike, X_dsn))
    K.append(K_spike)
    return y, X_dsn, K

def construct_design_matrix(explanatory_vars, binned_spikes, num_hist_bins):
    """
    Construct the design matrix spiking response to running by directly using the spike and running data
    Inputs:
    explanatory_vars: a (T,) array or measurements used to model spiking
    binned_spikes: spikes binned over same time period as stim
    num_hist_bins: integer - how many time bins of history to use as predictors

    Outputs:
    y: the dependent variable to be predicted (spiking response) - (T-num_hist_bins,) ndarray
    X_dsn: the design matrix of independent variables - 2D ndarray
    """
    
    T = len(explanatory_vars)    
    y = binned_spikes[num_hist_bins:]
    X_dsn = np.ones((T-num_hist_bins,(2*num_hist_bins)+1)) #the last column is constant 1s
    #print(f"Constructing design matrix ({T-num_hist_bins} rows)")
    for t in range(T-num_hist_bins):
        #stim inputs
        X_dsn[t,:num_hist_bins] = np.flip(explanatory_vars[t:t+num_hist_bins])
        #spike history inputs
        X_dsn[t,num_hist_bins:-1] = np.flip(binned_spikes[t:t+num_hist_bins])
        #print(f"row {t}")
    return y, X_dsn