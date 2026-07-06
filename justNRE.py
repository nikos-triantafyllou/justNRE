import numpy as np

def prepare_for_NRE(data_arr, params_arr):
    '''
    Prepares data and parameter arrays for NRE classification.
    Inputs: 
        - data_arr: np.array of shape: (number of sims, number of observables)
        - params_arr: np.array of shape: (number of sims, number of parameters)
    Outputs:
        - intsnces: the training/validation/testing data of the NRE (it is recommended to keep a different test set for coverage tests)
        - targets: whether the pair is jointly or independently drawn
        
    Specifically it performs the following steps:
        1. Reserves half of the data for the independently drawn samples
        2. Generates a permutation of indices and shuffles the indepenent part
        3. Introduce flags working as target fetures for the classifier and asosciate them with the independent and joint samples
        4. Shuffle all together in the end
    '''
    
    assert data_arr.shape[0] == params_arr.shape[0]
    
    # 1. Reserve half of the data for the independently drawn samples-----------------------------
    number_of_sims = data_arr.shape[0]
    print('number_of_simulations:', int(data_arr.shape[0]))
    half_of_sims = data_arr.shape[0]/2
    print('Half that is reserved for independetly drawn samples:', data_arr.shape[0]/2)
    
    # If the number of sims is not an even number just delete the last one for now
    if number_of_sims%2 != 0:
        data_arr   = data_arr[:number_of_sims-1, :]
        params_arr = params_arr[:number_of_sims-1, :]
        half_of_sims = data_arr.shape[0]/2
    
    print('Same but integer:', half_of_sims)
    
    # Shuffle the input data to be sure
    np.random.seed(42)
    permutation = np.random.permutation(params_arr.shape[0])
    data_arr_shuffled = data_arr[permutation, :]
    params_arr_shuffled = params_arr[permutation, :]

    # 1. Reserve half of the data for the independently drawn samples-----------------------------
    split_at = int(half_of_sims) #+1 to be half i just have 1 less due to indexing 
    
    # 2. Generate a permutation of indices and shuffles the indepenent part while keeping the joint part untouched-----------------------------

    # Take the joint samples and leave them as is
    data_arr_shuffled_joint    = data_arr_shuffled[:split_at]
    params_arr_shuffled_joint  = params_arr_shuffled[:split_at]

    # Take the independent samples and shuffle them
    np.random.seed(22)
    permutation = np.random.permutation(params_arr_shuffled[split_at:number_of_sims].shape[0])
    data_arr_shuffled_indep    = data_arr_shuffled[split_at:number_of_sims] # DO NOT SHUFFLE THIS ONLY THE PARAMS
    params_arr_shuffled_indep  = params_arr_shuffled[split_at:number_of_sims][permutation, :]
    
    # 3. Introduce flags working as target fetures for the classifier and asosciate them with the independent and joint samples
    flag_0 = np.zeros((split_at,1))
    flag_1 = np.ones((split_at,1))


    # Connect the 3 numpy arrays (data, params and flags) and shuffle them again in order to separate them afterwards (into x and y)
    joint_NRE_ready = np.concatenate((data_arr_shuffled_joint, params_arr_shuffled_joint), axis=1)
    joint_NRE_ready = np.concatenate((joint_NRE_ready, flag_1), axis=1)

    indep_NRE_ready = np.concatenate((data_arr_shuffled_indep, params_arr_shuffled_indep), axis=1)
    indep_NRE_ready = np.concatenate((indep_NRE_ready, flag_0), axis=1)

    all_unshuffled_NRE_ready = np.concatenate((joint_NRE_ready, indep_NRE_ready), axis=0)

    # 4. Shuffle all together in the end
    np.random.seed(11)
    permutation = np.random.permutation(all_unshuffled_NRE_ready.shape[0])
    all_shuffled_NRE_ready = all_unshuffled_NRE_ready[permutation, :]

    # Make 2 numpys training incances and targets
    instances = all_shuffled_NRE_ready[:,:-1]
    targets = all_shuffled_NRE_ready[:,-1]

    return instances, targets


   
    
def split(instances, targets, percentages=[70, 15, 15]):
    """
    Takes the first dimension of instances and targets and splits them 
    according to the specified percentages.
    
    Returns a list of tuples: [(X_train, y_train), (X_val, y_val), ...]
    """
    # 1. Normalize percentages to fractions just in case they don't sum to 100
    percentages = np.array(percentages) / np.sum(percentages)
    
    # 2. Calculate the exact integer cut-off indices based on dataset length
    total_len = len(instances)
    # np.cumsum(percentages)[:-1] gives [0.70, 0.85] for [70, 15, 15]
    split_indices = (np.cumsum(percentages)[:-1] * total_len).astype(int)
    
    # 3. Split both arrays at those exact positions
    instances_splits = np.split(instances, split_indices, axis=0)
    targets_splits = np.split(targets, split_indices, axis=0)
    
    # 4. Pair them up: [(X_1, y_1), (X_2, y_2), ...]
    return list(zip(instances_splits, targets_splits))



def normalize(datasets, norm_type='standard'):
    """
    Normalizes a tuple of datasets based on the first dataset (X_train).
    
    Returns:
        tuple: (normalized_datasets, denormalize_fn)
    """
    X_train = datasets[0]
    
    # Calculate statistics strictly from the training data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1e-8, std)  # Avoid division by zero
    minn = np.min(X_train, axis=0)
    maxx = np.max(X_train, axis=0)
    
    if norm_type=='standard':
        # 1. Map the normalization transformation across all sets
        normalized_datasets = tuple(map(lambda X: (X - mean) / std, datasets))

        # 2. Define a pure de-normalization function that embeds these exact stats
        def denormalize(X_norm):
            return (X_norm * std) + mean
        def normalize(X):
            return (X-mean)/std
        
    elif norm_type=='minmax':
        # 1. Map the normalization transformation across all sets
        normalized_datasets = tuple(map(lambda X: (X - minn) / (maxx-minn), datasets))

        # 2. Define a pure de-normalization function that embeds these exact stats
        def denormalize(X_norm):
            return (X_norm * (maxx-minn)) + minn
        def normalize(X):
            return (X-minn)/(maxx-minn)
        
    elif norm_type=='none':
        # 1. Map the normalization transformation across all sets
        normalized_datasets = tuple(map(lambda X: (X), datasets))

        # 2. Define a pure de-normalization function that embeds these exact stats
        def denormalize(X_norm):
            return (X_norm)
        def normalize(X):
            return (X)

        
    # Return both the transformed data AND the newly minted function
    return normalized_datasets, denormalize, normalize



def plot_confusion_matrix(X_norm, y, model):
    # Generate predictions for validation or test set (comment out things accordingly)
    y_pred = model.predict(X_norm)
    # y_pred = model.predict(X_test_norm)

    # For binary classification, convert probabilities to class labels
    y_pred_classes = (y_pred > 0.5).astype(int)

    from sklearn.metrics import confusion_matrix

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y, y_pred_classes)
    # conf_matrix = confusion_matrix(y_test, y_pred_classes)
    print(conf_matrix)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="bone", xticklabels=["Jointly drawn", "Independently drawn"], yticklabels=["Jointly drawn", "Independently drawn"])
    plt.imshow(conf_matrix)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
#     plt.show()







# def generate_mock_nre_dataset(num_simulations=1000, num_features=5, seed=42):
#     """
#     Generates a pure NRE dataset where:
#     - Features (instances) contain both: [Observables (0 to N-1), Chi Parameter (Last Column)]
#     - Targets contain binary tags: 1 for Jointly drawn, 0 for Independently drawn.
#     """
#     rng = np.random.default_rng(seed)
    
#     # Simulate the raw physical parameters (chi) matching a grid of 0.2 and 0.25
#     chi_true = np.random.uniform(0.05, 0.45, size=(num_simulations, 1))
    
#     # Simulate observables that depend directly on chi + some physical noise
#     observables = (chi_true * 4.0) + rng.normal(0, 0.05, size=(num_simulations, num_features))
    
#     # --- Create Joint Pairs (Label 1) ---
#     X_joint = np.hstack([observables, chi_true])
#     y_joint = np.ones((num_simulations, 1))
    
#     # --- Create Independent Pairs (Label 0) via 2-Value Swap ---
#     # Since it's exactly 0.2 and 0.25, subtracting from 0.45 cleanly flips them
#     chi_swapped = 0.45 - chi_true
#     X_indep = np.hstack([observables, chi_swapped])
#     y_indep = np.zeros((num_simulations, 1))
    
#     # Combine rows immutably
#     X_all = np.vstack([X_joint, X_indep])
#     y_all = np.vstack([y_joint, y_indep]).flatten()
    
#     # Global shuffle so 1s and 0s are completely mixed
#     perm = rng.permutation(len(X_all))
#     return X_all[perm], y_all[perm]

# def generate_mock_nre_dataset(num_simulations=1000, num_features=5, seed=42):
#     """
#     Generates a pure NRE dataset where:
#     - Features (instances) contain both: [Observables (0 to N-1), Chi Parameter (Last Column)]
#     - Targets contain binary tags: 1 for Jointly drawn, 0 for Independently drawn.
#     """
#     rng = np.random.default_rng(seed)
    
#     # Simulate the raw physical parameters (chi)
#     chi_true = rng.uniform(0.05, 0.45, size=(num_simulations, 1))
    
#     # Simulate observables that depend directly on chi + some physical noise
#     observables = (chi_true * 4.0) + rng.normal(0, 0.05, size=(num_simulations, num_features))
    
#     # --- Create Joint Pairs (Label 1) ---
#     X_joint = np.hstack([observables, chi_true])
#     y_joint = np.ones((num_simulations, 1))
    
#     # --- Create Independent Pairs (Label 0) via random permutation ---
#     # Marginal samples must be independent draws from p(chi), decoupled from
#     # the paired observable. A permutation of chi_true does exactly that:
#     # same marginal distribution, but shuffled so it no longer matches its
#     # original observable.
#     perm = rng.permutation(num_simulations)
#     chi_indep = chi_true[perm]
#     X_indep = np.hstack([observables, chi_indep])
#     y_indep = np.zeros((num_simulations, 1))
    
#     # Combine rows immutably
#     X_all = np.vstack([X_joint, X_indep])
#     y_all = np.vstack([y_joint, y_indep]).flatten()
    
#     # Global shuffle so 1s and 0s are completely mixed
#     global_perm = rng.permutation(len(X_all))
#     return X_all[global_perm], y_all[global_perm]


def generate_mock_raw_data(num_simulations=1000, num_features=5, seed=42):
    rng = np.random.default_rng(seed)
    params_arr = rng.uniform(0.05, 0.45, size=(num_simulations, 1))
    data_arr = (params_arr * 4.0) + rng.normal(0, 0.05, size=(num_simulations, num_features))
    return data_arr, params_arr



import corner
import matplotlib.pyplot as plt
# plt.style.use('default')
def cornerplot1(data, theta_true, param_names, fig = None, color = None, name=None, weights=None):  
    if len(theta_true)==1:
        theta_true_ = None
    else:
        theta_true_=theta_true
        
    fig = corner.corner(
    data,
    bins=40,
    hist_bin_factor=0.5,
    weights = weights, 
    fig = fig, 
    color = color, 
    levels = (0.68, 0.95), 
    plot_contour=True,
    fill_contours=True,
    plot_density=False,
    plot_datapoints=False,
    labels = param_names,
    smooth=1,
    truths = theta_true_,
    truth_color='black',
    linestyle='--',
    truth_kwargs={"linestyle":(0, (5, 5)), "linewidth": 1},
    label_kwargs={"fontsize": 12},
    contour_kwargs={"linestyles": "-", "alpha":0.2},
    hist_kwargs = {"linewidth": 2, "density":True},
    data_kwargs={"ms": 10},
#     contourf_kwargs={"alpha": 0.1}
    )
    if len(theta_true)==1:
        plt.axvline(theta_true, color='black')
    return fig
