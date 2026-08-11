import numpy as np


def prepare_for_NRE(data_arr, params_arr, shuffle_seed=42, indep_seed=22, final_seed=11):
    '''
    Prepares data and parameter arrays for NRE classification.
    Inputs:
        - data_arr: np.array of shape: (number of sims, number of observables)
        - params_arr: np.array of shape: (number of sims, number of parameters)
        - shuffle_seed: seed for the initial shuffle of the raw data
        - indep_seed: seed for shuffling params within the independent (marginal) half
        - final_seed: seed for the final shuffle of joint+independent samples together
    Outputs:
        - instances: the training/validation/testing data of the NRE (it is recommended to keep a different test set for coverage tests)
        - targets: whether the pair is jointly or independently drawn

    Specifically it performs the following steps:
        1. Reserves half of the data for the independently drawn samples
        2. Generates a permutation of indices and shuffles the independent part
        3. Introduce flags working as target features for the classifier and associate them with the independent and joint samples
        4. Shuffle all together in the end

    Note: unlike the original implementation, this uses local np.random.Generator
    instances (via the seed arguments) rather than np.random.seed(), so calling
    this function does not mutate global numpy random state / affect unrelated
    randomness elsewhere in your pipeline. Pass different seeds (or None for
    non-reproducible randomness) to get independent splits.
    '''

    assert data_arr.shape[0] == params_arr.shape[0]

    # 1. Reserve half of the data for the independently drawn samples-----------------------------
    number_of_sims = data_arr.shape[0]
    print('number_of_simulations:', int(data_arr.shape[0]))
    half_of_sims = data_arr.shape[0] / 2
    print('Half that is reserved for independetly drawn samples:', data_arr.shape[0] / 2)

    # If the number of sims is not an even number just delete the last one for now
    if number_of_sims % 2 != 0:
        data_arr = data_arr[:number_of_sims - 1, :]
        params_arr = params_arr[:number_of_sims - 1, :]
        half_of_sims = data_arr.shape[0] / 2

    print('Same but integer:', half_of_sims)

    # Shuffle the input data to be sure
    rng_shuffle = np.random.default_rng(shuffle_seed)
    permutation = rng_shuffle.permutation(params_arr.shape[0])
    data_arr_shuffled = data_arr[permutation, :]
    params_arr_shuffled = params_arr[permutation, :]

    # 1. Reserve half of the data for the independently drawn samples-----------------------------
    split_at = int(half_of_sims)  # +1 to be half i just have 1 less due to indexing

    # 2. Generate a permutation of indices and shuffles the independent part while keeping the joint part untouched-----------------------------

    # Take the joint samples and leave them as is
    data_arr_shuffled_joint = data_arr_shuffled[:split_at]
    params_arr_shuffled_joint = params_arr_shuffled[:split_at]

    # Take the independent samples and shuffle them
    rng_indep = np.random.default_rng(indep_seed)
    permutation = rng_indep.permutation(params_arr_shuffled[split_at:number_of_sims].shape[0])
    data_arr_shuffled_indep = data_arr_shuffled[split_at:number_of_sims]  # DO NOT SHUFFLE THIS ONLY THE PARAMS
    params_arr_shuffled_indep = params_arr_shuffled[split_at:number_of_sims][permutation, :]

    # 3. Introduce flags working as target features for the classifier and associate them with the independent and joint samples
    flag_0 = np.zeros((split_at, 1))
    flag_1 = np.ones((split_at, 1))

    # Connect the 3 numpy arrays (data, params and flags) and shuffle them again in order to separate them afterwards (into x and y)
    joint_NRE_ready = np.concatenate((data_arr_shuffled_joint, params_arr_shuffled_joint), axis=1)
    joint_NRE_ready = np.concatenate((joint_NRE_ready, flag_1), axis=1)

    indep_NRE_ready = np.concatenate((data_arr_shuffled_indep, params_arr_shuffled_indep), axis=1)
    indep_NRE_ready = np.concatenate((indep_NRE_ready, flag_0), axis=1)

    all_unshuffled_NRE_ready = np.concatenate((joint_NRE_ready, indep_NRE_ready), axis=0)

    # 4. Shuffle all together in the end
    rng_final = np.random.default_rng(final_seed)
    permutation = rng_final.permutation(all_unshuffled_NRE_ready.shape[0])
    all_shuffled_NRE_ready = all_unshuffled_NRE_ready[permutation, :]

    # Make 2 numpys training incances and targets
    instances = all_shuffled_NRE_ready[:, :-1]
    targets = all_shuffled_NRE_ready[:, -1]

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
        tuple: (normalized_datasets, denormalize_fn, normalize_fn)
    """
    X_train = datasets[0]

    # Calculate statistics strictly from the training data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1e-8, std)  # Avoid division by zero
    minn = np.min(X_train, axis=0)
    maxx = np.max(X_train, axis=0)

    if norm_type == 'standard':
        # 1. Map the normalization transformation across all sets
        normalized_datasets = tuple(map(lambda X: (X - mean) / std, datasets))

        # 2. Define a pure de-normalization function that embeds these exact stats
        def denormalize(X_norm):
            return (X_norm * std) + mean

        def normalize(X):
            return (X - mean) / std

    elif norm_type == 'minmax':
        # 1. Map the normalization transformation across all sets
        normalized_datasets = tuple(map(lambda X: (X - minn) / (maxx - minn), datasets))

        # 2. Define a pure de-normalization function that embeds these exact stats
        def denormalize(X_norm):
            return (X_norm * (maxx - minn)) + minn

        def normalize(X):
            return (X - minn) / (maxx - minn)

    elif norm_type == 'none':
        # 1. Map the normalization transformation across all sets
        normalized_datasets = tuple(map(lambda X: (X), datasets))

        # 2. Define a pure de-normalization function that embeds these exact stats
        def denormalize(X_norm):
            return (X_norm)

        def normalize(X):
            return (X)

    # Return both the transformed data AND the newly minted function
    return normalized_datasets, denormalize, normalize
