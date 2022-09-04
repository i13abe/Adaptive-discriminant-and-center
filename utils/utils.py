import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


def torch_to_numpy_image(torch_data):
    """
    Translate the torch tensor to numpy image.
    Args:
        torch_data (torch.tensor): torch data with (N, C, H, W)
    Rtuens:
        images (list of numpy array): list of ndarray with (N, H, W, C)
    """
    images = []
    for image in torch_data:
        image = image.detach().cpu().numpy()
        if image.shape[0]==1:
            image = np.tile(image, (3,1,1))
        image = np.transpose(image,(1,2,0)) # transopose channel
        images.append(image)
    return images


def dim_reduction_PCA(fit_data, test_data=None, dim=2, pca=None):
    """
    Dimensionality Reduction by PCA
    Args:
        fit_data (2d ndarray): numpy array (N, D)
        test_data (2d ndarray): numpy array (n, D). Default None
        dim (int): dimension when be reducted. Default 2
        pca (PCA): fitted PCA. Default is None
    Returns:
        fit_results (2d ndarray): dimensionality reducted numpy array (N, dim)
        test_results (2d ndarray): dimensionality reducted numpy array (N, dim).
                                    If test_data was None, this is also None
        pca (PCA): fitted pca
    """
    if pca is None:
        pca = PCA(n_components=dim)
        fit_results = pca.fit_transform(fit_data)
    else:
        fit_results = pca.transform(fit_data)
    if test_data is not None:
        test_results = pca.transform(test_data)
    else:
        test_results = None
        
    return fit_results, test_results, pca
    
    
def dict_to_PCA(fit_dict, test_dict=None, dim=2, pca=None):
    """
    Dimensionality Reduction by PCA with dict data
    example of dict data is below:
        fit_dict = {"aa": [[ndarray with shape (Na, D)]],
                    "bb": [[ndarray with shape (Nb, D)]],...}
                    
    Args:
        fit_dict (dict of 2d ndarray): dict of 2-d numpy array
        test_dict (dict of 2d ndarray): dict of 2-d numpy array. Default None
        dim (int): dimension when be reducted. Default 2
        pca (PCA): fitted PCA. Default is None
    Returns:
        fit_results_dict (dict of 2d ndarray): dict of dimensionality reducted numpy array (N, dim)
        test_results_dict (dict of 2d ndarray): dict of dimensionality reducted numpy array (N, dim).
                                    If test_data was None, this is also None
        pca (PCA): fitted pca
    """
    dict_name = []
    fit_pages = [0]
    items = []
    for key, item in fit_dict.items():
        dict_name.append(key)
        fit_pages.append(fit_pages[-1]+len(item))
        items.append(item)
    fit_data = np.vstack(items)
    
    if test_dict is not None:
        test_pages = [0]
        items = []
        for key, item in test_dict.items():
            test_pages.append(test_pages[-1]+len(item))
            items.append(item)
        test_data = np.vstack(items)
    else:
        test_data = None
    
    fit_results, test_results, pca = dim_reduction_PCA(fit_data, test_data, dim, pca)
    
    fit_results_dict = {}
    for i, key in tqdm(enumerate(dict_name)):
        fit_results_dict[key] = fit_results[fit_pages[i]:fit_pages[i+1]]
    
    if test_results is not None:
        test_results_dict = {}
        for i, key in tqdm(enumerate(dict_name)):
            test_results_dict[key] = test_results[test_pages[i]:test_pages[i+1]]
    else:
        test_results_dict = None
    
    return fit_results_dict, test_results_dict, pca


def dict_to_oneVSothers(data_dict, num_class=0):
    """
    Get the one class neurons of num_class and other class neurons from data_dict.
    example is below where "num_class=0":
        data_dict = {"class1":[[a,b,b,b],[a,b,b,b],...],
                     "class2":[[c,d,d,d],[c,d,d,d],...]
                     "class3":[[e,f,f,f],[e,f,f,f],...],...}
        
        => results = {"num_class":[a,a,a,a,...],
                      "others":[c,c,c,...e,e,e,...]}
    Args:
        data_dict (dict of 2d ndarray): dict of 2-d numpy array
        num_class (int): number of class to get
    Returns:
        results (dict of 2d ndarray): dict of 2-d numpy array
    """
    results = {}
    one_data = []
    other_data = []
    for i, items in enumerate(zip(data_dict.items())):
        key, item = items[0]
        if i==num_class:
            one_data.append(item[:, num_class])
        else:
            other_data.append(item[:, num_class])
            
    results[str(num_class)] = np.hstack(one_data)
    results["others"] = np.hstack(other_data)
    return results


def numpy_to_dict(data_numpy, labels, based_labels=None):
    """
    Get the label dict from numpy.
    example is below
        data_numpy = [[a,a,a,...],
                      [b,b,b,...],
                      [c,c,c,...],
                      [d,d,d,...],...]
        labels = [0,1,0,0,2,1,...]
        based_labels = ["0", "1", "2"]
        
        => results = {"0":[[a,a,a,...],[c,c,c,...],[d,d,d,...],...],
                      "1":[[b,b,b,...],...],...}
    Args:
        data_numpy (2d ndarray): 2-d numpy array
        labels (1d numpy): 1-d numpy
        based_labels (list): name of labels. default is None. If None, the based_labels becomes class number.
    Returns:
        results (dict of 2d ndarray): dict of 2-d numpy array
    """
    if based_labels is None:
        based_labels = [str(x) for x in np.unique(labels)]
    
    results = dict(zip(based_labels, [[] for i in range(len(based_labels))]))
    
    for data, label in zip(data_numpy, labels):
        results[based_labels[label]].append(data)
    
    for key in based_labels:
        results[key] = np.array(results[key])
    return results


def _binary_search_perplexity(sqdistances, desired_perplexity, verbose):
    """Estimate sigma of conditional probability p_j|i by using desired perplexity.
    However, This function is not used in QSNE.
    Instead of using this, we use _utils._binary_search_perplexity on cython, because it is faster computational time.
    If you can not use _utils, please change to this function in function _joint_probabilities().
    Parameters
    ----------
    sqdistances : array, shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed conditional probability matrix.
    """
    n_steps = 100
    
    n_samples = sqdistances.shape[0]
    n_neighbors = sqdistances.shape[1]
    using_neighbors = n_neighbors < n_samples
    
    beta_sum = 0.0
    desired_entropy = np.log(desired_perplexity)
    
    P = np.zeros((n_samples, n_neighbors))
        
    perplexity_tolerance=1e-5
    inf = 1e+10
    
    for i in range(n_samples):
        beta = 1.0
        beta_min = -inf
        beta_max = inf
        
        for l in range(n_steps):
            sum_Pi = 0.0
            for j in range(n_neighbors):
                if j != i or using_neighbors:
                    P[i, j] = np.exp(-sqdistances[i, j] * beta)
                    sum_Pi += P[i, j]
                    
            if sum_Pi == 0.0:
                sum_Pi = 1e-8
            sum_disti_Pi = 0.0
            
            for j in range(n_neighbors):
                P[i, j] /= sum_Pi
                sum_disti_Pi += sqdistances[i, j] * P[i, j]
            
            entropy = np.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy
            
            if np.abs(entropy_diff) <= perplexity_tolerance:
                break
            
            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -inf:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0
        
        beta_sum += beta
        
        if verbose and ((i + 1) % 1000 == 0 or i + 1 == n_samples):
            print("[q-SNE] Computed conditional probabilities for sample "
                  "%d / %d" % (i + 1, n_samples))
    
    if verbose:
        print("[q-SNE] Mean sigma: %f"
              % np.mean(np.sqrt(n_samples / beta_sum)))

    return P


def k_nearest_neighbors(fit_data, labels, k, kNN=None):
    """
    Get the scores by kNN.
    Args:
        fit_data (ndarray): ndarray data to fit.
        labels (1d ndarray): 1-d ndarray.
        k (int): The numer of nearest neighbors.
        kNN (KNeighborsClassifier or None): Fitted kNN. Default is None
    Returns:
        score (float): Results score of kNN fitted fit_data and labels
        kNN (KNeighborsClassifier): Fitted kNN object.
    """
    if kNN is None:
        kNN = KNeighborsClassifier(n_neighbors=k)
        kNN.fit(fit_data, labels)
        
    score = kNN.score(fit_data, labels)
    return score, kNN


def dict_to_kNN(fit_dict, k, kNN=None):
    """
    k nearest neighbors by kNN with dict data
    example of dict data is below:
        fit_dict = {"aa": [[ndarray with shape (Na, D)]],
                    "bb": [[ndarray with shape (Nb, D)]],...}
                    
    Args:
        fit_dict (dict of 2d ndarray): dict of 2-d numpy array
        k (int): The numer of nearest neighbors.
        kNN (KNeighborsClassifier or None): Fitted kNN. Default is None
    Returns:
        score (float): Results score of kNN fitted fit_data and labels
        kNN (KNeighborsClassifier): Fitted kNN object.
    """
    labels = []
    items = []
    for i, item in enumerate(fit_dict.values()):
        labels.append([i]*len(item))
        items.append(item)
    labels = np.hstack(labels)
    fit_data = np.vstack(items)
    
    score, kNN =  k_nearest_neighbors(fit_data, labels, k, kNN)
    return score, kNN