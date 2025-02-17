import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA, FastICA

from exp_env.data.data_maker_base import DataMaker, DataMakerSpec

class ReduceWeightDimConfig:
    def __init__(self, mode: str, dim: int):
        self.mode = mode
        self.dim = dim
    @staticmethod
    def pca(dim: int):
        return ReduceWeightDimConfig('pca', dim)
    @staticmethod
    def tsne(dim: int):
        return ReduceWeightDimConfig('tsne', dim)
    @staticmethod
    def umap(dim: int):
        return ReduceWeightDimConfig('umap', dim)
    @staticmethod
    def ica(dim: int):
        return ReduceWeightDimConfig('ica', dim)
    @staticmethod
    def none():
        return ReduceWeightDimConfig('none', 0)



import numpy as np
from numpy.linalg import eig
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns


def whiten(X, use_np=True):
    """Utility function to whiten data without zero-mean centering. Whitening means removing correlation between
    features and making the individual features have unit variance.

    :param
    X : np.array
        Data matrix. This is assumed to be in the (uncommon) format n_features x n_samples
    use_np : bool, optional (default: `True`)
        Whether to use numpy to compute the covariance matrix

    :returns
    Z : np.array
        Data matrix, whitened to remove covariance between the features
    """

    if use_np:
        C_X = np.cov(X, rowvar=True)
    else:
        C_X = (X - X.mean(1)) @ (X - X.mean(1)).T
    D, E = eig(C_X)
    V = E @ np.diag(1 / np.sqrt(D)) @ E.T
    Z = V @ X

    return Z


def rotation(phi):
    """Create 2D rotation matrix

    :param
    phi : float
        The angle by which we want to rotate.

    :returns
    A : np.array
        A 2D rotation matrix
    """

    return np.array([[np.cos(phi), np.sin(phi)],
                     [-np.sin(phi), np.cos(phi)]])


def loss(Y):
    """Compute the loss for a given reconstruction Y.
    This will simply be the sum of squared elements in the restriction
    of Y to it's negative elements

    :param
    Y : np.array
        Data matrix, reconstrcution of the sources

    :returns
    l : np.float
        The loss
    """
    # restrict Y to it's negative elements
    n_samples = Y.shape[1]
    Y_neg = np.where(Y < 0, Y, 0)

    return 1 / (2 * n_samples) * norm(Y_neg, ord='fro') ** 2


def obj_fun(phi, Z):
    """Objective to be used for finding the optimum rotation angle

    :param
    phi : float
        Rotation angle for which we wish to compute the los
    Z : np.matrix
        Whitened data matrix. Must have two rows, each corresponding to one feature.

    :returns
    l : float
        loss corresponding to a rotation of Z in 2D around phi
    """

    # check input
    if Z.shape[0] != 2:
        raise ValueError('Z has more than two features.')

    # rotate the data
    W = rotation(phi)
    Y = W @ Z

    return loss(Y)


def givens(n, i, j, phi):
    """Compute n-dimensional givens rotation

    :param
    n : int
        Dimension of the rotation matrix to be computed
    i, j : int
        Dimensions i and j define the surface we wish to rotate in
    phi : float
        Rotation angle

    :returns
    R : np.array
        Given's rotation

    """
    R = np.eye(n)
    R[i, i], R[j, j] = np.cos(phi), np.cos(phi)
    R[i, j], R[j, i] = np.sin(phi), -np.sin(phi)

    return R


def torque(Y):
    """Compute torque values of Y.

    These correspond to the gradient if different directions, where
    each direction is a possible rotation in a surface defines by two axis. The resulting matrix of
    torque values will have zeroes on the diagonal and will by symmetric.

    :param
    Y : np.matrix
        Reconstruction of the sourdes

    :returns
    t_max : float
        Maximum torque value found
    ixs : tuple
        i-j coordinates corresponding to the max. This defines a hyperplane in n-dimensional space.
    G : np.array
        Matrix of torque values. Symmetric and zero on the diagonal. Only the upper half is computed.
    """

    # compute the rectified parts of Y
    Y_pos = np.where(Y > 0, Y, 0)
    Y_neg = np.where(Y < 0, Y, 0)

    # compute torque values
    n = Y.shape[0]
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            G[i, j] = np.dot(Y_pos[i, :], Y_neg[j, :]) - np.dot(Y_neg[i, :], Y_pos[j, :])

    # find max and corresponding indices
    t_max = np.amax(np.abs(G))
    result = np.where(np.abs(G) == t_max)
    ixs = [result[i][0] for i in range(len(result))]

    return t_max, ixs, G


def run_nn_ica(X, t_tol=1e-1, t_neg=None, verbose=1, i_max=1e3, print_all=100,
               whiten_mat=True, keep='last', return_all=False):
    """Algorithm to run non-negative indipendent component analysis.

    Given some data X, find matrices A and S such that
    X = A S,
    where S is non-negative and has indipendent rows. We pose no
    constraint on the matrix A.

    This algorithm is implemented as described in Plumbley, 2003, and
    relies upon whitening and rotating the data. This is guaranteed to
    converge only if the sources are 'well grounded', i.e. have probability
    down to zero. Note that this is the implementation for a square mixing
    matrix A.

    Parameters
    --------
    X : np.array,
        The data matrix of shape (n_features, n_samples)
    t_tol : float, optional (default: `1e-1`)
        Stopping tolerance. If the maximum torque falls below this
        value, stop.
    t_neg: float, optional (default: `None`)
        Stopping number of negative elements. If #negative elements crosses
        this threshold, stop.
    verbose : int, optional (default: `1`)
        How much output to give
    i_max : int, optional (default: `1e3`)
        Maximum number of iterations
    print_all : int, optional (default: `100`)
        Print every print_all iterations
    whiten : bool, optional (default: `True`)
        whether to whiten the input matrix
    keep: Str, optional (default: `'last'`)
        which reconstruction to keep, possible options are:
        `'last'`, `'best_neg'`, `'best_tol'`
        and correspond to last, smallest #negative elements and smallest tolerance
        respectively
    return_all: bool, optional (default: `False`)
        whether to return all of the progress of Y, W
        returns Y_best, W_best, Z, t_max_arr, ys, ws

    Returns
    --------
    Y : np.array
        The reconstructed sources, up to scaling and permutation
    W : np.array
        The final rotation matrix
    Z : np.array
        The whitened data
    t_max_arr : np.array
        The maximum torque values for each iteration
    """

    assert keep in ('last', 'best_neg', 'best_tol'), f'Unknown selection criterion `{keep}`.'


    def set_best(Y, W, tol):
        nonlocal Y_best, W_best, tol_best

        if return_all:
            ys.append(Y)
            ws.append(W)

        if keep == 'last':
            is_better = True
        if keep == 'best_neg':
            is_better = Y_best is None or np.sum(Y_best < 0) > np.sum(Y < 0)
        else:
            is_better = tol_best > tol

        if is_better:
            Y_best, W_best = Y, W
            tol_best = tol  # need to keep memory of this

    # lists that records the progress of Y and W
    ys = []
    ws = []

    # best values and tolerance so far
    Y_best, W_best = None, None
    tol_best = np.inf

    t_neg = -np.inf if t_neg is None else t_neg

    # initialise
    n = X.shape[0]
    W = np.eye(n)
    t_max_arr = []

    # whiten the data
    Z = whiten(X) if whiten_mat else X
    Y = W @ Z

    i = 0
    while True:
        # compute the max torque of Y and corresponding indices
        t_max, ixs, _ = torque(Y)
        t_max_arr.append(t_max)

        set_best(Y, W, t_max)

        if t_max < t_tol or np.sum(Y_best < 0) < t_neg: # converged
            print('=' * 10)
            print('i = {}, t_max = {:.2f}, ixs = {}, #negative = {}'.format(i, t_max, ixs, np.sum(Y_best < 0)))
            print('Converged. Returning the reconstruction.')
            if return_all:
                return Y_best, W_best, Z, t_max_arr, ys, ws

            return Y_best, W_best, Z, t_max_arr

        if i > i_max  and t_max > t_max_arr[-2]:  # failed to converge
            print('=' * 10)
            print('i = {}, t_max = {:.2f}, ixs = {}, #negative = {}'.format(i, t_max, ixs, np.sum(Y_best < 0)))
            print(f'Error: Failed to converge. Returning current matrices.')
            if return_all:
                return Y_best, W_best, Z, t_max_arr, ys, ws

            return Y_best, W_best, Z, t_max_arr

        # print some information
        if (verbose > 0) and (i % print_all == 0):
            print('i = {}, t_max = {:.2f}, ixs = {}, #negative = {}'.format(i, t_max, ixs, np.sum(Y < 0)))

        # reduce to axis pair, find rotation angle and construct givens matrix
        Y_red = Y[ixs, :]
        opt_res = minimize_scalar(fun=obj_fun, bounds=(0, 2 * np.pi), method='bounded', args=Y_red)
        R = givens(n, ixs[0], ixs[1], opt_res['x'])

        # update the rotation matrix W and the reconstruction matrix Y
        W = R @ W
        Y = R @ Y

        i += 1

def plot_ica_reconstruction(data, labels=None, *args, **kwargs):
    """Plotting function for ICA reconstruction

    Parameters
    --------
    data: List[np.ndarray]
        Matrices to be plotted.
    titles: Union[List, Str, NoneType], optional (default: `None`)
        Corresponding titles, if present
    *args, **kwargs:
        additional arguments for `sns.distplot`

    Returns
    --------
    None
    """

    plt.close('all')

    n_sources, n_mat = data[0].shape[0], len(data)
    for i, d in enumerate(data):
        assert d.shape[0] == n_sources, f'Wrong number of features for `data[{i}].shape[0]` == `{d.shape[0]}` != `{n_sources}`.'

    if labels is not None and len(labels) != n_mat:
        print(f'Inconsistent number of titles given `{len(labels)}` != `{n_mat}`, showing no labels.')
        labels = None

    fig, axes = plt.subplots(nrows=n_sources, ncols=n_mat, figsize=(4 * n_mat, 4 * n_sources))
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    elif axes.ndim != 2:
        # treat only 1 source specifically
        axes = np.expand_dims(axes, axis=(n_sources) != 1)

    for i, row in enumerate(axes):
        for j, (A, ax) in enumerate(zip(data, row)):
            sns.distplot(A[i, :], *args, ax=ax, kde=False,
                         axlabel='{}_{}'.format(labels[j], i) if labels is not None else None,
                         **kwargs)

    fig.suptitle("Non-negative ICA")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.show()

class EmbeddingDataConfig(DataMakerSpec):
    model_name: str
    is_bardirectional: bool
    def __init__(self, model_name: str, is_bardirectional: bool,pad_token:str|None|bool=None,time_shift=0,alpha=0.3,device:str='cpu',reduce_weight_dim: ReduceWeightDimConfig=ReduceWeightDimConfig.none(),quantization_config=None):
        self.model_name = model_name
        self.is_bardirectional = is_bardirectional
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if pad_token is True:
            self.tokenizer.pad_token=self.tokenizer.eos_token
        elif type(pad_token) is str:
            self.tokenizer.pad_token=pad_token
        model= AutoModel.from_pretrained(model_name,
            quantization_config=quantization_config
            )
        self.device=device
        self.emmbedding_weight = model.get_input_embeddings().weight.cpu().detach().numpy()
        if reduce_weight_dim.mode=='pca':
            if reduce_weight_dim.dim<0:
                #auto select dim
                reduce_weight_dim.dim=self.emmbedding_weight.shape[1]
                pca=PCA(n_components=reduce_weight_dim.dim)
                #寄与度が0.9以上になる次元数を選択
                pca.fit(self.emmbedding_weight)
                for i in range(reduce_weight_dim.dim):
                    if np.sum(pca.explained_variance_ratio_[:i])>0.9:
                        reduce_weight_dim.dim=i
                        break
                del pca
            pca=PCA(n_components=reduce_weight_dim.dim)
            self.emmbedding_weight=pca.fit_transform(self.emmbedding_weight)
            del pca
        elif reduce_weight_dim.mode=='ica':
            ica=FastICA(n_components=reduce_weight_dim.dim, max_iter=10000, tol=1e-4)
            self.emmbedding_weight=ica.fit_transform(self.emmbedding_weight)
            from scipy.stats import skew
            skewness = skew(self.emmbedding_weight)
            self.emmbedding_weight*=np.sign(skewness)
            del ica
        self.emmbedding_weight=torch.tensor(self.emmbedding_weight).to(device)
        self.emmbedding_weight.requires_grad=False
        self.emmbedding_out_dim=reduce_weight_dim.dim
        self.time_shift=time_shift
        self.alpha=alpha

class EmbeddingDataMaker(DataMaker):
    def __init__(self, config: EmbeddingDataConfig):
        self.tokenizer = config.tokenizer
        self.embedding_weight = config.emmbedding_weight
        self.cut_off=1/8
    def make(self, texts:list[list[int]]|list[str],is_reverse:bool=True,max_length:int=256)->torch.Tensor:
        padding='right' if is_reverse else 'left'
        self.tokenizer.padding_side=padding
        #get embedding
        if type(texts[0][0]) is int:
            tokenized={}
            tokenized['input_ids']=[text+[self.tokenizer.pad_token_id]*(max_length-len(text))for text in texts]
            tokenized['attention_mask']=[[1]*len(text)+[0]*(max_length-len(text))for text in texts]
        else:
            tokenized=self.tokenizer(texts, truncation=True, padding='max_length',max_length=max_length)
        ids=tokenized['input_ids']
        ids=torch.tensor(ids).to(self.embedding_weight.device)
        embedding= torch.nn.functional.embedding(ids,self.embedding_weight)
        #masking
        embedding=embedding*torch.tensor(tokenized['attention_mask']).unsqueeze(2).to(embedding.device)
        
        return embedding,torch.tensor(tokenized['attention_mask']).to(embedding.device)

class EmbeddingsDataMaker(EmbeddingDataMaker):
    def __init__(self, config: EmbeddingDataConfig):
        self.is_berdirectional = config.is_bardirectional
        self.time_shift=config.time_shift
        self.alpha=config.alpha
        super().__init__(config)
    def make(self, texts:list[str]|list[list[int]],use_mask=False,is_reverse=False)->torch.Tensor:
        max_length=-1
        time_shift=self.time_shift
        alpha=self.alpha
        for text in texts:
            if type(text) is str:
                token=self.tokenizer(text, truncation=False, padding=False)['input_ids']
            else:
                token=text
            if len(token)>max_length:
                max_length=len(token)
            del token
        forward,f_mask=super().make(texts,is_reverse=is_reverse,max_length=max_length)
        if self.is_berdirectional:
            reversed,_=super().make(texts,is_reverse=not is_reverse,max_length=max_length)
            reversed=reversed.flip(1)

        if self.is_berdirectional:
            forward = torch.cat([forward,reversed],dim=2)
        if time_shift>0:
            with torch.no_grad():
                forward=alpha*forward+(1-alpha)*torch.roll(forward,time_shift,dims=1)
        if use_mask:
            return forward,f_mask
        return forward