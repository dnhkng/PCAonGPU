import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA as SklearnIPCA
from gpu_pca.pca_module import IncrementalPCAonGPU

def generate_data(n_samples=50000, n_features=100, random_state=None):
    """Generate random Gaussian data."""
    rng = np.random.default_rng(random_state)
    return rng.standard_normal(size=(n_samples, n_features))

data1 = generate_data()
data2 = generate_data()

data1gpu = torch.tensor(data1, device='cuda')
data2gpu = torch.tensor(data2, device='cuda')

def test_fit_method():
    sklearn_model = SklearnIPCA(n_components=5)
    our_model = IncrementalPCAonGPU(n_components=5)

    sklearn_model.fit(data1)
    our_model.fit(data1gpu)

    transformed_sklearn = sklearn_model.transform(data1)
    transformed_our_model = our_model.transform(data1gpu).cpu().numpy()

    print(transformed_sklearn)
    print(transformed_our_model)
    
    # assert torch.allclose(torch.tensor(transformed_sklearn), torch.tensor(transformed_our_model), atol=1e-3)

def test_partial_fit_method():
    sklearn_model = SklearnIPCA(n_components=5)
    our_model = IncrementalPCAonGPU(n_components=5)

    sklearn_model.partial_fit(data1)
    sklearn_model.partial_fit(data2)

    our_model.partial_fit(data1gpu)
    our_model.partial_fit(data2gpu)

    transformed_sklearn = sklearn_model.transform(data1)
    transformed_our_model = our_model.transform(data1gpu).cpu().numpy()

    assert torch.allclose(torch.tensor(transformed_sklearn), torch.tensor(transformed_our_model), atol=5e-2)

if __name__ == "__main__":
    test_fit_method()
    test_partial_fit_method()