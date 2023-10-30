import time
import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA as SklearnIPCA
from gpu_pca import IncrementalPCAonGPU

rand_data = torch.rand(50000, 10000)

def timer_func(func):
    def function_timer(*args, **kwargs):
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "{func} took {time} seconds to complete its execution."
        print(msg.format(func = func.__name__,time = runtime))
        return value
    return function_timer



@timer_func
def test_sklearn_pca():
    sklearn_model = SklearnIPCA(n_components=10)
    sklearn_model.fit(rand_data)
    transformed_sklearn = sklearn_model.transform(rand_data)


@timer_func
def test_gpu_pca():
    gpu_model = IncrementalPCAonGPU(n_components=10)
    gpu_model.fit(rand_data)
    gpu_transformed_sklearn = gpu_model.transform(rand_data)


if __name__ == '__main__':
    test_sklearn_pca()
    test_gpu_pca()