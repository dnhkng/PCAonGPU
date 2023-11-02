# PCAonGPU

A powerful implementation of Incremental Principal Components Analysis that runs on GPU, built on top of PyTorch. Up to 20x speed-ups on gigabyte-scale PCA.

## Installation

### From PyPi
`pip install PCAonGPU`


### From Source
1. Clone the repository:

`git clone https://github.com/dnhkng/PCAonGPU.git`

2. Navigate to the cloned directory:

`cd PCAonGPU`

3. Install the required dependencies:

`pip install -r requirements.txt`

## Usage

```python
from gpu_pca import IncrementalPCAonGPU

# Create an instance
model = IncrementalPCAonGPU(n_components=5)

# Fit the model (either using `fit` or `partial_fit`)
model.fit(your_data)

# Transform the data
transformed_data = model.transform(your_data)
```

## Benchmark

SKlearn on an AMD Ryzen 9 5900X 12-Core Processor
 vs PCAonGPU on an Nvidia 4090

Data size: 5000 samples of 5000 dimensional data:
```
> python tests/benchmark_gpu_pca.py 
test_sklearn_pca took 21.78324556350708 seconds to complete its execution.
test_gpu_pca took 6.523377895355225 seconds to complete its execution.
```

Data size: 50000 samples of 10000 dimensional data.
```
> python tests/benchmark_gpu_pca.py 
test_sklearn_pca took 314.3792634010315 seconds to complete its execution.
test_gpu_pca took 35.23140811920166 seconds to complete its execution.
```
