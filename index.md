---
layout: default
---

# RELICS

The [RELICS Batched Dataset](https://raw.githubusercontent.com/as595/NuzaRelics/master/nuza-batches-py.tar.gz) Contains NVSS images of the radio relics from the combined catalog of:

*Can cluster merger shocks reproduce the luminosity and shape distribution of radio relics?*
Nuza, Gelszinnis, Hoeft & Yepes, 2017, Monthly Notices of the Royal Astronomical Society, vol. 470, pp. 240-263 [arXiv:1704.06661](https://arxiv.org/abs/1704.06661)

## The RELICS Batched Dataset

The [RELICS Batched Dataset](https://raw.githubusercontent.com/as595/NuzaRelics/master/nuza-batches-py.tar.gz) contains only 59 objects, all of which are contained in the **test batch**. These are in random order, but each dataset is accompanied by an identifiable filename containing the coordinates of the source. Each image is 300 x 300 pixels in size, with a standard NVSS pixel size of 15 arcseconds.

## Using the Dataset in PyTorch

The [nuza.py](https://raw.githubusercontent.com/as595/NuzaRelics/master/nuza.py) file contains an instance of the [torchvision Dataset()](https://pytorch.org/docs/stable/torchvision/datasets.html) for the RELICS Batched Dataset.

To use it with PyTorch in Python, first import the torchvision datasets and transforms libraries:

```python
from torchvision import datasets
import torchvision.transforms as transforms
```

Then import the HTRU1 class:

```python
from nuza import RELICS
```

Define the transform:

```python
# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
                                transforms.CenterCrop(32),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5],[0.5]),
])
 ```

Read the HTRU1 dataset:

```python
# choose the training and test datasets
testset = RELICS(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True, num_workers=2)
```
 
### Jupyter Notebooks

An example of notebook iterating the RELICS Dataloader() is provided as a [Jupyter notebook](https://github.com/as595/NuzaRelics/blob/master/NuzaRelics_example.ipynb).

[![HitCount](http://hits.dwyl.io/as595/RELICS.svg)](http://hits.dwyl.io/as595/RELICS)

