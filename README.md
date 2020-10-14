# Automatic Searching and Pruning of Deep Neural Networks for Medical Imaging Diagnostic

**Authors:** Francisco Erivaldo Fernandes Junior and Gary G. Yen

This code can be used to replicate the results from the following paper:

F. E. Fernandes Junior and G. G. Yen, “**Automatic Searching and Pruning of Deep Neural Networks for Medical Imaging Diagnostic**,” IEEE Transactions on Neural Networks and Learning Systems, Oct. 2020.

```
@article{fernandes_automatic_2020,
	title = {Automatic {Searching} and {Pruning} of {Deep} {Neural} {Networks} for {Medical} {Imaging} {Diagnostic}},
	issn = {2162-237X, 2162-2388},
	url = {https://ieeexplore.ieee.org/document/9222548/},
	doi = {10.1109/TNNLS.2020.3027308},
	urldate = {2020-10-14},
	journal = {IEEE Transactions on Neural Networks and Learning Systems},
	author = {Fernandes, Francisco Erivaldo and Yen, Gary G.},
	year = {2020},
	pages = {1--11}
}
```

## Dependencies
To run this code, you will need the following packages installed on you machine:

- Python 3.7;
- PyTorch 1.6.0;
- Numpy 1.16.4;
- Matplotplib 3.1.0;
- Tabulate 0.8.7;
- Pandas 1.1.1;
- Scikit-learn 0.23.2;
- THOP: PyTorch-OpCounter (modified):
    - Install THOP from the [https://github.com/feferna/pytorch-OpCounter](https://github.com/feferna/pytorch-OpCounter) repository.
    
**Note1:** If your system has all these packages installed, the code presented here should be able to run on Windows, macOS, or Linux.

**Note2:** This code is hardcoded to run on a Nvidia GPU. To run it on a CPU, you will need to eliminate all .cuda() calls throughout the code.
    
## Modified code used in this repository created by others

- The modified THOP repository ([https://github.com/feferna/pytorch-OpCounter](https://github.com/feferna/pytorch-OpCounter)) is a slight modification of the repository created by Ligeng Zhu, which can be found on [https://github.com/Lyken17/pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter).
- The code found on **Utils/BuildModelResidual.py** is a slight modification of the code developed by Yerlan Idelbayev. The original version can be found in the following link: [https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py](https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py). The original copyright disclaimer is reproduced in the file **Utils/BuildModelResidual.py** in accordance with the original author's license.
- Any problem you may face using the above mentioned codes should be directed to their respective authors.

## Usage

1. First, clone this repository:

	```
	git clone https://github.com/feferna/DNNDeepeningPruning.git
	```

2. Download the following datasets and extract them to their corresponding folders inside the ```datasets``` folder:
	1. ISIC 2016: [https://challenge.kitware.com/#phase/5667455bcad3a56fac786791](https://challenge.kitware.com/#phase/5667455bcad3a56fac786791)
	2. Chest X-Ray: [https://data.mendeley.com/datasets/rscbjbr9sj/2](https://data.mendeley.com/datasets/rscbjbr9sj/2)
	3. CIFAR10: [https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
	4. CIFAR100: [https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

3. Now, you can test the algorithm by running the ```main.py``` file:

	```
	python main.py
	```
	
	or
	
	```
	python3 main.py
	```

**Note2:** The algorithm's parameters can modified in the file ```main.py```.

**Note3:** due to our limited resources, we cannot provide any support to the code in this repository.

