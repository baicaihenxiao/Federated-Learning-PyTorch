# Federated-Learning (PyTorch)

Implementation of the vanilla federated learning paper : [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).


Experiments are produced on MNIST and CIFAR-10, each with IID and Dirichlet non-IID client splits.

Since the purpose of these experiments are to illustrate the effectiveness of the federated learning paradigm, compact models such as MLP, CNN, and ResNet-18 are used.

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on MNIST and CIFAR-10.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment with MNIST on CNN using automatic device selection (`CUDA > MPS > CPU`):
```
python src/baseline_main.py --dataset=mnist --epochs=10
```
* Or to run it on a specific CUDA GPU (eg: if `cuda:0` is available):
```
python src/baseline_main.py --dataset=mnist --gpu=0 --epochs=10
```
-----

Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR-10 on ResNet-18 (IID):
```
python src/federated_main.py --dataset=cifar --gpu=0 --iid=1 --epochs=10
```
* To run the same experiment under Dirichlet non-IID condition:
```
python src/federated_main.py --dataset=cifar --gpu=0 --iid=0 --dirichlet_alpha=0.1 --epochs=10
```

### Typical Commands

MNIST baseline:
```
python src/baseline_main.py --dataset=mnist --model=cnn --epochs=10 --gpu=0
```

CIFAR-10 baseline:
```
python src/baseline_main.py --dataset=cifar --model=resnet18 --epochs=150 --batch_size=128 --lr=0.1 --scheduler=cosine --norm=batch_norm --gpu=0
```

MNIST federated IID:
```
python src/federated_main.py --dataset=mnist --iid=1 --epochs=200 --num_users=100 --frac=0.1 --local_ep=1 --local_bs=10 --lr=0.01 --gpu=0
```

MNIST federated Dirichlet non-IID:
```
python src/federated_main.py --dataset=mnist --iid=0 --epochs=200 --num_users=100 --frac=0.1 --local_ep=1 --local_bs=10 --lr=0.01 --dirichlet_alpha=0.5 --gpu=0
```

CIFAR-10 federated IID:
```
python src/federated_main.py --dataset=cifar --iid=1 --epochs=1000 --num_users=100 --frac=0.1 --local_ep=1 --local_bs=32 --lr=0.03 --scheduler=cosine --norm=batch_norm --gpu=0
```

CIFAR-10 federated Dirichlet non-IID:
```
python src/federated_main.py --dataset=cifar --iid=0 --epochs=1000 --num_users=100 --frac=0.1 --local_ep=1 --local_bs=32 --lr=0.05 --scheduler=cosine --dirichlet_alpha=0.1 --norm=batch_norm --test_interval=1 --gpu=0
```

CIFAR-10 federated Dirichlet non-IID with GroupNorm for comparison:
```
python src/federated_main.py --dataset=cifar --iid=0 --epochs=1000 --num_users=100 --frac=0.1 --local_ep=1 --local_bs=32 --lr=0.05 --scheduler=cosine --dirichlet_alpha=0.1 --norm=group_norm --test_interval=1 --gpu=0
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'cifar'. Options: 'mnist', 'cifar'
* ```--model:```    Default depends on dataset ('cnn' for MNIST, 'resnet18' for CIFAR-10). Options: 'mlp', 'cnn', 'resnet18'
* ```--gpu:```      Default: auto-select best device (`CUDA > MPS > CPU`). Can also be set to a specific CUDA GPU id.
* ```--epochs:```   Number of rounds of training. Default: 200 for MNIST, 1000 for CIFAR-10.
* ```--lr:```       Learning rate. Default depends on dataset and experiment setting.
* ```--verbose:```  Detailed log outputs. Default: 0. Set to 1 to activate.
* ```--seed:```     Random Seed. Default set to 1.

#### Federated Parameters
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for Dirichlet non-IID.
* ```--dirichlet_alpha:``` Dirichlet concentration for non-IID label skew. Smaller values are more heterogeneous. Default is 0.5.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 1 for MNIST and CIFAR-10 federated runs.
* ```--local_bs:``` Batch size of local updates in each user. Default depends on dataset.

Federated defaults when an option is omitted; explicit command-line values
override these defaults.

| Parameter | MNIST IID | MNIST non-IID | CIFAR-10 IID | CIFAR-10 non-IID |
| --- | --- | --- | --- | --- |
| `--dataset` | `mnist` | `mnist` | `cifar` | `cifar` |
| `--iid` | `1` | `0` | `1` | `0` |
| `--model` | `cnn` | `cnn` | `resnet18` | `resnet18` |
| `--epochs` | `200` | `200` | `1000` | `1000` |
| `--num_users` | `100` | `100` | `100` | `100` |
| `--frac` | `0.1` | `0.1` | `0.1` | `0.1` |
| `--local_ep` | `1` | `1` | `1` | `1` |
| `--local_bs` | `10` | `10` | `32` | `32` |
| `--batch_size` | `64` | `64` | `128` | `128` |
| `--optimizer` | `sgd` | `sgd` | `sgd` | `sgd` |
| `--lr` | `0.01` | `0.01` | `0.03` | `0.03` |
| `--momentum` | `0.9` | `0.9` | `0.9` | `0.9` |
| `--weight_decay` | `0.0` | `0.0` | `0.0005` | `0.0005` |
| `--scheduler` | `none` | `none` | `cosine` | `cosine` |
| `--norm` | `batch_norm` | `batch_norm` | `batch_norm` | `batch_norm` |
| `--dirichlet_alpha` | `0.5` (unused) | `0.5` | `0.5` (unused) | `0.5` |
| `--test_interval` | `1` | `1` | `1` | `1` |
| `--gpu` | auto | auto | auto | auto |
| `--verbose` | `0` | `0` | `0` | `0` |
| `--seed` | `1` | `1` | `1` | `1` |

## Results on MNIST
#### Baseline Experiment:
The experiment involves training a single model in the conventional way.

Parameters: <br />
* ```Optimizer:```    : SGD 
* ```Learning Rate:``` 0.01

```Table 1:``` Test accuracy after training for 10 epochs:

| Model | Test Acc |
| ----- | -----    |
|  MLP  |  92.71%  |
|  CNN  |  98.42%  |

----

#### Federated Experiment:
The experiment involves training a global model in the federated setting.

Federated parameters (default values):
* ```Fraction of users (C)```: 0.1 
* ```Local Batch size  (B)```: 10 
* ```Local Epochs      (E)```: 10 for IID, 1 for non-IID 
* ```Optimizer            ```: SGD 
* ```Learning Rate        ```: 0.01 <br />

```Table 2:``` Test accuracy after training for 10 global epochs with:

| Model |    IID   | Non-IID (equal)|
| ----- | -----    |----            |
|  MLP  |  88.38%  |     73.49%     |
|  CNN  |  97.28%  |     75.94%     |


## Further Readings
### Papers:
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

### Blog Posts:
* [CMU MLD Blog Post: Federated Learning: Challenges, Methods, and Future Directions](https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/)
* [Leaf: A Benchmark for Federated Settings (CMU)](https://leaf.cmu.edu/)
* [TensorFlow Federated](https://www.tensorflow.org/federated)
* [Google AI Blog Post](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)

---

`src.update.LocalUpdate.train_val_test` 每个客户端使用自己的全部本地样本训练，不再拆成 train/validate/test。
```python
trainloader = DataLoader(DatasetSplit(dataset, idxs),
                         batch_size=self.args.local_bs, shuffle=True)
```
