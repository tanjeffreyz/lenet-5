<!--  
mlpi
title: Gradient-Based Learning Applied to Document Recognition (LeNet-5)

images: models/lenet-5/02_02_2022/15_27_14/losses.png
category: Architectures/Convolutional Neural Networks
-->


<h1 align="center">LeNet-5</h1>
 
An implementation of the LeNet-5 architecture using PyTorch. Here are the losses and accuracies over 25 epochs, with the minimum test loss and maximum test accuracy labeled:
<div align="center">
  <img src="https://github.com/tanjeffreyz02/py-lenet-5/blob/cc14503e76c8d41975570e4f0d84af6847bff077/models/lenet-5/02_02_2022/15_27_14/losses.png" />
</div>
Overall, the LeNet-5 architecture was able to achieve an accuracy of roughly 99.1% (error rate of 0.9%) on the MNIST dataset, which supports the 0.95% error rate found in [1].

<h2>Instructions</h2>
<ol>
  <li>
    Run <code>python -m pip install -r requirements.txt</code>
  </li>
  <li>
    Run <code>python train.py</code> to train LeNet-5
  </li>
  <li>
    In a new command line terminal, run <code>tensorboard --logdir=runs</code> to visualize the training progress
  </li>
</ol>

<h2>References</h2>

[[1](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)] Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," in Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998, doi: 10.1109/5.726791.
