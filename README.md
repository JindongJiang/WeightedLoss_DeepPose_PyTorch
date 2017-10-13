# WeightedLoss_DeepPose_PyTorch

In the experiment of my previous work [1] for indoor Semantic Segmentation, I use median-frequency weighting [2] to balance the loss from different classes of object. Later when I worked on another project of Structured Prediction, I start to think about bringing weighted loss to these regression task since the data for this regression tasks are also unbalance, for example, a standing person in the center of an image appears much more then a flipping person in the corner.

In this work, I try to use principal component analysis following with a kernel density estimation to estimate the distribution of the human pose. And use the median-frequency weighting to modify the loss function. However, it worth noticed that after many times of experiment for various combination of PCA and KDE parameters. This method fail to accelerate the learning (at least not in long run) or decrease the testing error. 😰😰

Nevertheless, this work are still useful for people trying to program in PyTorch or conduct experiment on pose estimate. Also this work has the usage `weighted loss` as a option, you can disable `weighted loss` and totally treat it as a primitive implementation of deep pose [3] with only stage 1. 🤓🤓

For those who achieves a better result base on this idea, please cite this repo in your paper or website, and it would be great if you can inform me about the work so that I know which part in my original idea is the problem. 🤔🤔

Beside, this is not the only one experiment I code for this "weighted loss" idea, I also implemented convolutional pose machine and a facial expression detection program in PyTorch, I will submit them very soon in other repos.

## Prerequisites
- [ ] TODO

## Download datas
- [ ] TODO

## Usage
- [ ] TODO

## References

[1] [J. Jiang, Z. Zhang, Y. Huang, and L. Zheng. Incorporating depth into both cnn and crf for indoor semantic segmentation. arXiv preprint arXiv:1705.07383, 2017.](https://arxiv.org/abs/1705.07383)

[2] [D. Eigen and R. Fergus, “Predicting depth, surface normals and semantic labels with a common multi-scale convolutional
architecture,” arXiv:1411.4734, 2014.](https://arxiv.org/abs/1606.00915)

[2] [A. Toshev and C. Szegedy. Deeppose: Human pose estimation via deep neural networks. CoRR, abs/1312.4659, 2013.](http://bit.ly/2ylKJCj)
