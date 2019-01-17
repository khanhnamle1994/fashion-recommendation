# Fashion Recommendation

In this project, I create an end-to-end solution for large-scale clothing retrieval and visual recommendation on fashion images. More specifically, my system can learn the important regions in an image and generate diverse recommendations based on such semantic similarity.

## Writeup
- [Project Paper](https://github.com/khanhnamle1994/fashion-recommendation/blob/master/JamesLeProjectFinalPaper.pdf)
- [Project Presentation](https://github.com/khanhnamle1994/fashion-recommendation/blob/master/Clothing%20Retrieval%20and%20Visual%20Recommendation%20for%20Fashion%20Images.pdf)

## Dataset
1. [Fashion144k](https://esslab.jp/~ess/en/data/fashion144k_stylenet/)
2. [DeepFashion In-Shop Retrieval](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)

## Code

1. [train_fashion_144k.py](https://github.com/khanhnamle1994/fashion-recommendation/blob/master/code/train_fashion_144k.py): This code is used to train the model on Fashion144k dataset.
2. [test_deep_fashion_in_shop.py](https://github.com/khanhnamle1994/fashion-recommendation/blob/master/code/test_deep_fashion_in_shop.py): This code is used to test the model on the In-Shop Retrieval benchmark of Deep Fashion dataset. The model weight file from the training step is stored [here](https://github.com/khanhnamle1994/fashion-recommendation/blob/master/model/model_fashion_144k.pt).
3. [extract_features.py](https://github.com/khanhnamle1994/fashion-recommendation/blob/master/code/extract_features.py): This code is used to extract features using the network from the images which can then be compared to get the testing accuracy.

## Requirements
- [Pytorch](https://pytorch.org/) latest version
- [sklearn](https://scikit-learn.org/) latest version
- [numpy](http://www.numpy.org/) latest version
- PyTorch's [Encoding](https://hangzhang.org/PyTorch-Encoding/nn.html)
