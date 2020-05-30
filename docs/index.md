# Seq-CALC: Lightweight and Robust Deep Loop Detection for SLAM

## Introduction

Simultaneous localization and mapping (SLAM) has been widely applied in mobile robots, autonomous driving and other fields. Efficient and robust loop detection is significant in SLAM system, which reduces the accumulative error by detecting loops and then correcting the trajectory and map. In visual SLAM, appearance-based loop detection has become the mainstream due to its adaptability. However, the complex variations in real environment bring great challenges to identify loops by raw images, such as variations of illumination, seasons and viewpoints.

Currently the most widely used loop detection method in visual SLAM system is Bag-of-Words (BoW). However, because of the limitation of artificial features, BoW cannot deal with complex illumination variations. Recently, the development of deep learning has encouraged researchers to use deep network to extract images’ features and detect loops, which has shown higher accuracy. Unfortunately, the deep-learning based methods tend to rely on large-scale networks, and be time-consuming in computation. Besides, they sometimes suffer in adapting to different types of scenes. Therefore, it's hard to embed them in SLAM systems with high real-time requirement and limited computing resources.

We propose a lightweight SLAM loop detection method based on deep learning, which is named Seq-CALC. An unsupervised denoising auto-encoder network is trained to extract an image's deep descriptor, which uses the projective-transformed images as input data, and HOG descriptor as the reconstruction object. Besides, we apply PCA to reduce the descriptor's dimension, and combine linear query with fast approximate nearest neighbor search to further improve the efficiency. With the help of sequence match, we significantly improve its accurary. The results of experiments on NVIDIA TX2 demonstrate that, our Seq-CALC outperforms BoW in terms of both accurary and efficiency on various challenging datasets. Its accuary is even close to NetVLAD under certain conditions, which is the state-of-art method in the field of place recognition. We open-source the C++ library of Seq-CALC, which can be conveniently embedded in any SLAM system.

## Method

Our Seq-CALC is built based on [CALC](https://github.com/rpng/calc), which trains an unsupervised denoising auto-encoder network to extract an image's deep descriptor. It uses the randomly projective-transformed images as the noisy input, and use the HOG descriptor of the raw image as the reconstruction object. The model is trained on the whole [Places](http://places2.csail.mit.edu/download.html) dataset, which contains about 8 million images for scene classification. We utilize the pre-trained model of CALC. For more details, you can refer to their [paper](http://www.roboticsproceedings.org/rss14/p32.pdf) in RSS 2018.

We apply PCA to recude the dimension of the descriptors extracted by CALC from 1064 to 128. The PCA transform matrix is solved on val_256 subset in Places, which contains 36,500 images.

We calculate the similarity score between two images according to the cosine similarity of their descriptors , which is more efficient than calculating Euclidean distance. We use linear search to query the image with highest similarity score from the database, because experiments show that linear search is more efficent than [FLANN](https://github.com/mariusmuja/flann) (Fast Library for Approximate Nearest Neighbors) when the database size is smaller than 10 thousand frames. The reason is that FLANN is designed for offline matching, but in real-time online SLAM loop detection, it needs to keep adding new images to the database, which is relatively time-consuming. 

The time-adjacent keyframes in SLAM is usually taken in almost the same places, so the loop is usually not just connection between two isolated keyframes, but two frame sequences. Inspired by [SeqSLAM](http://www.cim.mcgill.ca/~dudek/417/Resources/seqslam-milford.pdf), we do sequence matching to determine the final loop, which greatly improves the accurary. 

<p align="center">
  <img width="50%" src="https://raw.githubusercontent.com/Mingrui-Yu/Seq-CALC/master/docs/sequence.png">
</p>


