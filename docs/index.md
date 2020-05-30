<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>


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
  <img width="60%" src="https://raw.githubusercontent.com/Mingrui-Yu/Seq-CALC/master/docs/sequence.png">
</p>

In Seq-CALC, we maintain a local similarity score matrix $S$, which records the similarity scores between every two images in the database (as image above shows). Darker element means higher similarity score. In online SLAM loop detection, we only compare query frames with previous frames, so the matrix is actually a lower triangular matrix. As the sequence length $d_s$ is finite, we only need to store the recent $d_s$ rows of the matrix $S$, as the part in green box.

The process of sequence matching is as follows. For a query frame $I_T$, first, it will find $K$ candidate frames with the highest similarity scores. Then, it will do sequence matching starting from these $K$ candidates. Take the candidate $I_c$ as an example: starting from element $(T, c)$ in matrix $S$, trajectories can be drawn like the red lines. The slope $k$ of the line is related to the speeds when the agent pass this position in two times:

$$  V = \frac{v_{current}}{v_{loop}} = -k  $$

As the relative speed is unknown, we need to search for the optimal speed at intervals ${V_{interval}}$, like the several red lines in above images. In Seq-CALC, users only need to set $V_{max}$ and ${V_{interval}}$, and the values of all $V$ to search are:

$$  V \in \{  \frac{1}{V_{max}}, ..., 1.0, 1.0 + V_{interval}, ..., V_{max}   \}$$

The average of the $S$ elements each red line passes is the trajectory's sequence matching score $s^{seq}$:

$$  s^{seq}(T, c, V) = \frac{1}{d_s} \sum_{t=T-d_s-1}^{T}S(t,j)   $$
$$ j = c - V(T-t) $$

As a result, the final sequence similarity score between candidate $I_c$ and query frame $I_T$ is:

$$ s^{seq}(T,c) = {max}_V \{  s^{seq}(T,c,V) \}   $$

According to the the sequence similarity score, we can find the best candidate loop frame. We will accept it if the score is higher than a pre-set threshold.

## Evaluation

All the experiments for resource costs are executed on NVIDIA Jetson TX2.

### Offline detection
We evaluate our CALC's offline performance on four place recoginition datasets with different variations:
* GardensPoint Day Left vs. Day Right
* GardensPoint Day Left vs. Night Right
* CampusLoop
* Nordland

The comparison methods include:
* [DBoW2](https://github.com/dorian3d/DBoW2)
* AlexNet-conv3
* [VGGNet-pool4+trainedFC](https://github.com/jmfacil/single-view-place-recognition)
* [NetVLAD](https://github.com/uzh-rpg/netvlad_tf_open)


Picture below shows the PR curves of these methods in the four testsets:

<p align="center">
  <img width="80%" src="https://raw.githubusercontent.com/Mingrui-Yu/Seq-CALC/master/docs/experiment_calc.png">
</p>

Picture below shows the performance of CALC+PCA:

<p align="center">
  <img width="80%" src="https://raw.githubusercontent.com/Mingrui-Yu/Seq-CALC/master/docs/experiment_calcpca.png">
</p>

Picture below shows the performance of Seq-CALC with different $d_s$, compared with original CALC and SeqSLAM ($d_s$ = 3):

<p align="center">
  <img width="80%" src="https://raw.githubusercontent.com/Mingrui-Yu/Seq-CALC/master/docs/experiment_seqcalc.png">
</p>

Table below shows the resource cost of these methods:

|         Method         | Time cost for extracting descriptors (ms) | Model file size (MB) | Descriptor dimension |
|:----------------------:|:-----------------------------------------:|:--------------------:|:--------------------:|
|          CALC          |                    2.46                   |          44          |         1064         |
|      AlexNet-conv3     |                    61.4                   |          244         |         64896        |
| VGGNet-pool4+trainedFC |                    68.5                   |          246         |          128         |
|         NetVLAD        |                    325                    |          596         |         4096         |
|        Seq-CALC        |                    2.65                   |          44          |          128         |



### Online detection

We compare the performance of the C++ libraries of Seq-CALC, [DBoW3](https://github.com/rmsalinas/DBow3) and CALC in online SLAM loop detection. 

Picture below shows the loops detected by Seq-CALC in KITTI 00 (left) and KITTI 05 (right). Z axis is the time.

<p align="center">
  <img width="80%" src="https://raw.githubusercontent.com/Mingrui-Yu/Seq-CALC/master/docs/experiment_KITTI_3d.png">
</p>

Picture below shows the PR curves in KITTI 00 (above) and KITTI 05 (below):

<p align="center">
  <img width="80%" src="https://raw.githubusercontent.com/Mingrui-Yu/Seq-CALC/master/docs/experiment_KITTI_PR.png">
</p>

Table below shows the time cost of these three methods (KITTI 00, 4541 frames, all are average values). Notice that for DBoW3 extracting descriptors contrains two steps: 1) extracting features (in bracket); 2) transforming to BoW vector.


|                     |     DBoW3    |   CALC  | Seq-CALC |
|:-------------------:|:------------:|:-------:|:--------:|
| Extract descriptors (ms) | (37.3+) 2.75 |   2.46  |   2.65   |
|   Add to database (ms)   |     0.153    | 0.00643 |  0.00576 |
|        Query (ms)        |     3.37     |   2.03  |   0.386  |



Picture below shows the time cost of query in KITTI 00 (4541 frames):

<p align="center">
  <img width="80%" src="https://raw.githubusercontent.com/Mingrui-Yu/Seq-CALC/master/docs/query_speed_3method.png">
</p>