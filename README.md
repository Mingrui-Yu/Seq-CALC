# Seq-CALC

Seq-CALC is a loop detection method for SLAM based on [CALC](https://github.com/rpng/calc) with the help of sequence match. CALC is a convolutional auto-encoder loop detection model, which has shown competitive performance on both accuracy and time cost against [DBoW2](https://github.com/dorian3d/DBoW2). Inspired with [SeqSLAM](http://www.cim.mcgill.ca/~dudek/417/Resources/seqslam-milford.pdf), our Seq-CALC combines CALC with  sequence match, greatly improving its accurary in complex environment. To ensure real-time, we reduce the dimension of the descriptor from 1064 to 128 using PCA, which leads to a very little accuracy loss but much faster query speed.  We also do some further optimization for real-time online loop detection.

Here is a  C++ library for online SLAM loop detection based on Seq-CALC. The pre-trained CALC model is provided by [CALC](https://github.com/rpng/calc) and can be downloaded on compilation. Notice that now this library is totally designed for online SLAM loop detection, so if you want to use this method in pure place recogintion work, some modification is required.

## Dependencies

Required:
- OpenCV 3
- Eigen
- Boost filesystem
- Caffe 

Optional but HIGHLY Recommended:
- CUDA

## To Compile

```
$ mkdir build && cd build
$ cmake .. && make 

# Already set to Release build. Notice if you are using VSCode, you need to select CMake:Release manually 
```

Note that if your caffe is not installed in ~/caffe, you must use 

```
$ cmake -DCaffe_ROOT_DIR=</path/to/caffe> .. && make
```
instead. Or you can change Caffe PATH  in CMakeLists.txt.

## Usage
There is a simple demo in demo/demo.cpp, which detects loops online in [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) sequence (gray images). To run the demo, please execute the following command in SeqCALC folder:
```
./build/demo  PATH_TO_KITTI00_GRAY_DIR
```
For more functions you can refer to include/deeplcd/deeplcd.h.

---
**This project is a part of my undergraduate thesis. More details of the method and library will be added after my defense.**