# Seq-CALC

Seq-CALC is a loop closure method for SLAM based on [CALC](https://github.com/rpng/calc) with the help of sequence informance. CALC is a convolutional auto-encoder loop closure model, which has shown competitive performance on both accuracy and time cost against [DBoW2](https://github.com/dorian3d/DBoW2). Inspired with [SeqSLAM](http://www.cim.mcgill.ca/~dudek/417/Resources/seqslam-milford.pdf), our Seq-CALC combines CALC with the sequence information, greatly improving its accurary in complex environment. To ensure real-time, we reduce the dimension of the descriptor from 1064 to 128 using PCA, which leads to a very little accuracy loss but much faster query speed. 

Here is a  C++ library for online SLAM loop closure based on Seq-CALC. The pre-trained CALC model is provided by [CALC](https://github.com/rpng/calc) and can be downloaded on compilation.

## Dependencies

Required:
- OpenCV >= 2.0
- Eigen >= 3.0
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