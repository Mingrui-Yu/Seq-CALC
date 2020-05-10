#include "deeplcd.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

int main(int argc, char** argv){

const int len = 1064, len2 = 128;
    typedef Eigen::Matrix<float, len, 1> VecLong;
    typedef Eigen::Matrix<float, len2, 1> VecShort;

    Eigen::MatrixXd a = Eigen::MatrixXd::Random(len, 1);
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(len, 1);
    VecLong a_long, b_long;
    for(int i = 0; i < len; i++){
        a_long(i, 0) = a(i, 0);
    }
    for(int i = 0; i < len; i++){
        b_long(i, 0) =b(i, 0);
    }

    Eigen::MatrixXd c = Eigen::MatrixXd::Random(len2, 1);
    Eigen::MatrixXd d = Eigen::MatrixXd::Random(len2, 1);
    VecShort a_short, b_short;
    for(int i = 0; i < len2; i++){
        a_short(i, 0) = c(i, 0);
    }
    for(int i = 0; i < len2; i++){
        b_short(i, 0) = d(i, 0);
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    for(int i = 0; i < 100000; i++){
        double prod = a_long.transpose() * b_long;
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
    std::cout << len << ", " << time_used.count() << "s" << std::endl; 

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    for(int i = 0; i < 100000; i++){
        double prod2 = a_short.transpose() * b_short;
    }
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used_2 = std::chrono::duration_cast <std::chrono::duration<double>> (t4 - t3);
    std::cout << len2 << ", "  << time_used_2.count() << "s" << std::endl; 

}