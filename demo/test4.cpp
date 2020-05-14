#include "deeplcd.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

int main(int argc, char** argv){

    
    
    const int len = 128;  // 64896
    unsigned long N = 1e6; 
    int nn = 10;
    unsigned int interval = N / nn;
    typedef Eigen::Matrix<float, len, 1> VecLong;

    // ---------------------------------- create the file to store test results ----------------------------------
    ofstream f;
    std::string filename = "results/test4/timecost_" + std::to_string(len) + ".txt";
    f.open(filename.c_str());
    f << fixed;


    std::vector<std::vector<double> > vvTimeCost(10);


    for(size_t k = 0; k < 10; k++){

        vvTimeCost[k].reserve(N/interval);

        VecLong a_long, b_long;

        int numRandom = 1000;
        std::vector<VecLong> v_a(numRandom);
        
        for(size_t c = 0; c < numRandom; c++){
            Eigen::MatrixXd a = Eigen::MatrixXd::Random(len, 1);
            for(int i = 0; i < len; i++){
                a_long(i, 0) = a(i, 0);
            }
            v_a[c] = a_long;
        }

        // Eigen::MatrixXd a = Eigen::MatrixXd::Random(len, 1);
        // for(int i = 0; i < len; i++){
        //     a_long(i, 0) = a(i, 0);
        // }
        
        Eigen::MatrixXd b = Eigen::MatrixXd::Random(len, 1);
        for(int i = 0; i < len; i++){
            b_long(i, 0) =b(i, 0);
        }

        double maxScore = 0.0;
        std::chrono::steady_clock::time_point t1, t2;
        std::chrono::duration<double> time_used;

        int j = 0;
        t1 = std::chrono::steady_clock::now();
        for(unsigned long i = 0; i < N; i++){
            double prod = v_a[j].transpose() * b_long;

            if(prod > maxScore) maxScore = prod;

            if( (i+1) % interval == 0){
                t2 = std::chrono::steady_clock::now();
                time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
                vvTimeCost[k].push_back(time_used.count());
            }

            j++;
            if (j >= numRandom) j = 0;
        }
        
        std::cout << len << ", " << time_used.count() << "s" << std::endl; 

        std::cout << maxScore << std::endl;
    }

    for(size_t i = 0; i < nn; i++){
        f  << (i+1) * interval;
        for(int k = 0; k < 10; k++){
            f << " " << vvTimeCost[k][i];
        }
        f  << std::endl;
    }
    
    
    return 0;
}






// int main(int argc, char** argv){

    
    
//     const int len = 128;  // 64896
//     unsigned long N = 1e6; 
//     int nn = 10;
//     unsigned int interval = N / nn;
//     typedef Eigen::Matrix<float, len, 1> VecLong;

//     // ---------------------------------- create the file to store test results ----------------------------------
//     ofstream f;
//     std::string filename = "results/test4/timecost_" + std::to_string(len) + ".txt";
//     f.open(filename.c_str());
//     f << fixed;


//     std::vector<std::vector<double> > vvTimeCost(10);


//     for(size_t k = 0; k < 10; k++){

//         vvTimeCost[k].reserve(N/interval);

//         VecLong a_long, b_long;

//         int numRandom = 1000;
//         std::vector<VecLong> v_a(numRandom);
        
//         for(size_t c = 0; c < numRandom; c++){
//             Eigen::MatrixXd a = Eigen::MatrixXd::Random(len, 1);
//             for(int i = 0; i < len; i++){
//                 a_long(i, 0) = a(i, 0);
//             }
//             v_a[c] = a_long;
//         }

//         // Eigen::MatrixXd a = Eigen::MatrixXd::Random(len, 1);
//         // for(int i = 0; i < len; i++){
//         //     a_long(i, 0) = a(i, 0);
//         // }
        
//         Eigen::MatrixXd b = Eigen::MatrixXd::Random(len, 1);
//         for(int i = 0; i < len; i++){
//             b_long(i, 0) =b(i, 0);
//         }

//         double maxScore = 0.0;
//         std::chrono::steady_clock::time_point t1, t2;
//         std::chrono::duration<double> time_used;

//         int j = 0;
//         t1 = std::chrono::steady_clock::now();
//         for(unsigned long i = 0; i < N; i++){
//             double prod = v_a[j].transpose() * b_long;

//             if(prod > maxScore) maxScore = prod;

//             if( (i+1) % interval == 0){
//                 t2 = std::chrono::steady_clock::now();
//                 time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
//                 vvTimeCost[k].push_back(time_used.count());
//             }

//             j++;
//             if (j >= numRandom) j = 0;
//         }
        
//         std::cout << len << ", " << time_used.count() << "s" << std::endl; 

//         std::cout << maxScore << std::endl;
//     }

//     for(size_t i = 0; i < nn; i++){
//         f  << (i+1) * interval;
//         for(int k = 0; k < 10; k++){
//             f << " " << vvTimeCost[k][i];
//         }
//         f  << std::endl;
//     }
    
    
//     return 0;
// }