// 保存 KITTI00， KITTI05 上的回环检测结果，用于绘制3D轨迹图+回环边

#include "deeplcd.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, 3, 1> Vec3;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

void LoadGroundtruthPose(const string &strPose, 
    std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t);

bool PoseIsVeryNear(std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t,
    int current_id, int loop_id);

bool PoseIsAcceptablyNear(std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t,
    int current_id, int loop_id);


// --------------------------------------------------------------------------------------------------------------

int main(int argc, char** argv){

    // ---------------------------------- create the file to store test results ----------------------------------
    ofstream f;
    std::string filename = "results/test5/descriptors.txt";
    f.open(filename.c_str());
    f << fixed;

    std::vector<std::string> vId(22);
    vId[0] = "00";
    vId[1] = "01";
    vId[2] = "02";
    vId[3] = "03";
    vId[4] = "04";
    vId[5] = "05";
    vId[6] = "06";
    vId[7] = "07";
    vId[8] = "08";
    vId[9] = "09";
    vId[10] = "10";
    vId[11] = "11";
    vId[12] = "12";
    vId[13] = "13";
    vId[14] = "14";
    vId[15] = "15";
    vId[16] = "16";
    vId[17] = "17";
    vId[18] = "18";
    vId[19] = "19";
    vId[20] = "20";
    vId[21] = "21";


    for (int seq = 0; seq < 22; seq++)
    {
        // ---------------------------------- Load the images' path -----------------------------------------
        std::vector<std::string> vstrImageFilenames;
        std::vector<double> vTimestamps;
        std::string sequenceDir = "/media/mingrui/MyPassport/SLAMdatabase/KITTI/gray/" + vId[seq];
        std::cout << sequenceDir << std::endl;
        LoadImages(sequenceDir, vstrImageFilenames, vTimestamps);

        
        // ----------------------------------------- 一些数据初始化 ----------------------------------------------
        std::vector<bool>vLoopExist;
        std::vector<bool>vQueryCorrect;
        std::vector<double>vQueryScore;

        std::vector<unsigned int> vFrameIdInDatabase; 

        std::chrono::steady_clock::time_point t1, t2;


        deeplcd::DeepLCD lcd;
        lcd.SetParameters(3, 2.0, 0.5, 20); // ds, V_max, V_interval, numMaxResult

        for(size_t ni = 0, nImages = vstrImageFilenames.size(); ni < nImages; ni += 5){
            std::cout << "loading image " << ni << "." << std::endl;
            cv::Mat im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
            vFrameIdInDatabase.push_back(ni);


            auto descriptor = lcd.CalcDescrOriginalImg(im);

            f << std::setprecision(6) << " " << ni;
            for (int j = 0; j < 128; j++){
                f << std::setprecision(6) << " " << descriptor(j, 0);
            }
            f << std::setprecision(6) << std::endl;
            
        }
    }
    
    

    f.close();

    return 0;
}
















// --------------------------------------------------------------------------------------------------------------------------

bool PoseIsVeryNear(std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t,
    int current_id, int loop_id){

    Eigen::AngleAxisd rotation_vector;
    rotation_vector.fromRotationMatrix(vPoses_R[current_id].inverse() * vPoses_R[loop_id]);

    bool bCorrect = ((vPoses_t[current_id] - vPoses_t[loop_id]).norm() < 10)
                                        && ((std::abs(rotation_vector.angle()) < 3.14 / 6));
                                    
    return bCorrect;
}

// --------------------------------------------------------------------------------------------------------------------------

bool PoseIsAcceptablyNear(std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t,
    int current_id, int loop_id){

    Eigen::AngleAxisd rotation_vector;
    rotation_vector.fromRotationMatrix(vPoses_R[current_id].inverse() * vPoses_R[loop_id]);

    bool bCorrect = ((vPoses_t[current_id] - vPoses_t[loop_id]).norm() < 20)
                                        && ((std::abs(rotation_vector.angle()) < 3.14 / 6));
                                    
    return bCorrect;
}




// -----------------------------------------------------------------------------------------------------------------

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";  // 使用 rgb 图

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}


// ---------------------------------------------------------------------------------------

void LoadGroundtruthPose(const string &strPose, 
    std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t){

    vPoses_R.clear();
    vPoses_t.clear();

    ifstream fPoses;
    std::string strPathPoseFile = strPose;
    fPoses.open(strPathPoseFile.c_str());
    while(!fPoses.eof())
    {
        double R0, R1, R2, R3, R4, R5, R6, R7, R8;
        double t0, t1, t2;
        fPoses >> R0 >> R1 >> R2 >> t0
                      >> R3 >> R4 >> R5 >> t1
                      >> R6 >> R7 >> R8 >> t2;

        Mat33 R; 
        R << R0, R1, R2,
                  R3, R4, R5,
                  R6, R7, R8;
        Vec3 t; 
        t << t0, t1, t2;

        vPoses_R.push_back(R);
        vPoses_t.push_back(t);
    }

    fPoses.close();
}