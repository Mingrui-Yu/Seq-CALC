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

    // ---------------------------------- Load the images' path -----------------------------------------
    std::vector<std::string> vstrImageFilenames;
    std::vector<double> vTimestamps;
    std::string sequenceDir = "/media/mingrui/MyPassport/SLAMdatabase/KITTI/gray/00";
    LoadImages(sequenceDir, vstrImageFilenames, vTimestamps);
    // ---------------------------------- Load the groud_truth file -----------------------------------------
    std::string strPathToPose = "/media/mingrui/MyPassport/SLAMdatabase/KITTI/data_odometry_poses/dataset/poses/00.txt";
    std::vector<Mat33, Eigen::aligned_allocator<Mat33>> vPoses_R;
    std::vector<Vec3, Eigen::aligned_allocator<Vec3>> vPoses_t;
    LoadGroundtruthPose(strPathToPose,  vPoses_R, vPoses_t);
     // ---------------------------------- create the file to store test results ----------------------------------
    ofstream f;
    std::string filename = "results/test3/timecost.txt";
    f.open(filename.c_str());
    f << fixed;
    // ----------------------------------------- 一些数据初始化 ----------------------------------------------
    std::vector<bool>vLoopExist;
    std::vector<bool>vQueryCorrect;
    std::vector<double>vQueryScore;

    std::vector<unsigned int> vFrameIdInDatabase; 

    std::chrono::steady_clock::time_point t1, t2;


    deeplcd::DeepLCD lcd;
    lcd.SetParameters(3, 2.0, 0.5, 20); // ds, V_max, V_interval, numMaxResult

    for(size_t ni = 0, nImages = vstrImageFilenames.size(); ni < nImages; ni += 1){
        std::cout << "loading image " << ni << "." << std::endl;
        cv::Mat im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        vFrameIdInDatabase.push_back(ni);

        t1 = std::chrono::steady_clock::now();
        auto descriptor = lcd.CalcDescrOriginalImg(im);
        t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used_extract = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
        // std::cout << time_used.count() << "s" << std::endl; 

        if (ni < 500){
            t1 = std::chrono::steady_clock::now();
            lcd.Add(descriptor, ni);
            t2 = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_used_add = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
            continue;
        }
    

        // DeepLCD 查询
        t1 = std::chrono::steady_clock::now();
        deeplcd::QueryResults qrs = lcd.Query(descriptor, 10, 1); // image, windowSize, numReturnResults
        t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used_query = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);

        t1 = std::chrono::steady_clock::now();
        lcd.AddAfterQuery(ni);
        t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used_add = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
        

        f << std::setprecision(6) << ni << " " << time_used_extract.count() << " " << time_used_add.count() << " " << time_used_query.count() << std::endl;
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