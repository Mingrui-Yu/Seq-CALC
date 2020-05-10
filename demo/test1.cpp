// 保存 KITTI00， KITTI05 上的回环检测的 PR 

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
    std::string sequenceDir = "/media/mingrui/MyPassport/SLAMdatabase/KITTI/gray/05";
    LoadImages(sequenceDir, vstrImageFilenames, vTimestamps);
    // ---------------------------------- Load the groud_truth file -----------------------------------------
    std::string strPathToPose = "/media/mingrui/MyPassport/SLAMdatabase/KITTI/data_odometry_poses/dataset/poses/05.txt";
    std::vector<Mat33, Eigen::aligned_allocator<Mat33>> vPoses_R;
    std::vector<Vec3, Eigen::aligned_allocator<Vec3>> vPoses_t;
    LoadGroundtruthPose(strPathToPose,  vPoses_R, vPoses_t);
     // ---------------------------------- create the file to store test results ----------------------------------
    ofstream f;
    std::string filename = "results/test1/KITTI05.txt";
    f.open(filename.c_str());
    f << fixed;
    // ----------------------------------------- 一些数据初始化 ----------------------------------------------
    std::vector<bool>vLoopExist;
    std::vector<bool>vQueryCorrect;
    std::vector<double>vQueryScore;

    std::vector<unsigned int> vFrameIdInDatabase; 


    deeplcd::DeepLCD lcd;
    lcd.SetParameters(3, 2.0, 0.5, 50); // ds, V_max, V_interval, numMaxResult

    for(size_t ni = 0, nImages = vstrImageFilenames.size(); ni < nImages; ni += 10){
        std::cout << "loading image " << ni << "." << std::endl;
        cv::Mat im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        vFrameIdInDatabase.push_back(ni);

        if (ni < 500){
            lcd.Add(im, ni);
            continue;
        }
        
        int current_id = ni;
        // ------------------   根据 groundtruth 位姿找出数据集中是否存在回环
        bool hasGroundTruthP = false;
        for(auto &db_id: vFrameIdInDatabase){
            bool bGroundTruthCorrect = PoseIsVeryNear(vPoses_R, vPoses_t, current_id, db_id);
            if (std::abs((int)current_id - (int)db_id) > 100 && bGroundTruthCorrect){
                std::cout << std::setprecision(3) << "GroundTruth for Image " 
                    << current_id << ":  db id: " << db_id  << std::endl;
                hasGroundTruthP = true;
                break;
            }
        }
        if(hasGroundTruthP) // 真实存在回环
            vLoopExist.push_back(true);
        else 
            vLoopExist.push_back(false);

        // DeepLCD 查询
        deeplcd::QueryResults qrs = lcd.Query(im, 10, 1); // image, windowSize, numReturnResults
        lcd.AddAfterQuery(ni);

        std::cout << qrs << std::endl;

        deeplcd::query_result qr = qrs[0];
        int loop_id = qr.id;
        std::cout << "query result: " << current_id << ":  loop id: " << loop_id  << ", score: " << qr.score << std::endl;
        
        bool bCorrect = PoseIsVeryNear(vPoses_R, vPoses_t, current_id, loop_id);         
        bool bAcceptable = PoseIsAcceptablyNear(vPoses_R, vPoses_t, current_id, loop_id);

        std::cout << "correct?: " << bAcceptable << std::endl;

        if(bAcceptable){
            vQueryCorrect.push_back(true);
        } else{
            vQueryCorrect.push_back(false);
        }
        vQueryScore.push_back(qr.score);
    }

    assert(vQueryCorrect.size() == vQueryScore.size() && vQueryScore.size() == vLoopExist.size());

    // 结果输出 保存至 txt 文件
    for(size_t i = 0; i < vQueryCorrect.size(); i++){
        f <<  vLoopExist[i] << " " << vQueryCorrect[i] << " " << vQueryScore[i] << std::endl;
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