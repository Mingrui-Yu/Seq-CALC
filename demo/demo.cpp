#include "deeplcd.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;



void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);


// --------------------------------------------------------------------------------------------------------------

int main(int argc, char** argv){

    std::vector<std::string> vstrImageFilenames;
    std::vector<double> vTimestamps;
    std::string sequenceDir = "/home/mingrui/Mingrui/SLAMProject/00";
    LoadImages(sequenceDir, vstrImageFilenames, vTimestamps);


    deeplcd::DeepLCD lcd;
    lcd.SetParameters(3, 2.0, 0.5, 20); // ds, V_max, V_interval, numMaxResult

    for(size_t ni = 0, nImages = vstrImageFilenames.size(); ni < nImages; ni += 10){
        std::cout << "loading image " << ni << "." << std::endl;
        cv::Mat im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        if (ni <= 500){
            lcd.Add(im, ni);
        }else{
            deeplcd::QueryResults qr = lcd.Query(im, 10, 1); // image, windowSIze, numReturnResults
            lcd.AddAfterQuery(ni);
            std::cout << qr << std::endl;
        }
    }

    




    return 0;
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