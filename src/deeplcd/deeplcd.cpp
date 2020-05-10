#include "deeplcd.h"

#include <opencv2/core.hpp>
#include <algorithm> 
#include <iostream>
#include <chrono>

namespace deeplcd{

DeepLCD::DeepLCD(const std::string& network_definition_file, 
	const std::string& pre_trained_model_file,
	const std::string &pca_params_dir,
	int gpu_id){

	std::string mode = "CPU";
	if(gpu_id >= 0){
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
		caffe::Caffe::SetDevice(gpu_id);
		mode = "GPU";
	}
	else {
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
	}
	clock_t begin = clock();
	autoencoder = new caffe::Net<float>(network_definition_file, caffe::TEST);	
	autoencoder->CopyTrainedLayersFrom(pre_trained_model_file);
	clock_t end = clock();
	std::cout << "\nCaffe mode = " << mode << "\n";
	std::cout << "Loaded CALC model in " <<   double(end - begin) / CLOCKS_PER_SEC << " seconds\n";
	autoencoder_input = autoencoder->input_blobs()[0]; // construct the input blob shared_ptr
	autoencoder_output = autoencoder->output_blobs()[0]; // construct the output blob shared_ptr

	LoadPCAparams(pca_params_dir, _PCAtransformMatrix, _PCAmean);

	for (float v = V_max; v > 1.0 + 1e-2; v -= V_interval){
		_mvVelocity.push_back(1.0 / v);
	}
	for (float v = 1.0; v <= V_max + 1e-2; v += V_interval){
		_mvVelocity.push_back(1.0 * v);
	}
}

// -------------------------------------------------------------------------------------------

void DeepLCD::SetParameters(int ds_, int v_max, int v_interval,  int n_max_result){
	ds = ds_;
	V_max = v_max;
	V_interval = v_interval;
	numMaxResult = n_max_result;
}

// -------------------------------------------------------------------------------------------

void DeepLCD::Add(const cv::Mat &img, unsigned int imageId){
	DescrVector descr = CalcDescrOriginalImg(img);
	Add(descr, imageId);
}

// -------------------------------------------------------------------------------------------

void DeepLCD::Add(DescrVector descr_, unsigned int imageId){
	DescrVector descr = descr_;
	std::vector<float> vScores;
	vScores.reserve(_mDatabase.mvDescriptorVectors.size());
	for(size_t i = 0, N =  _mDatabase.mvDescriptorVectors.size(); i < N; i++){
		float simScore = Score(descr, _mDatabase.mvDescriptorVectors[i]);
		vScores.push_back(simScore);
	}

	_mDatabase.mvImageIds.push_back(imageId);
	_mDatabase.mvDescriptorVectors.push_back(descr);
	_mDatabase.mvvScores.push_back(vScores);

	// if(_mDatabase.mvImageIds.size() == 1){
	// 	_mDatabase.mDescriptorsMatrix = descr.transpose();
	// }else{
	// 	int numRows = _mDatabase.mDescriptorsMatrix.rows();
	// 	int numCols = _mDatabase.mDescriptorsMatrix.cols();
	// 	_mDatabase.mDescriptorsMatrix.conservativeResize(
	// 			numRows + 1, numCols);
	// 	_mDatabase.mDescriptorsMatrix.row(numRows) = descr.transpose();
	// }	
}

// -------------------------------------------------------------------------------------------
QueryResults DeepLCD::Query(const cv::Mat &img, int windowSize, int numReturnResults){

	assert(_mDatabase.mvImageIds.size() == _mDatabase.mvDescriptorVectors.size()
		&& _mDatabase.mvDescriptorVectors.size() == _mDatabase.mvvScores.size());
	
	DescrVector descr = CalcDescrOriginalImg(img);

	return Query(descr, windowSize, numReturnResults);
}


// ---------------------------------------------------------------------------------------------------------------------

QueryResults DeepLCD::Query(DescrVector descr_, int windowSize, int numReturnResults){

	assert(_mDatabase.mvImageIds.size() == _mDatabase.mvDescriptorVectors.size()
		&& _mDatabase.mvDescriptorVectors.size() == _mDatabase.mvvScores.size());
	
	DescrVector descr = descr_;
	
	int dbSize =  _mDatabase.mvImageIds.size();
	numMaxResult = std::min(dbSize, numMaxResult);

	std::vector<float> vScores;
	vScores.reserve(dbSize);

	QueryResults goodResults(numMaxResult);
	goodResults.invalidate();
	QueryResults seqGoodResults(numReturnResults);
	seqGoodResults.invalidate();

	if (dbSize <= windowSize){
		std::cerr << "Warning: the size of database is smaller than the window size, cannot query a similiar image!" << std::endl;
	}

	for(size_t i = 0, N = dbSize; i < N; i++){
		float simScore = Score(descr, _mDatabase.mvDescriptorVectors[i]);
		vScores.push_back(simScore);
		if ( i + windowSize  < dbSize){
			query_result q(i, simScore);
			goodResults.insertInPlace(q);
		}
	}
	
	unsigned int currentId = (dbSize-1) + 1;
	int T = currentId;
	for (auto &currentGoodQuery: goodResults){
		if (currentGoodQuery.score == -1) continue;
		unsigned int goodId = currentGoodQuery.id;
		int s = goodId;
		int temp_ds = 1;
		float seqScore = vScores[s];

		for (float V: _mvVelocity){
			for(int t = T - ds + 1; t < T; t++){
				if (t < 0) continue;

				int j = int(s + V * (t - T) + 0.5);
				if (j < 0) 
					j = 0;
				else if(j >= _mDatabase.mvvScores[t].size()) 
					j = _mDatabase.mvvScores[t].size() - 1;

				seqScore += _mDatabase.mvvScores[t][j];
				temp_ds++;
			}
			seqScore /= temp_ds;
			query_result q(_mDatabase.mvImageIds[goodId], seqScore);
			seqGoodResults.insertInPlace(q);
		}
	}

	_mCurrentDescrVector = descr;
	_mvCurrentScores = vScores;

	for (auto &r: goodResults){
		if(r.score == -1) continue;
		r.id = _mDatabase.mvImageIds[r.id];
	}

	_bHasQuery = true;

	return seqGoodResults;
}

// ---------------------------------------------------------------------------------------

void DeepLCD::AddAfterQuery(unsigned int imageId){
	if(! _bHasQuery){
		std::cerr << "Error: Add(imageId) could only be done after a query" << std::endl;
	}
	_mDatabase.mvImageIds.push_back(imageId);
	_mDatabase.mvDescriptorVectors.push_back(_mCurrentDescrVector);
	_mDatabase.mvvScores.push_back(_mvCurrentScores);

	_bHasQuery = false;
}

// ----------------------------------------------------------------------------------------

const float DeepLCD::Score(const DescrVector& d1, const DescrVector& d2)
{
	float result = d1.transpose() * d2;
	return result;
}


// ---------------------------------------------------------------------------------------------------------

void QueryResults::insertInPlace(const query_result& q) 
{
	if (q.score > at(size()-1).score)
	{	
		pop_back();
		QueryResults::iterator itToInsert = std::lower_bound(begin(), end(), q,
			[](const query_result& q1, const query_result& q2){return q1.score > q2.score;});
		insert(itToInsert, q);
	}
}


// ----------------------------------------------------------------------------------------
DescrVector DeepLCD::CalcDescrOriginalImg(const cv::Mat& originalImg){
	assert(!originalImg.empty());
	// cv::GaussianBlur(originalImg, originalImg, cv::Size(7, 7), 0);

	cv::Size _sz(160, 120);
	cv::Mat imResize;
	cv::resize(originalImg, imResize, _sz);

	if(imResize.channels() > 1){
		cv::cvtColor(imResize, imResize, cv::COLOR_BGR2GRAY);
	}

	return CalcDescr(imResize);
}
// ----------------------------------------------------------------------------------------
DescrVector DeepLCD::CalcDescr(const cv::Mat& im_){
	// the input image needs to be resized before
	
	std::vector<cv::Mat> input_channels(1); //We need this wrapper to place data into the net. Allocate space for at most 3 channels	
	int w = autoencoder_input->width();
	int h = autoencoder_input->height();
	float* input_data = autoencoder_input->mutable_cpu_data();
	cv::Mat channel(h, w, CV_32FC1, input_data);
	input_channels.emplace(input_channels.begin(), channel);
	input_data += w * h;
	cv::Mat im(im_.size(), CV_32FC1);
	im_.convertTo(im, CV_32FC1, 1.0/255.0); // convert to [0,1] grayscale. Place in im instead of im_
	// This will write the image to the input layer of the net
	cv::split(im, input_channels);
	autoencoder->Forward(); // Calculate the forward pass
	const float* tmp_descr;
	tmp_descr = autoencoder_output->cpu_data(); 
	int p = autoencoder_output->channels(); // Flattened layer get the major axis in channels dimension

	// We need to copy the data, or it will be overwritten on the next Forward() call
	// We may have a TON of desciptors, so allocate on the heap to avoid stack overflow
	int sz = p * sizeof(float);
	float* descr_ = (float*)std::malloc(sz);
	std::memcpy(descr_, tmp_descr, sz);

	assert(p == 1064);

	Eigen::Matrix<float, 1064, 1> descriptor1064;
	for(int i = 0; i < p; i++){
		descriptor1064(i, 0) = *(descr_ + i);
	}
	descriptor1064 /= descriptor1064.norm();

	DescrVector descriptor;
	descriptor = _PCAtransformMatrix * descriptor1064;  // (128, 1064) * (1064, 1) = (128, 1)
	// normalization
	descriptor /= descriptor.norm();

	return descriptor;
}
	

// -----------------------------------------------------------------------------------------------------------------

void DeepLCD::LoadPCAparams(const std::string &strPathToDir,
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &transformMatrix, 
	Eigen::Matrix<float, 1064, 1> &mean){

	std::cout << "Loading PCA parameters ... " << std::endl;
    
	std::ifstream fileTransformMatrix, fileMean;
    std::string strFileTransformMatrix = strPathToDir + "/PCAtransformMatrix.txt";
	std::string strFileMean = strPathToDir + "/PCAmean.txt";
    fileTransformMatrix.open(strFileTransformMatrix.c_str());

    while(!fileTransformMatrix.eof()){
        for(size_t i = 0; i < 128; i++){
			for(size_t j = 0; j < 1064; j++){
				fileTransformMatrix >> transformMatrix(i, j);
			}
		}
    }

	fileMean.open(strFileMean.c_str());
	while(!fileMean.eof()){
        for(size_t i = 0; i < 1064; i++){
			fileMean >> mean(i, 0);
		}
    }

	std::cout << "done." << std::endl;
}







} // end namespace















