#ifndef MYSLAM_DEEPLCD_H
#define MYSLAM_DEEPLCD_H

#include "caffe/caffe.hpp"

#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Dense> 
#include <Eigen/Geometry>  

#include <list>
#include <vector>
#include <iostream>


namespace deeplcd
{
typedef Eigen::Matrix<float, 128, 1> DescrVector;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatDynamic;


// --------------------------------------------------------------------------------------

class Database{
public:
	std::vector<unsigned int> mvImageIds;
	std::vector<DescrVector>  mvDescriptorVectors;
	std::vector<std::vector<float> > mvvScores;
};

// ---------------------------------------------------------------------------------------

struct query_result{
	unsigned int id;
	float score;

	query_result(unsigned int id_, float score_) : id(id_), score(score_) {}

	query_result() // Only needed for vector construction in QueryResults
	{
		query_result(0, -1.0);
	}
	
	friend std::ostream& operator << (std::ostream& stream, const query_result& q)
	{
		stream << "query_result: ID=" << q.id << ", score=" << q.score;
		return stream;
	}
};

// ------------------------------------------------------------------------------------------------

class QueryResults : public std::vector<query_result>{
public:
	QueryResults(int sz) : std::vector<query_result>(sz) {}
	QueryResults() : std::vector<query_result>(1) {}
	void insertInPlace(const query_result &q); 
	friend std::ostream& operator << (std::ostream& stream, const QueryResults& Q)
	{
		stream << "QueryResults: {";
		for (query_result q : Q)
			stream << "\n\t" << q;
		stream << "\n}\n";
		return stream;
	}
	void invalidate() 
	{
		for (size_t i = 0; i < size(); i++)
			at(i) = query_result(0, -1.0); // This is needed so old results wont interfere with new ones
	}
};


// -------------------------------------------------------------------------------------------------------------

// Deep Loop Closure Detector
class DeepLCD{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	typedef std::shared_ptr<DeepLCD> Ptr;

	// If gpu_id is -1, the cpu will be used
	DeepLCD(const std::string& network_definition_file="calc_model/deploy.prototxt", 
		const std::string& pre_trained_model_file="calc_model/calc.caffemodel", 
		const std::string &pca_params_dir = "PCAparams", int gpu_id=0);
	
	~DeepLCD(){
		delete autoencoder;
	}

	void SetParameters(int ds_ = 0,
		int v_max = 1.0, int v_interval = 0.5, int n_max_result = 20);
	
	void Add(const cv::Mat &img, unsigned int imageId);
	void Add(DescrVector descr_, unsigned int imageId);
	void AddAfterQuery(unsigned int imageId);  // this could only be done after a query

	QueryResults Query(const cv::Mat &img, int nOutWindow=0, int numResults=1);
	QueryResults Query(DescrVector descr_, int nOutWindow=0, int numResults=1);

	const float Score(const DescrVector& d1, const DescrVector& d2);

	DescrVector CalcDescrOriginalImg(const cv::Mat& originalImg);
	DescrVector CalcDescr(const cv::Mat& im); // make a forward pass through the net, return the descriptor


private:
	void LoadPCAparams(const std::string &strPathToDir,
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &transformMatrix, 
		Eigen::Matrix<float, 1064, 1> &mean);


public:
	


private:
	caffe::Net<float>* autoencoder; // the deploy autoencoder
	caffe::Blob<float>* autoencoder_input; // The encoder's input blob
	caffe::Blob<float>* autoencoder_output; // The encoder's input blob

	Database _mDatabase;

	DescrVector _mCurrentDescrVector;
	std::vector<float> _mvCurrentScores;

	bool _bHasQuery = false;

	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>  _PCAtransformMatrix
		= Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(128, 1064);
	Eigen::Matrix<float, 1064, 1> _PCAmean;

	int ds = 0;
	int numMaxResult = 20;
	float V_max = 1.0;
	float V_interval = 0.5;
	std::vector<float> _mvVelocity;

};


} // end namespace 

#endif
