#include <iostream>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <opencv2/flann/flann.hpp>
#include <ctime>
#include <sys/time.h>
#include "detector.h"
#include "classifier.h"
using namespace std;
using namespace cv;

class VIMG{
	public:
		int id;
		string classname;
		string filename;
};


class VladIndex{
	public:
		VladIndex();
		string predict(const char* path, bool isDetect);

	private:
		Classifier* classifier;
		Detector* detector;
		int vladDimension;
		vector<float> codes;
		//cvflann::Matrix<float>* dataset;
		cvflann::Index<cvflann::L2<float> >* index;
		vector<VIMG> imgs;
};