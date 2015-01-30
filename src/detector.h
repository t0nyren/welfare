#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "flandmark_detector.h"

using namespace std;
using namespace cv;

class Detector{
	public:
		Detector();
		Mat detect(string imgname);
		void setDebug(bool isdebug);
	private:
		CvHaarClassifierCascade* faceCascade;
		FLANDMARK_Model* fmodel;
		Mat rotateImage(const Mat& source, double angle);
		bool debug;
};
