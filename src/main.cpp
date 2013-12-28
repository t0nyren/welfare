#include "classifier.h"
#include "detector.h"
#include <iostream>
#include <ctime>
#include <sys/time.h>
using namespace std;

int main(int argc, char** argv){
	if (argc != 2){
		cout<<"Usage: classify [query_img]"<<endl;
		return 0;
	}
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	Detector detector;
	Classifier classifier;
	classifier.setDebug(true);
	gettimeofday(&end, NULL);
	double elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	cout<<"load model: "<<elapsed<<" seconds"<<endl;
		
	gettimeofday(&begin, NULL);
	detector.setDebug(true);
	Mat src = detector.detect(argv[1]);
	
	if (src.empty()){
		cout<<"Zero or more than one face detected"<<endl;
		return 0;
	}
	Mat img;
	cvtColor(src, img, CV_RGB2GRAY);
	//imwrite( "./data/face.jpg" , img ); 
	//Mat img2 = imread("./data/face.jpg", 2|4);
	gettimeofday(&end, NULL);
	elapsed = (end.tv_sec - begin.tv_sec) + 
			  ((end.tv_usec - begin.tv_usec)/1000000.0);
	cout<<"face detection: "<<elapsed<<" seconds"<<endl;
	
	gettimeofday(&begin, NULL);
	int predict = classifier.classify(img);
	gettimeofday(&end, NULL);
	elapsed = (end.tv_sec - begin.tv_sec) + 
			  ((end.tv_usec - begin.tv_usec)/1000000.0);
	cout<<"classification: "<<elapsed<<" seconds"<<endl;
	cout<<"predicted class: "<<predict<<endl;
	return 0;
}