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
	Mat img;
	if (src.empty()){
		cout<<"Zero or more than one face detected"<<endl;
		return 0;
	}
	else if (src.channels()==3 || src.channels()==4){
			cvtColor(src, img, CV_RGB2GRAY);
	}
	else if (img.channels() != 1){
			cout<<"cvtcolor failed"<<endl;
			return 0;
	}
	else
		img = src;
	//imwrite( "./data/face.jpg" , img ); 
	//Mat img2 = imread("./data/face.jpg", 2|4);
	gettimeofday(&end, NULL);
	elapsed = (end.tv_sec - begin.tv_sec) + 
			  ((end.tv_usec - begin.tv_usec)/1000000.0);
	cout<<"face detection: "<<elapsed<<" seconds"<<endl;
	
	gettimeofday(&begin, NULL);
	int ids[5];
	int similarities[5];
	int num_predict = classifier.classify(img, 5, ids, similarities);
	for (int i = 0; i < 5; i++){
		cout<<i<<" predict: "<<ids[i]<<" similarity: "<<similarities[i]<<endl;
	}
	gettimeofday(&end, NULL);
	elapsed = (end.tv_sec - begin.tv_sec) + 
			  ((end.tv_usec - begin.tv_usec)/1000000.0);
	cout<<"classification: "<<elapsed<<" seconds"<<endl;
	cout<<"predicted class: "<<ids[0]<<endl;
	return 0;
}