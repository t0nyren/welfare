#include <iostream>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include "detector.h"
#include "classifier.h"
using namespace std;
using namespace cv;

class VIMG{
	public:
		int id;
		string classname;
		string filename;
		float* code;
};

float l2(float* code1, float* code2, int dim);

int main(int argc, char** argv){
	if (argc != 2){
		cout<<"Usage: bfclassify <img>"<<endl;
		return -1;
	}

	vector<VIMG> model;
	
	ifstream fin;
	fin.open("model/db.dat");
	if (fin.fail()){
		cout<<"fail to open img database"<<endl;
	}
	
	string filename;
	string classname;
		
	int id;
	//TODO: hard code vlad dimension
	int vladDimension = 22272;
	
		
	fin>>classname;
	string preClassname = classname;
	int classcount = 1;	
	while(!fin.eof()){
		if (classname != preClassname){
			cout<<classname<<endl;
			classcount ++;
		}
		fin>>filename;
		fin>>id;
		float* code = new float[vladDimension];
		for (int i = 0; i < vladDimension; i++){
			fin>>code[i];
		}
		VIMG img;
		img.id = id;
		img.filename = filename;
		img.code = code;
		img.classname = classname;
		model.push_back(img);
		fin>>classname;

	}
	cout<<"num class:"<<classcount<<" num img:"<<model.size()<<endl;

	Detector detector;
	Classifier classifier;
	Mat src = detector.detect(argv[1]);
	Mat img;
	cvtColor(src, img, CV_RGB2GRAY);
	float* code = classifier.encodeImg(img);
	classname = "";
	filename = "";
	double mindis = 10000;
	for (int i = 0; i < model.size(); i++){
		double dis = l2(model[i].code, code, vladDimension);
		if (dis < mindis){
			classname = model[i].classname;
			filename = model[i].filename;
		}
	}
	cout<<"Prediction: "<<classname<<endl;
	cout<<"File: "<<filename<<endl;
	return 0;
}

float l2(float* code1, float* code2, int dim){
	float ret = 0;	
	for (int i = 0; i < dim; i++){
		ret += pow((code1[i] - code2[i]),2);
	}
	return sqrt(ret);
}
