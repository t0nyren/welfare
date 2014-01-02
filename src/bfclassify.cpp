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

double l2(float* code1, float* code2, int dim);
double scalar(float* code1, float* code2, int dim){
	double ret = 0;
	for (int i = 0; i < dim; i++){
		ret += code1[i]*code2[i];
	}
	return ret;
}

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
	
	char filename[1024];
	char classname[1024];
		
	int id;
	//TODO: hard code vlad dimension
	const int vladDimension = 22272;
	//int vladDimension = 4;
		
	fin.getline(classname,1024);
	int preid = -1;
	int classcount = 0;	
	while(!fin.eof()){
		fin.getline(filename,1024);
		fin>>id;
		//cout<<filename<<" "<<id<<endl;
		if (id != preid){
			cout<<classname<<endl;
			classcount++;
			preid = id;	
		}
		float* code = new float[vladDimension];
		for (int i = 0; i < vladDimension; i++){
			fin>>code[i];
		//	cout<<code[i]<<" ";
		}
		//cout<<endl;
		char tmp;
		fin.get(tmp);
		fin.get(tmp);
		VIMG img;
		img.id = id;
		img.filename = filename;
		img.code = code;
		img.classname = classname;
		model.push_back(img);
		//classname = "";
		//filename = "";
		fin.getline(classname,1024);
		//cout<<classname<<endl;

	}
	cout<<"num class:"<<classcount<<" num img:"<<model.size()<<endl;

	Detector detector;
	Classifier classifier;
	Mat src = detector.detect(argv[1]);
	Mat img;
	cvtColor(src, img, CV_RGB2GRAY);
	float* code = classifier.encodeImg(img);
	ofstream fout;
	fout.open("code.txt");
	for (int i = 0; i < vladDimension; i++){
		fout<<code[i]<<" ";
	}
	cout<<endl;
	fout.close();
	string pclassname = "";
	string pfilename = "";
	double maxsim = -10000;
	for (int i = 0; i < model.size(); i++){
		double sim = scalar(model[i].code, code, vladDimension);
		cout<<model[i].classname<<" "<<sim<<endl;
		if (sim > maxsim){
			pclassname = model[i].classname.data();
			pfilename = model[i].filename.data();
			maxsim = sim;
		}
	}
	cout<<"Prediction: "<<pclassname<<endl;
	cout<<"File: "<<pfilename<<endl;
	return 0;
}

double l2(float* code1, float* code2, int dim){
	double ret = 0;	
	for (int i = 0; i < dim; i++){
		ret += pow((code1[i] - code2[i]),2);
	}
	return sqrt(ret);
}
