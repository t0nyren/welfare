#include "vladIndex.h"

VladIndex::VladIndex(){
	cout<<"loading model..."<<endl;
	//TODO: hard code vlad dimension
	vladDimension = 22272;
	//init classifier and detector
	classifier = new Classifier();
	detector = new Detector();
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	//vector<float> codes;
	ifstream fin;
	fin.open("model/db.dat");
	if (fin.fail()){
		cout<<"fail to open img database"<<endl;
	}
	
	char filename[1024];
	char classname[1024];
	int id;
	fin.getline(classname,1024);
	int preid = -1;
	int classcount = 0;	
	while(!fin.eof()){
		fin.getline(filename,1024);
		fin>>id;
		//cout<<filename<<" "<<id<<endl;
		if (id != preid){
			cout<<classname<<" "<<classcount<<endl;
			classcount++;
			preid = id;	
		}
		float buf;
		for (int i = 0; i < vladDimension; i++){
			fin>>buf;
			codes.push_back(buf);
		}
		//cout<<endl;
		char tmp;
		fin.get(tmp);
		fin.get(tmp);
		VIMG img;
		img.id = id;
		img.filename = filename;
		img.classname = classname;
		imgs.push_back(img);
		//classname = "";
		//filename = "";
		fin.getline(classname,1024);
		//cout<<classname<<endl;

	}
	gettimeofday(&end, NULL);
	double elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
    cout<<"model loaded in "<<elapsed<<" seconds"<<endl;
	cout<<"num class:"<<classcount<<" num img:"<<imgs.size()<<endl;
	//build index
	gettimeofday(&begin, NULL);
	//dataset = new cvflann::Matrix<float>(codes.data(), imgs.size(), vladDimension);
	cvflann::Matrix<float> dataset(codes.data(), imgs.size(), vladDimension);
	index = new cvflann::Index<cvflann::L2<float> >(dataset, cvflann::KDTreeIndexParams(10));
	index->buildIndex();
	gettimeofday(&end, NULL);
	elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
    cout<<"build index: "<<elapsed<<" seconds"<<endl;
}


string VladIndex::predict(const char* path, bool isDetect){
	int nn = 3;
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	Mat src;
	if (isDetect){
		src = detector->detect(path);
	}
	else{
		Mat src1 = imread(path, CV_LOAD_IMAGE_COLOR);
		resize(src1, src, Size(100, 100));
	}
	
	Mat img;
	if (src.empty()){
		cout<<"fail to detect"<<endl;
		return "";
	}
	else if (src.channels()==3 || src.channels()==4){
		cvtColor(src, img, CV_RGB2GRAY);
	}
	else if (img.channels() != 1){
		cout<<"cvtcolor failed"<<endl;
		return "";
	}
	else
		img = src;
	
	float* code = classifier->encodeImg(img);
	//ofstream fout;
	//fout.open("code.txt");
	//for (int i = 0; i < vladDimension; i++){
	//	fout<<code[i]<<" ";
	//}
	//fout<<endl;
	//fout.close();
	
	gettimeofday(&end, NULL);
	double elapsed = (end.tv_sec - begin.tv_sec) + 
			  ((end.tv_usec - begin.tv_usec)/1000000.0);
	cout<<"encode face: "<<elapsed<<" seconds"<<endl;

	
	
	gettimeofday(&begin, NULL);
	int* indices_array = new int[nn];
	float* dists_array = new float[nn];
	cvflann::Matrix<float> query(code, 1, vladDimension);
	cvflann::Matrix<int> indices(indices_array, 1, nn);
	cvflann::Matrix<float> dists(dists_array, 1, nn);
	index->knnSearch(query, indices, dists, nn, cvflann::SearchParams(128));
	gettimeofday(&end, NULL);
	elapsed = (end.tv_sec - begin.tv_sec) + 
			  ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"query: "<<elapsed<<" seconds"<<endl;	
	for (int i = 0; i < nn; i++){
		cout<<"Prediction: "<<imgs[indices[0][i]].classname<<endl;
		cout<<"File: "<<imgs[indices[0][i]].filename<<endl;
		cout<<"Distance: "<<dists[0][i]<<endl;
	}
	string ret = imgs[indices[0][1]].classname;
	delete[] code;
	delete[] indices_array;
	delete[] dists_array;
	return ret;
}
