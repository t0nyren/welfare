#include "classifier.h"
#include <vl/generic.h>
#include <vl/dsift.h>
#include <vl/vlad.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include <opencv2/imgproc/imgproc.hpp>

#define MAX_NUM_RETURNS 5
#define MODEL_W "model/w.dat"
#define MODEL_B "model/b.dat"
#define MODEL_WORDS "model/encoder_words.dat"
#define MODEL_CENTER "model/encoder_projectionCenter.dat"
#define MODEL_PROJECT "model/encoder_projection.dat"

//default constructor, load default param
Classifier::Classifier(){
	debug = 0;
	descrDimension = 128;
	totalNumFrames = 1936;
	pcaDimension = 85;
	numWords = 256;
	maxNumFrames = 10000;
	frameDimension = 2;
	numClass = 726;
	geoDimension = pcaDimension + frameDimension;
	vladDimension = numWords * geoDimension;

	//model
	ifstream fin;
	fin.open(MODEL_W);
	if (fin.fail()){
		cout<<"fail to open w.dat"<<endl;
		exit(1);
	}
	w = new float*[numClass];
	for (int i = 0; i < numClass; i++){
		w[i] = new float[vladDimension];
		for (int j = 0; j < vladDimension; j++){
			fin>>w[i][j];
		}
	}
	fin.close();
	
	fin.open(MODEL_B);
		if (fin.fail()){
		cout<<"fail to open b.dat"<<endl;
		exit(1);
	}
	b = new float[numClass];
	for (int i = 0; i < numClass; i++){
		fin>>b[i];
	}
	fin.close();
	
	fin.open(MODEL_WORDS);
	if (fin.fail()){
		cout<<"fail to open encoder_words.dat"<<endl;
		exit(1);
	}
	encoderWords = new float[geoDimension*numWords];
	for (int i = 0; i < numWords; i++){
		for (int j = 0; j < geoDimension; j++){
			fin>>encoderWords[i*geoDimension + j];
		}
	}
	fin.close();
	
	fin.open(MODEL_CENTER);
	if (fin.fail()){
		cout<<"fail to open encoder_projectionCenter.dat"<<endl;
		exit(1);
	}
	encoderCenter = new float[descrDimension];
	for (int i = 0; i < descrDimension; i++){
		fin>>encoderCenter[i];
	}
	fin.close();
	
	fin.open(MODEL_PROJECT);
	if (fin.fail()){
		cout<<"fail to open encoder_projection.dat"<<endl;
		exit(1);
	}
	encoderProjection = new float*[pcaDimension];
	for (int i = 0; i < pcaDimension; i++){
		encoderProjection[i] = new float[descrDimension];
		for (int j = 0; j < descrDimension; j++){
			fin>>encoderProjection[i][j];
		}
	}
	fin.close();
	
	buildKDForest();
}
void Classifier::setDebug(bool isdebug){
	debug = isdebug;
}

void Classifier::buildKDForest(){
	forest = vl_kdforest_new (VL_TYPE_FLOAT, geoDimension, 2, VlDistanceL2) ;
	vl_kdforest_set_thresholding_method (forest, VL_KDTREE_MEDIAN) ;
	vl_kdforest_build (forest, numWords, encoderWords) ;
	
	//verbose
	//vl_uindex ti ;
    //for (ti = 0 ; ti < vl_kdforest_get_num_trees(forest) ; ++ ti) {
    //  printf("vl_kdforestbuild: tree %d: depth %d, num nodes %d\n",
    //           ti,
    //            vl_kdforest_get_depth_of_tree(forest, ti),
    //            vl_kdforest_get_num_nodes_of_tree(forest, ti)) ;
    //}
}

int Classifier::classify(Mat img, int num, int* ids, int* similarities){
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	pair<vector<float>, vector<float> > features = featureExtraction(img);
	gettimeofday(&end, NULL);
	double elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"Feature extraction: "<<elapsed<<" seconds"<<endl;
	gettimeofday(&begin, NULL);
	float* code = encode(features.first, features.second);
	gettimeofday(&end, NULL);
	elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"Encoding: "<<elapsed<<" seconds"<<endl;
	gettimeofday(&begin, NULL);
	int prediction = search(code, num, ids, similarities);
		gettimeofday(&end, NULL);
	elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"Searching: "<<elapsed<<" seconds"<<endl;
	delete[] code;
	return prediction;
}

float* Classifier::encodeImg(Mat img){
	pair<vector<float>, vector<float> > features = featureExtraction(img);
	float* code = encode(features.first, features.second);
	return code;
}

int Classifier::getCodeDimension(){
	return vladDimension;
}

pair<vector<float>, vector<float> > Classifier::featureExtraction(const Mat& img){
	//hard code scales
	//TODO: fix later
	//double scales[7];
	//scales[0] = 2;
	//scales[1] = 1.414213562373095;
	//scales[2] = 1;
	//scales[3] = 0.707106781186548;
	//scales[4] = 0.5;
	//scales[5] = 0.353553390593274;
	//scales[6] = 0.25;
	double scales[1];
	scales[0] = 2;
	
	pair<vector<float>, vector<float> > descriptors;
	float* tmpDescr = new float[descrDimension];


	for ( int scale = 0; scale < 1 ; scale++){
	  vector<float> descrsOut;
	  vector<float> framesOut;
	  Mat resized;
	  //hard code img size
	  //TODO: fix later
	  cout<<"feature extraction img size: "<<img.rows<<" "<<img.cols<<endl;
	  resize(img,resized,Size(100*scales[scale],100*scales[scale]),0,0,INTER_NEAREST);
	  char buf[10];
	  sprintf(buf,"%d", scale+1);
	  //imwrite("./tmp/face" + string(buf) + ".jpg", resized);
	  //resized = imread("./tmp/face" + string(buf) + ".jpg", 2|4);
	  //cout<<"Image size: "<<resized.rows<<" "<<resized.cols<<endl;
	  vector<float> data;
	  for (int i = 0; i < resized.cols; i++){
		for (int j = 0; j < resized.rows; j++){
			//printf("%f ", (float)resized.at<unsigned char>(i, j)/255);
			data.push_back((float)resized.at<unsigned char>(j, i)/255);
			
		}
		//cout<<endl;
	  }
	  //cout<<"after data"<<endl;
	  int numFrames ;
	  int descrSize ;
	  VlDsiftKeypoint const *frames ;
	  float const *descrs ;
	  int k, i ;
	  VlDsiftFilter *dsift ;
	  VlDsiftDescriptorGeometry geom ;
	  geom.numBinX = 4 ;
	  geom.numBinY = 4 ;
	  geom.numBinT = 8 ;
	  geom.binSizeX = 8 ;
	  geom.binSizeY = 8 ;
	  
	  //hard code img size to 100x100
	  //TODO: fix later
	  dsift = vl_dsift_new (100*scales[scale], 100*scales[scale]) ;
	  vl_dsift_set_geometry(dsift, &geom) ;
	  vl_dsift_set_steps(dsift, 4, 4) ;
	  vl_dsift_set_flat_window(dsift, 1) ;
	  numFrames = vl_dsift_get_keypoint_num (dsift) ;
	  descrSize = vl_dsift_get_descriptor_size (dsift) ;
	  geom = *vl_dsift_get_geometry (dsift) ;
	  //cout<<"processing dsift"<<endl;
	  vl_dsift_process (dsift, &data[0]) ;
	  frames = vl_dsift_get_keypoints (dsift) ;
	  descrs = vl_dsift_get_descriptors (dsift) ;
	  //cout<<"descriptor size: "<<descrSize<<endl;
	  //cout<<"number of frames: "<<numFrames<<endl;

	  
	  for (int k = 0; k < numFrames; ++k){
		framesOut.push_back(frames[k].y);
		framesOut.push_back(frames[k].x);
		//framesOut[k][2] = frames[k].norm;
	  }
	  //cout<<"transposing..."<<endl;
	  for (int k = 0; k < numFrames; ++k){
		vl_dsift_transpose_descriptor (tmpDescr, descrs + descrSize * k, geom.numBinT, geom.numBinX, geom.numBinY) ;
		for (i = 0 ; i < descrSize ; ++i) {
			tmpDescr[i] = VL_MIN(512.0F * tmpDescr[i], 255.0F) ;
			descrsOut.push_back(tmpDescr[i]);
			//cout<<tmpDescr[i]<<" ";
		}
		//cout<<endl;
	  }
	  
	  //cout<<"norming..."<<endl;
	  norm(descrsOut, descrSize, numFrames);
	  for (int i = 0; i < numFrames; i++){
		for (int j = 0; j < descrSize; j++){
			//cout<<descrsOut[i*descrSize + j]<<" ";
		}
		//cout<<endl;
	  }
	  //cout<<"norming frames..."<<endl;
	  for (int i = 0; i < numFrames; i++){
		for (int j = 0; j < frameDimension; j++){
			framesOut[i*frameDimension + j] = framesOut[i*frameDimension + j]/scales[scale] + 1;
			//framesOut[i][2] = 8 / scales[scale] / 3;
		}
	  }
	 
	  //cout<<"pushing back..."<<endl;
		for (int i = 0; i < numFrames; i++){
			for (int j = 0; j < descrSize; j++){
				descriptors.first.push_back(descrsOut[i*descrSize + j]);
			}
			for (int j = 0; j < frameDimension; j++){
				descriptors.second.push_back(framesOut[i*frameDimension + j]);
			}
		}
	  //cout<<"release mem"<<endl;
	  vl_dsift_delete (dsift) ;
	}
	delete[] tmpDescr;
	return descriptors;
}

float* Classifier::encode(vector<float> descrs, vector<float> frames){
	
	//cout<<"projecting"<<endl;
	struct timeval begin, end;
	gettimeofday(&begin, NULL);

	vector<float> descrps = project(descrs);
	gettimeofday(&end, NULL);
	double elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"PCA projection: "<<elapsed<<" seconds"<<endl;
	//cout<<"renorming"<<endl;
	renorm(descrps);

	//cout<<"frame norm"<<endl;
	normFrame(frames);

	gettimeofday(&begin, NULL);
	vector<float> geodescrps;
	for (int i = 0; i < totalNumFrames; i++){
		for (int j = 0; j < pcaDimension; j++){
			geodescrps.push_back(descrps[i*pcaDimension + j]);
		}
		for (int j = 0; j < frameDimension; j++){
			geodescrps.push_back(frames[i*frameDimension + j]);
		}
	}
	gettimeofday(&end, NULL);
	elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"Expanding: "<<elapsed<<" seconds"<<endl;
	
	//cout<<"query kd-tree"<<endl;
	gettimeofday(&begin, NULL);
	vl_kdforest_set_max_num_comparisons (forest, 15) ;
	vl_uint32* index = new vl_uint32[totalNumFrames];
	float* distance = new float[totalNumFrames];
	int numComparisons = vl_kdforest_query_with_array (forest, index, 1, totalNumFrames, distance, &geodescrps[0]) ;
	
	//for (int i = 0 ; i < descrps.size() ; ++i) { 
	//	index[i] ++ ; 
	//}
	//for (int i = 0; i < descrps.size(); i++)
	//cout<<index[i]<<" ";
	//cout<<endl;

	//ifstream fin;
	//fin.open("index.txt");

	//for (int i = 0; i < 3400; i++)
	//	fin>>index[i];
	// fin.close();

	//cout<<"assign"<<endl;
	vector<int> assign;

	for (int i = 0; i < totalNumFrames; i++){
		for (int j = 0; j < numWords; j++){
			if (j == index[i]){
				assign.push_back(1);
				//cout<<j<<" ";
			}
			else
				assign.push_back(0);
		}
	}
	//cout<<endl;

	float* assignments;
	assignments  =  new float[totalNumFrames * numWords];
	memset(assignments, 0, sizeof(float) * totalNumFrames * numWords);
	for(int i = 0; i < totalNumFrames; i++) {
		assignments[i * numWords + index[i]] = 1.;
	}
	//cout<<"vlad encode"<<endl;
	gettimeofday(&end, NULL);
	elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"bovw: "<<elapsed<<" seconds"<<endl;


	//for (int i = 0; i < descrps.size(); i++){
	//	for (int j = 0; j < descrps[i].size(); j++){
	//	vldata.push_back(descrps[i][j]);
	//}
	//cout<<endl;
	//}
	gettimeofday(&begin, NULL);
	int vlad_flags = 0;
	vlad_flags |= VL_VLAD_FLAG_SQUARE_ROOT ;
	vlad_flags |= VL_VLAD_FLAG_NORMALIZE_COMPONENTS;
	//cout<<"normalized components: "<<VL_YESNO(vlad_flags & VL_VLAD_FLAG_NORMALIZE_COMPONENTS)<<endl;
	//cout<<"square root: "<<VL_YESNO(vlad_flags & VL_VLAD_FLAG_SQUARE_ROOT)<<endl;
	//cout<<"unnormalized: "<<VL_YESNO(vlad_flags & VL_VLAD_FLAG_UNNORMALIZED)<<endl;


	float* vlad = new float[numWords*geoDimension];
	vl_vlad_encode(vlad, VL_TYPE_FLOAT, encoderWords, geoDimension, numWords, &geodescrps[0], totalNumFrames, assignments, vlad_flags);
	gettimeofday(&end, NULL);
	elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"vlad: "<<elapsed<<" seconds"<<endl;
	
	delete[] distance;
	delete[] assignments;
	delete[] index;
	
	return vlad;
}

int Classifier::search(float* code, int num, int* ids, int* similarities){
	//cout<<"category scores:"<<endl;
	vector<float> scores;
	if (num > MAX_NUM_RETURNS)
		num = MAX_NUM_RETURNS;
	/*
	float max = -1000;
	float max2 = -1000;
	float max3 = -1000;
	int max_index = -1;
	int max_index2 = -1;
	int max_index3 = -1;*/
	float max[MAX_NUM_RETURNS];
	for (int i = 0; i < num; i++){
		max[i] = -1000;
		ids[i] = -1;
	}
	for (int i = 0; i < numClass; i++){
		float score = 0;
		for (int j = 0; j < vladDimension; j++){
			score += code[j]*w[i][j];
		}
		score += b[i];
		scores.push_back(score);
		/*
		if (scores[i] > max){
			max2 = max;
			max_index2 = max_index;
			
			max = scores[i];
			max_index = i;
		}
		else if (scores[i] > max2){
			max3 = max2;
			max_index3 = max_index2;
		
			max2 = scores[i];
			max_index2 = i;
		}
		else if (scores[i] > max3){
			max3 = scores[i];
			max_index3 = i;
		}*/
		//cout<<i<<" "<<score<<endl;
		
	}
	for (int i = 0; i < num; i++){
		for (int j = 0; j < numClass; j++){
			if (scores[j] > max[i]){
				bool add = 1;
				for (int k = 0; k < i; k++){
					if (ids[k] == j){
						add = 0;
					}
				}
				if (add){
					max[i] = scores[j];
					ids[i] = j;
				}
			}
		}
	}
	
	if (debug){
		for (int i = 0; i < num; i++){
			cout<<i<<"th: "<<ids[i]<<" "<<max[i]<<endl;
		}
	}
	int ret = num;
	srand (time(NULL));
	float maxSim = -0.65;
	float minSim = -0.85;
	if (max[0] > -0.5)
		maxSim = max[0];
	for (int i = 0; i < num; i++){
		max[i] = 1 - (maxSim - max[i])/(maxSim - minSim);
		if (max[i] >= 0.95)
			similarities[i] = 95;
		else if (max[i] <= 0.05)
			similarities[i] = 5;
		else
			similarities[i] = max[i]*100;
	
		//no similar class, return one of the defaults
		if (max[i] < -0.85){
			if (ret ==num)
				ret = i;
			int ran =  rand() % 5;
			switch(ran){
			case 0:
				ids[i] =  0;
				break;
			case 1:
				ids[i] = 1;
				break;
			case 2:
				ids[i] = 2;
				break;
			case 3:
				ids[i] = 3;
				break;
			default:
				ids[i] = 283;
			}
		}
	}
	/*
	if (max > -0.85)
		return max_index;
	else
		return 259;
	//cout<<"classify: "<<max_index<<endl;
	return max_index;*/
	return ret;
}

void Classifier::norm(vector<float>& src, int descrSize, int numFrames){
	//ugly stack memory
	//TODO: fix it later
	float val[10000];
	vector<float> tmp;
	for (int i = 0; i < numFrames; i++){
		val[i] = 0;
	}
	
	for (int i = 0; i < numFrames; i++){
		for (int j = 0; j < descrSize; j++){
			tmp.push_back(pow(src[i*descrSize + j],2));
			val[i] += tmp[i*descrSize + j];
		}
	}
	
	for (int i = 0; i < numFrames; i++){
		val[i] = (1e-5 > sqrt(val[i])) ? 1e-5 : sqrt(val[i]);
		val[i] = 1/val[i];
		//cout<<val[i]<<" ";
	}
	//cout<<endl;
	
	for (int i = 0; i < numFrames; i++){
		for (int j = 0; j < descrSize; j++){
			src[i*descrSize + j] = src[i*descrSize + j] * val[i];
		}
	}
}

void Classifier::renorm(vector<float>& descrps){
	//ugly stack memory
	//TODO: fix it later
	float val[10000];
	vector<float> tmp;
	for (int i = 0; i < totalNumFrames; i++){
		val[i] = 0;
	}
	
	for (int i = 0; i < totalNumFrames; i++){
		for (int j = 0; j < pcaDimension; j++){
			tmp.push_back(pow(descrps[i*pcaDimension + j],2));
			val[i] += tmp[i*pcaDimension + j];
		}
	}
	for (int i = 0; i <totalNumFrames; i++){
		val[i] = (1e-10 > sqrt(val[i])) ? 1e-10 : sqrt(val[i]);
		val[i] = 1/val[i];
	}
	
	for (int i = 0; i < totalNumFrames; i++){
		for (int j = 0; j < pcaDimension; j++){
			descrps[i*pcaDimension + j] = descrps[i*pcaDimension + j] * val[i];
		}
	}
}

void Classifier::normFrame(vector<float>& frames){
	for (int i = 0; i < totalNumFrames; i++){
		for (int j = 0; j < frameDimension; j++){
			frames[i*frameDimension + j] = (frames[i*frameDimension + j] - 50)*0.01;
		}
	}
}

vector<float> Classifier::project(vector<float>& descrs){
	vector<float> ret;
	for (int i = 0; i < totalNumFrames; i++){
		for (int j = 0; j < descrDimension; j++){
			descrs[i*descrDimension + j] -= encoderCenter[j];
			//cout<<descrs[i][j]<<" ";
		}
		//cout<<endl;
	}
	for (int col = 0; col < totalNumFrames; col++) {
		vector<float> tmp;
		for (int row = 0; row < pcaDimension; row++) {
			float product = 0;
			for (int inner = 0; inner < descrDimension; inner++) {
				product += encoderProjection[row][inner] * descrs[col*descrDimension + inner];
			}
			ret.push_back(product);
		}
	}
	return ret;
}
