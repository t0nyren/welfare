#include <vl/generic.h>
#include <vl/dsift.h>
#include <vl/kdtree.h>
#include <vl/vlad.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include "detector.h"

using namespace std;


void norm(float** src, float** ret, int descrSize, int numFrames){
	float val[10000];
	for (int i = 0; i < numFrames; i++){
		val[i] = 0;
	}
	
	for (int i = 0; i < numFrames; i++){
		for (int j = 0; j < descrSize; j++){
			ret[i][j] = pow(src[i][j],2);
			val[i] += ret[i][j];
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
			ret[i][j] = src[i][j] * val[i];
		}
	}
}

void renorm(vector<vector<float> >& descrps){
	float val[10000];
	float tmp[10000][80];
	for (int i = 0; i < descrps.size(); i++){
		val[i] = 0;
	}
	
	for (int i = 0; i < descrps.size(); i++){
		for (int j = 0; j < 80; j++){
			tmp[i][j] = pow(descrps[i][j],2);
			val[i] += tmp[i][j];
		}
	}
	
	for (int i = 0; i < descrps.size(); i++){
		val[i] = (1e-10 > sqrt(val[i])) ? 1e-10 : sqrt(val[i]);
		val[i] = 1/val[i];
		//cout<<val[i]<<" ";
	}
	//cout<<endl;
	
	for (int i = 0; i < descrps.size(); i++){
		for (int j = 0; j < 80; j++){
			descrps[i][j] = descrps[i][j] * val[i];
		}
	}
}

void normFrame(vector<vector<float> >& frames){
	for (int i = 0; i < frames.size(); i++){
		frames[i][0] = (frames[i][0] - 50)*0.01;
		frames[i][1] = (frames[i][1] - 50)*0.01;
	}
}

VlKDForest* buildKDTree(const vector<float>& words, int descrSize, int numWords){
	VlKDForest * forest ;
	forest = vl_kdforest_new (VL_TYPE_FLOAT, descrSize, 2, VlDistanceL2) ;
	vl_kdforest_set_thresholding_method (forest, VL_KDTREE_MEDIAN) ;
	vl_kdforest_build (forest, numWords, &words[0]) ;
	
	vl_uindex ti ;
    for (ti = 0 ; ti < vl_kdforest_get_num_trees(forest) ; ++ ti) {
      printf("vl_kdforestbuild: tree %d: depth %d, num nodes %d\n",
                ti,
                vl_kdforest_get_depth_of_tree(forest, ti),
                vl_kdforest_get_num_nodes_of_tree(forest, ti)) ;
    }
	

	return forest;
}

vector<float> loadWords(int descrSize, int numWords){
	ifstream fin;
	fin.open("./model/encoder_words.dat");
	vector<float> data;
	float voc[82][256];
	for (int i = 0; i < descrSize; i++){
		for (int j = 0; j < numWords; j++){
			float f;
			fin>>voc[i][j];
		}
	}
	for (int j = 0; j < numWords; j++){
		for (int i = 0; i < descrSize; i++){
			data.push_back(voc[i][j]);
		}
	}
	fin.close();
	return data;
}

vector<vector<float> > project(vector<vector<float> > descrs){
	vector<vector<float> > ret;
	ifstream finc, finp;
	float center[128];
	float projection[80][128];
	finc.open("./model/encoder_projectionCenter.dat");
	finp.open("./model/encoder_projection.dat");
	if (finc.fail() || finp.fail()){
		cout<<"fail to open"<<endl;
		exit(1);
	}
	for (int i = 0; i < 128; i++){
		finc>>center[i];
	}
	
	for (int i = 0; i < 80; i++){
		for (int j = 0; j < 128; j++){
			finp>>projection[i][j];
			//cout<<projection[i][j]<<" ";
		}
		//cout<<endl;
	}
	//cout<<endl;cout<<endl;cout<<endl;
	for (int i = 0; i < descrs.size(); i++){
		for (int j = 0; j < descrs[i].size(); j++){
			descrs[i][j] -= center[j];
			//cout<<descrs[i][j]<<" ";
		}
		//cout<<endl;
	}
	
	
		
	for (int col = 0; col < descrs.size(); col++) {
		vector<float> tmp;
		for (int row = 0; row < 80; row++) {
			float product = 0;
			for (int inner = 0; inner < 128; inner++) {
				product += projection[row][inner] * descrs[col][inner];
			}
			tmp.push_back(product);
		}
		ret.push_back(tmp);
	}
	finc.close();
	finp.close();
	return ret;
}

int main (int argc, const char * argv[]) {
  VL_PRINT ("Hello world!\n") ;
  Detector detector("./model/haarcascade_frontalface_alt.xml","./model/flandmark_model.dat");
  Mat src = detector.detect(argv[1]);
  Mat img;
  cvtColor(src, img, CV_RGB2GRAY);
  cout<<"Image size: "<<img.rows<<" "<<img.cols<<endl;
  imwrite( "./data/face.jpg" , img ); 
  Mat img2 = imread("./data/face.jpg", 2|4);  
  
  //float const *data = (float*)img.data;
  
  
  
 // for (int i = 0; i < img2.rows; i++){
//	for (int j = 0; j < img2.cols; j++){
//		printf("%d ", img2.at<unsigned char>(i, j));
//	}
//	cout<<endl;
 // }
  cout<<endl;cout<<endl;
  
  double scales[7];
  scales[0] = 2;
  scales[1] = 1.414213562373095;
  scales[2] = 1;
  scales[3] = 0.707106781186548;
  scales[4] = 0.5;
  scales[5] = 0.353553390593274;
  scales[6] = 0.25;

  vector<vector<float> > scaledDescrs;
  vector<vector<float> > scaledFrames;
  
  float* tmpDescr = new float[128];
  float** descrsOut = new float*[10000];
  float** framesOut = new float*[10000];
  float** ret = new float*[10000];
  for (int i = 0; i < 10000; i++){
	ret[i] = new float[128];
  }
  for (int i = 0; i < 10000; i ++){
	descrsOut[i] = new float[128];
  }
  for (int i = 0; i < 10000; i++){
	framesOut[i] = new float[2];
  }
  for ( int scale = 0; scale < 7 ; scale++){

	  Mat resized;
	  resize(img2,resized,Size(100*scales[scale],100*scales[scale]),0,0,INTER_NEAREST);
	  char buf[10];
	  sprintf(buf,"%d", scale+1);
	  imwrite("./data/face" + string(buf) + ".jpg", resized);
	  resized = imread("./data/face" + string(buf) + ".jpg", 2|4);
	  cout<<"Image size: "<<resized.rows<<" "<<resized.cols<<endl;
	  vector<float> data;
	  for (int i = 0; i < resized.cols; i++){
		for (int j = 0; j < resized.rows; j++){
			//printf("%f ", (float)resized.at<unsigned char>(i, j)/255);
			data.push_back((float)resized.at<unsigned char>(j, i)/255);
			
		}
		//cout<<endl;
	  }
	  cout<<"after data"<<endl;
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
	  dsift = vl_dsift_new (100*scales[scale], 100*scales[scale]) ;
	  vl_dsift_set_geometry(dsift, &geom) ;
	  vl_dsift_set_steps(dsift, 4, 4) ;
	  vl_dsift_set_flat_window(dsift, 1) ;
	  numFrames = vl_dsift_get_keypoint_num (dsift) ;
	  descrSize = vl_dsift_get_descriptor_size (dsift) ;
	  geom = *vl_dsift_get_geometry (dsift) ;
	  cout<<"processing dsift"<<endl;
	  vl_dsift_process (dsift, &data[0]) ;
	  frames = vl_dsift_get_keypoints (dsift) ;
	  descrs = vl_dsift_get_descriptors (dsift) ;
	  cout<<"descriptor size: "<<descrSize<<endl;
	  cout<<"number of frames: "<<numFrames<<endl;

	  
	  for (int k = 0; k < numFrames; ++k){
		framesOut[k][0] = frames[k].y;
		framesOut[k][1] = frames[k].x;
		//framesOut[k][2] = frames[k].norm;
	  }
	  cout<<"transposing..."<<endl;
	  for (int k = 0; k < numFrames; ++k){
		vl_dsift_transpose_descriptor (tmpDescr, descrs + descrSize * k, geom.numBinT, geom.numBinX, geom.numBinY) ;
		for (i = 0 ; i < descrSize ; ++i) {
			tmpDescr[i] = VL_MIN(512.0F * tmpDescr[i], 255.0F) ;
			descrsOut[k][i] = tmpDescr[i];
			//cout<<tmpDescr[i]<<" ";
		}
		//cout<<endl;
	  }
	  
	  cout<<"norming..."<<endl;
	  norm(descrsOut, ret, descrSize, numFrames);
	  for (int i = 0; i < numFrames; i++){
		for (int j = 0; j < descrSize; j++){
			//cout<<ret[i][j]<<" ";
		}
		//cout<<endl;
	  }
	  cout<<"norming frames..."<<endl;
	  for (int i = 0; i < numFrames; i++){
		framesOut[i][0] = framesOut[i][0]/scales[scale] + 1;
		framesOut[i][1] = framesOut[i][1]/scales[scale] + 1;
		//framesOut[i][2] = 8 / scales[scale] / 3;
	  }
	 
	  cout<<"pushing back..."<<endl;

	  for (int i = 0; i < numFrames; i++){
	  	vector<float> vecDescrs;
	    vector<float> vecFrames;
	    for (int j = 0; j < descrSize; j++){
			vecDescrs.push_back(ret[i][j]);
		}
		for (int j = 0; j < 3; j++){
			vecFrames.push_back(framesOut[i][j]);
		}
		scaledDescrs.push_back(vecDescrs);
		scaledFrames.push_back(vecFrames);
	  }
	  cout<<"release mem"<<endl;
	  vl_dsift_delete (dsift) ;
	 // for (int i = 0; i < numFrames; i++)
	//	delete[] descrsOut[i];
	 // delete[] descrsOut;
	 // descrsOut = NULL;
  }
  
  cout<<"frames: "<<scaledFrames.size()<<endl;
  
  
  for (int i = 0; i < scaledFrames.size(); i++){
	for (int j = 0; j < 128; j++){
		//cout<<scaledDescrs[i][j]<<" ";
	}
	//cout<<endl;
  }
  for (int i = 0; i < scaledFrames.size(); i++){
	for (int j = 0; j < 2; j++){
		//cout<<scaledFrames[i][j]<<" ";
	}
	//cout<<endl;
  }
  
  cout<<"projecting"<<endl;
  vector<vector<float> > descrps = project(scaledDescrs);

  
  cout<<"renorming"<<endl;
  renorm(descrps);
  
  for (int i = 0; i < descrps.size(); i++){
	for (int j = 0; j < descrps[i].size(); j++){
		//cout<<descrps[i][j]<<" ";
	}
	//cout<<endl;
  }
  
  cout<<"frame norm"<<endl;
  normFrame(scaledFrames);
  for (int i = 0; i < scaledFrames.size(); i++){
	for (int j = 0; j < 2; j++){
		//cout<<scaledFrames[i][j]<<" ";
	}
	//cout<<endl;
  }
  for (int i = 0; i < descrps.size(); i++){
	descrps[i].push_back(scaledFrames[i][0]);
	descrps[i].push_back(scaledFrames[i][1]);
  }
  
  vector<float> kdquery;
  for (int i = 0; i < descrps.size(); i++){
	for (int j = 0; j < descrps[i].size(); j++){
		kdquery.push_back(descrps[i][j]);
	}
  }
  cout<<"load words"<<endl;
  vector<float> words = loadWords(82, 256);
  for (int i = 0; i < 256; i++){
	for (int j = 0; j < 82; j++){
		//cout<<words[i*82+j]<<" ";
	}
	//cout<<endl;
  }
  
  cout<<"build kd-tree"<<endl;
  VlKDForest* forest = buildKDTree(words, 82, 256);
  cout<<"query kd-tree"<<endl;
  vl_kdforest_set_max_num_comparisons (forest, 15) ;
  vl_uint32* index = new vl_uint32[descrps.size()];
  float* distance = new float[descrps.size()];
  int numComparisons = vl_kdforest_query_with_array (forest, index, 1, descrps.size(), distance, &kdquery[0]) ;
  for (int i = 0 ; i < descrps.size() ; ++i) { 
	index[i] ++ ; 
  }
  //for (int i = 0; i < descrps.size(); i++)
	//cout<<index[i]<<" ";
  //cout<<endl;
  
  //ifstream fin;
  //fin.open("index.txt");
  
  //for (int i = 0; i < 3400; i++)
//	fin>>index[i];
 // fin.close();
  
  cout<<"assign"<<endl;
  vector<int> assign;
 
  for (int i = 0; i < 3400; i++){
	for (int j = 0; j < 256; j++){
		if (j == index[i]-1){
			assign.push_back(1);
			cout<<j<<" ";
		}
		else
			assign.push_back(0);
	}
  }
  cout<<endl;
  
  int dimension = 82;
  int numData = 3400;
  int numClusters = 256;
  float* assignments;
  assignments  =  new float[numData * numClusters];
  memset(assignments, 0, sizeof(float) * numData * numClusters);
  for(int i = 0; i < numData; i++) {
	assignments[i * numClusters + index[i]-1] = 1.;
  }
  cout<<"vlad encode"<<endl;
  vector<float> vldata;
  
	
  for (int i = 0; i < descrps.size(); i++){
	for (int j = 0; j < descrps[i].size(); j++){
		vldata.push_back(descrps[i][j]);
	}
	//cout<<endl;
  }
  int vlad_flags = 0;
  vlad_flags |= VL_VLAD_FLAG_SQUARE_ROOT ;
  vlad_flags |= VL_VLAD_FLAG_NORMALIZE_COMPONENTS;
  cout<<"normalized components: "<<VL_YESNO(vlad_flags & VL_VLAD_FLAG_NORMALIZE_COMPONENTS)<<endl;
  cout<<"square root: "<<VL_YESNO(vlad_flags & VL_VLAD_FLAG_SQUARE_ROOT)<<endl;
  cout<<"unnormalized: "<<VL_YESNO(vlad_flags & VL_VLAD_FLAG_UNNORMALIZED)<<endl;
  

  float* vlad = new float[numClusters*dimension];
  vl_vlad_encode(vlad, VL_TYPE_FLOAT, &words[0], dimension, numClusters, &vldata[0], numData, assignments, vlad_flags);
  for (int i = 0 ; i < numClusters*dimension; i++){
	//cout<<vlad[i]<<endl;
  }
  
  //cout<<"homkermap"<<endl;
  //VlHomogeneousKernelType kernelType = VlHomogeneousKernelChi2 ;
  //VlHomogeneousKernelMapWindowType windowType = VlHomogeneousKernelMapWindowRectangular ;
  //double gamma = 1;
  //int period = 1;
  //VlHomogeneousKernelMap * map = vl_homogeneouskernelmap_new (kernelType, 1, n, 1, windowType) ;
  
  //load w b
  cout<<"load w and b"<<endl;
  vector<vector<float > > w;
  vector<float> b;
  
  ifstream finw, finb;
  finw.open("model/w.dat");
  finb.open("model/b.dat");
  
  for (int i = 0;  i < 20; i++){
	vector<float> tmpvec;
	float tmp;
	for (int j = 0; j < 20992; j++){
		finw>>tmp;
		tmpvec.push_back(tmp);
		//cout<<tmp<<" ";
	}
	//cout<<endl;
	w.push_back(tmpvec);
	finb>>tmp;
	b.push_back(tmp);
  }
  
  for (int i = 0; i < 20; i++){
	//cout<<b[i]<<" ";
  }
  //cout<<endl;
  
  cout<<"category scores:"<<endl;
  vector<float> scores;
  float max = -1000;
  int max_index = -1;
  for (int i = 0; i < 20; i++){
	float score = 0;
	for (int j = 0; j < 20992; j++){
		score += vlad[j]*w[i][j];
	}
	score += b[i];
	if (score > max){
		max = score;
		max_index = i;
	}
	cout<<i<<" "<<score<<endl;
	scores.push_back(score);
  }
  cout<<"classify: "<<max_index<<endl;
  return 0;
}
