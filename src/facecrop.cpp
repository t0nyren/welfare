/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Michal Uricar
 * Copyright (C) 2012 Michal Uricar
 */
#define PI 3.14159265
#include "opencv/cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cvaux.h"
#include "opencv/highgui.h"

#include <cstring>
#include <cmath>
#include <stdio.h>
#include <fstream>
#include <dirent.h>
#include<sys/stat.h>
#include<sys/types.h>
#include <string>
#include "flandmark_detector.h"

using namespace cv;
using namespace std;

Mat rotateImage(const Mat& source, double angle)
{
    Point2f src_center(source.cols/2.0F, source.rows/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    Mat dst;
    warpAffine(source, dst, rot_mat, source.size());
    return dst;
}

bool detectFaceInImage(IplImage *orig, IplImage* input, CvHaarClassifierCascade* cascade, FLANDMARK_Model *model, int *bbox, double *landmarks, string imgname, string dir_path, string fail_path, string fp_path)
{
    // Smallest face size.
    CvSize minFeatureSize = cvSize(40, 40);
    int flags =  CV_HAAR_DO_CANNY_PRUNING;
    // How detailed should the search be.
    float search_scale_factor = 1.1f;
    CvMemStorage* storage;
    CvSeq* rects;
    int nFaces;

    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);

    if (imgname.find(".jpg") == std::string::npos) {
               imgname = imgname + ".jpg";
    }
    // Detect all the faces in the greyscale image.
    rects = cvHaarDetectObjects(input, cascade, storage, search_scale_factor, 2, flags, minFeatureSize);
    nFaces = rects->total;
	if (nFaces == 1){
		//printf("img: %s\n", imgname.data());
		string img_path = dir_path + '/' + imgname;
		double t = (double)cvGetTickCount();
		Mat mat_orig(orig, 0);
		for (int iface = 0; iface < (rects ? nFaces : 0); ++iface)
		{
			CvRect *r = (CvRect*)cvGetSeqElem(rects, iface);
			
			bbox[0] = r->x;
			bbox[1] = r->y;
			bbox[2] = r->x + r->width;
			bbox[3] = r->y + r->height;
			
			flandmark_detect(input, bbox, model, landmarks);

			// display landmarks
			/*cvRectangle(orig, cvPoint(bbox[0], bbox[1]), cvPoint(bbox[2], bbox[3]), CV_RGB(255,0,0) );
			cvRectangle(orig, cvPoint(model->bb[0], model->bb[1]), cvPoint(model->bb[2], model->bb[3]), CV_RGB(0,0,255) );
			cvCircle(orig, cvPoint((int)landmarks[0], (int)landmarks[1]), 3, CV_RGB(0,0,255), CV_FILLED);
			
			cvCircle(orig, cvPoint(int(landmarks[2]), int(landmarks[3])), 3, CV_RGB(255,0,0), CV_FILLED);
			cvCircle(orig, cvPoint(int(landmarks[4]), int(landmarks[5])), 3, CV_RGB(255,0,0), CV_FILLED);
			cvCircle(orig, cvPoint(int(landmarks[6]), int(landmarks[7])), 3, CV_RGB(255,0,0), CV_FILLED);

			cvCircle(orig, cvPoint(int(landmarks[8]), int(landmarks[9])), 3, CV_RGB(255,0,0), CV_FILLED);
			cvCircle(orig, cvPoint(int(landmarks[10]), int(landmarks[11])), 3, CV_RGB(255,0,0), CV_FILLED);
			cvCircle(orig, cvPoint(int(landmarks[12]), int(landmarks[13])), 3, CV_RGB(255,0,0), CV_FILLED);
			cvCircle(orig, cvPoint(int(landmarks[14]), int(landmarks[15])), 3, CV_RGB(255,0,0), CV_FILLED);
			*/
			double angle[3];
			angle[0] = atan((landmarks[7]-landmarks[9])/(landmarks[6]-landmarks[8]));
			angle[1] = atan((landmarks[11]-landmarks[13])/(landmarks[10]-landmarks[12]));
			angle[2] = atan((landmarks[3]-landmarks[5])/(landmarks[2]-landmarks[4]));
			cout<<angle[0]*180<<" "<<angle[1]*180<<" "<<angle[2]*180<<endl;
			double angle_rotate = 0;
			if (angle[0] > angle[1]){
				if (angle[1] > angle[2])
					angle_rotate = angle[1];
				else if (angle[0] > angle[2])
					angle_rotate = angle[2];
				else
					angle_rotate = angle[0];
			}

			else{
				if (angle[1] < angle[2])
					angle_rotate = angle[1];
				else if (angle[2] < angle[0])
					angle_rotate = angle[0];
				else
					angle_rotate = angle[2];

			}
			cout<<"angle: "<<angle_rotate*180/PI<<endl;
			//printf("box: %d, %d, %d, %d\n",r->x,r->y,model->bb[2], model->bb[3]);
			Rect faceRect(r->x, r->y,r->width, r->height);
			if (faceRect.height < 100 && faceRect.width < 100){
				cout<<"small face"<<endl;
				continue;
			}
			Mat croppedFaceImage = mat_orig(faceRect).clone();
			Mat rotated = rotateImage(croppedFaceImage, angle_rotate * 180 / PI);
			Mat resized;
			resize(rotated, resized, Size(100, 100));
			cout<<img_path<<endl;
			imwrite( img_path, resized );
			cvReleaseMemStorage(&storage);
			return true;
			
			//for (int i = 2; i < 2*model->data.options.M; i += 2)
			//{
			//    cvCircle(orig, cvPoint(int(landmarks[i]), int(landmarks[i+1])), 3, CV_RGB(255,0,0), CV_FILLED);

		   // }
		    t = (double)cvGetTickCount() - t;
			int ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
		}
	}
	else if (nFaces == 0){
		string fail_file = fail_path + '/' + imgname;
		cout<<fail_file.data()<<endl;
		cvSaveImage(fail_file.data(), orig);
	}
	else if (nFaces > 1){
		string fp_file = fp_path + '/' + imgname;
		cout<<fp_file.data()<<endl;
		cvSaveImage(fp_file.data(), orig);
	}


   // if (nFaces > 0)
   // {
   //     printf("Faces detected: %d; Detection of facial landmark on all faces took %d ms\n", nFaces, ms);
   // } else {
   //     printf("NO Face\n");
   // }
    
    cvReleaseMemStorage(&storage);
	return false;
}

int main( int argc, char** argv ) 
{
	ofstream fout;
	fout.open("people.csv", ofstream::out);
    char flandmark_window[] = "flandmark_example1";
    double t;
    int ms;
    
    if (argc < 2)
    {
      fprintf(stderr, "Usage: flandmark_1 <path_to_input_image> [<path_to_output_image>]\n");
      exit(1);
    }
    
    //cvNamedWindow(flandmark_window, 0);
    
    // Haar Cascade file, used for Face Detection.
    char faceCascadeFilename[] = "haarcascade_frontalface_alt.xml";
    // Load the HaarCascade classifier for face detection.
    CvHaarClassifierCascade* faceCascade;
    faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0);
    if( !faceCascade )
    {
        printf("Couldnt load Face detector '%s'\n", faceCascadeFilename);
        exit(1);
    }

     // ------------- begin flandmark load model
    t = (double)cvGetTickCount();
    FLANDMARK_Model * model = flandmark_init("flandmark_model.dat");

    if (model == 0)
    {
        printf("Structure model wasn't created. Corrupted file flandmark_model.dat?\n");
        exit(1);
    }

    t = (double)cvGetTickCount() - t;
    ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
    printf("Structure model loaded in %d ms.\n", ms);
    // ------------- end flandmark load model
    

    
    int *bbox = (int*)malloc(4*sizeof(int));
    double *landmarks = (double*)malloc(2*model->data.options.M*sizeof(double));
	
	mkdir(argv[2], S_IRWXU);
	DIR *pDIR;
	struct dirent *entry;
	struct stat buf;
	pDIR = opendir(argv[1]);
	if(( pDIR=opendir(argv[1])) == NULL){
		std::cout<<"cannot open"<<std::endl;
		exit(1);
	}
	entry = readdir(pDIR);
    int id = 1;
	while(entry != NULL)
	{
		if(0 != strcmp( ".", entry->d_name) && //Skip those directories
		   0 != strcmp( "..", entry->d_name) )
		{
			char * name = entry->d_name;
			stat(name, &buf);
			std::cout << name<<std::endl;
			std::string s1 = argv[2];
			std::string s2 = name;
			fout<<id<<','<<name<<',';
			int goodCount = 0;
			char buf[10];
			sprintf(buf, "%d", id);
			std::string dir_path = s1 + '/' + buf;
			std::cout<<dir_path<<std::endl;
			string failpath = string("fail/") + buf;
			string fppath = string("fp/") + buf;
			mkdir(dir_path.data(), S_IRWXU);
			mkdir(failpath.data(), S_IRWXU);
			mkdir(fppath.data(), S_IRWXU);
			
			DIR* pDIR2;
			std::string s3 = argv[1];
			std::string origin_path = s3 + '/' + s2;
			pDIR2 = opendir(origin_path.data());
			struct dirent *entry2;
			entry2 = readdir(pDIR2);
			while(entry2 != NULL){
				if(0 != strcmp( ".", entry2->d_name) && 0 != strcmp( "..", entry2->d_name) )
				{
					//std::cout<<"\t"<<entry2->d_name<<std::endl;
					std::string s4 = entry2->d_name;
					std::string img_path = dir_path + '/' + s4;
					std::string img_origin_path = origin_path + '/' + s4;
					//Mat originalImage = imread( img_origin_path, CV_LOAD_IMAGE_GRAYSCALE );
					 // input image
					IplImage *frame = cvLoadImage(img_origin_path.data());
					if (frame != NULL)
					{
					  // convert image to grayscale
						IplImage *frame_bw = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);
						cvConvertImage(frame, frame_bw);
						if (detectFaceInImage(frame, frame_bw, faceCascade, model, bbox, landmarks, entry2->d_name, dir_path, failpath, fppath))
							goodCount++;
						cvReleaseImage(&frame_bw);
					}
					cvReleaseImage(&frame);
				}
				entry2 = readdir(pDIR2);
			}
			fout<<goodCount<<endl;
			id++;
			closedir(pDIR2);
			
		}
		entry = readdir(pDIR);             //Next file in directory        
	}
	closedir(pDIR);
	fout.close();
    //detectFaceInImage(frame, frame_bw, faceCascade, model, bbox, landmarks);
    
    //cvShowImage(flandmark_window, frame);
    //cvWaitKey(0);
    
    //if (argc == 3)
    //{
    //  printf("Saving image to file %s...\n", argv[2]);
    //  cvSaveImage(argv[2], frame);
    //}
    
    // cleanup
    free(bbox);
    free(landmarks);
    //cvDestroyWindow(flandmark_window);
    
    cvReleaseHaarClassifierCascade(&faceCascade);
    flandmark_free(model);
}
