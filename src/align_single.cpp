/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 */
#define PI 3.14159265
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <cstring>
#include <cmath>
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

void detectFaceInImage(IplImage *orig, IplImage* input, CvHaarClassifierCascade* cascade, FLANDMARK_Model *model, int *bbox, double *landmarks)
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

    // Detect all the faces in the greyscale image.
	double t = (double)cvGetTickCount();
    rects = cvHaarDetectObjects(input, cascade, storage, search_scale_factor, 2, flags, minFeatureSize);
    t = (double)cvGetTickCount() - t;
    int ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
	printf("Face detection finishes in %d ms.\n", ms);
	
	nFaces = rects->total;

    t = (double)cvGetTickCount();
	Mat mat_orig(orig, 0);
    for (int iface = 0; iface < (rects ? nFaces : 0); ++iface)
    {
        CvRect *r = (CvRect*)cvGetSeqElem(rects, iface);
        
        bbox[0] = r->x;
        bbox[1] = r->y;
        bbox[2] = r->x + r->width;
        bbox[3] = r->y + r->height;
        
        flandmark_detect(input, bbox, model, landmarks);
	cout<<landmarks[1]<<" "<<landmarks[2]<<" "<<landmarks[3]<<" "<<landmarks[4]<<endl;
        // display landmarks
	cvRectangle(orig, cvPoint(bbox[0], bbox[1]), cvPoint(bbox[2], bbox[3]), CV_RGB(255,0,0) );
	cvRectangle(orig, cvPoint(model->bb[0], model->bb[1]), cvPoint(model->bb[2], model->bb[3]), CV_RGB(0,0,255) );
	cvCircle(orig, cvPoint((int)landmarks[0], (int)landmarks[1]), 3, CV_RGB(0,0,255), CV_FILLED);
		
	cvCircle(orig, cvPoint(int(landmarks[2]), int(landmarks[3])), 3, CV_RGB(255,0,0), CV_FILLED);
	cvCircle(orig, cvPoint(int(landmarks[4]), int(landmarks[5])), 3, CV_RGB(255,0,0), CV_FILLED);
	cvCircle(orig, cvPoint(int(landmarks[6]), int(landmarks[7])), 3, CV_RGB(255,0,0), CV_FILLED);

	cvCircle(orig, cvPoint(int(landmarks[8]), int(landmarks[9])), 3, CV_RGB(255,0,0), CV_FILLED);
	cvCircle(orig, cvPoint(int(landmarks[10]), int(landmarks[11])), 3, CV_RGB(255,0,0), CV_FILLED);
	cvCircle(orig, cvPoint(int(landmarks[12]), int(landmarks[13])), 3, CV_RGB(255,0,0), CV_FILLED);
	cvCircle(orig, cvPoint(int(landmarks[14]), int(landmarks[15])), 3, CV_RGB(255,0,0), CV_FILLED);
	
	//center face
//	double centerx = r->x + r->width/2;
//	double centery = r->y + r->height/2;

//	double offsetx = landmarks[0] - centerx;
//	double offsety = landmarks[1] - centery;

//	r->x += offsetx;
//	r->y += offsety;
	
	//align face
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
	Rect faceRect(r->x, r->y,r->width, r->height);
	if (faceRect.height < 100 && faceRect.width < 100){
		cout<<"Face too small"<<endl;
		continue;
	}
	Mat croppedFaceImage = mat_orig(faceRect).clone();
	Mat rotated = rotateImage(croppedFaceImage, angle_rotate * 180 / PI);
	Mat resized;
	resize(rotated, resized, Size(100, 100));
	//output face
	imwrite( "./tmp/face.jpg" , resized );
        //for (int i = 2; i < 2*model->data.options.M; i += 2)
        //{
        //    cvCircle(orig, cvPoint(int(landmarks[i]), int(landmarks[i+1])), 3, CV_RGB(255,0,0), CV_FILLED);

       // }
    }
    t = (double)cvGetTickCount() - t;
    ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );

    if (nFaces > 0)
    {
        printf("Faces detected: %d; Detection of facial landmark on all faces took %d ms\n", nFaces, ms);
    } else {
        printf("NO Face\n");
    }
    
    cvReleaseMemStorage(&storage);
}

int main( int argc, char** argv ) 
{
    char flandmark_window[] = "align_example";
    double t;
    int ms;
    
    if (argc < 2)
    {
      fprintf(stderr, "Usage: align_single <path_to_input_image>\n");
      exit(1);
    }
    
    //cvNamedWindow(flandmark_window, 0);
    
    // Haar Cascade file, used for Face Detection.
    char faceCascadeFilename[] = "./model/haarcascade_frontalface_alt.xml";
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
    FLANDMARK_Model * model = flandmark_init("./model/flandmark_model.dat");

    if (model == 0)
    {
        printf("Structure model wasn't created. Corrupted file flandmark_model.dat?\n");
        exit(1);
    }

    t = (double)cvGetTickCount() - t;
    ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
    printf("Structure model loaded in %d ms.\n", ms);
    // ------------- end flandmark load model
    
    // input image
    IplImage *frame = cvLoadImage(argv[1]);
    if (frame == NULL)
    {
      fprintf(stderr, "Cannot open image %s. Exiting...\n", argv[1]);
      exit(1);
    }
    // convert image to grayscale
    IplImage *frame_bw = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);
    cvConvertImage(frame, frame_bw);
    
    int *bbox = (int*)malloc(4*sizeof(int));
    double *landmarks = (double*)malloc(2*model->data.options.M*sizeof(double));
    detectFaceInImage(frame, frame_bw, faceCascade, model, bbox, landmarks);
    
    //cvShowImage(flandmark_window, frame);
    //cvWaitKey(0);
    
    if (argc == 2)
    {
      
      cvSaveImage("./tmp/landmark.jpg", frame);
	  printf("Landmark saved to file ./tmp/landmark.jpg\nFace saved to file ./tmp/face.jpg\n");
    }
    
    // cleanup
    free(bbox);
    free(landmarks);
    //cvDestroyWindow(flandmark_window);
    cvReleaseImage(&frame);
    cvReleaseImage(&frame_bw);
    cvReleaseHaarClassifierCascade(&faceCascade);
    flandmark_free(model);
}
