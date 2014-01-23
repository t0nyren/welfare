#include "detector.h"
#include "mblbp-detect.h"
#define PI 3.14159265

#define PATH_CASCADE "./model/szu.bin"
#define PATH_FLANDMARK "./model/flandmark_model.dat"

Detector::Detector(){
	faceCascade = LoadMBLBPCascade(PATH_CASCADE);
	//faceCascade = (CvHaarClassifierCascade*)cvLoad(model_haarcascade.data(), 0, 0, 0);
	debug = 0;
	if(faceCascade == NULL)
    {
        printf("Couldn't load Face detector '%s'\n", PATH_CASCADE);
        exit(1);
    }
	fmodel = flandmark_init(PATH_FLANDMARK);
}

void Detector::setDebug(bool isdebug){
	debug = isdebug;
}

Mat Detector::detect(string imgname){
	//cout<<"Debug: "<<debug<<endl;
	Mat resized;
	IplImage *frame = cvLoadImage(imgname.data(), 2|4);
	if (frame == NULL)
    {
      fprintf(stderr, "Cannot open image %s.Returning empty Mat...\n", imgname.data());
      return resized;
    }
	
	else if (frame->width < 100 || frame->height < 100)
    {
      fprintf(stderr, "image %s too small.Returning empty Mat...\n", imgname.data());
      cvReleaseImage(&frame);
	  return resized;
    }	
	else if (frame->width > 100000 || frame->height > 100000)
    {
      fprintf(stderr, "image %s too large.Returning empty Mat...\n", imgname.data());
      cvReleaseImage(&frame);
	  return resized;
    }
	
	// convert image to grayscale
    IplImage *frame_bw = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);
    cvConvertImage(frame, frame_bw);
	Mat frame_mat(frame, 1);
	// Smallest face size.
    CvSize minFeatureSize = cvSize(100, 100);
    int flags =  CV_HAAR_DO_CANNY_PRUNING;
    // How detailed should the search be.
    float search_scale_factor = 1.1f;
    CvMemStorage* storage;
    CvSeq* rects;
    int nFaces;
	
    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);

    // Detect all the faces in the greyscale image.
   //rects = cvHaarDetectObjects(frame_bw, faceCascade, storage, search_scale_factor, 2, flags, minFeatureSize);
    rects = MBLBPDetectMultiScale(frame_bw, faceCascade, storage, 1229, 1, 50, 500);
	nFaces = rects->total;
	if (nFaces != 1){
		if (debug)
			printf("%d faces detected\n", nFaces);
		storage = cvCreateMemStorage(0);
		cvReleaseMemStorage(&storage);
		cvReleaseImage(&frame_bw);
		cvReleaseImage(&frame);	
		return resized;
	}
		
	int iface = 0;
	CvRect *r = (CvRect*)cvGetSeqElem(rects, iface);
	double* landmarks  = new double[2*fmodel->data.options.M];
	int bbox[4];
	bbox[0] = r->x;
	bbox[1] = r->y;
	bbox[2] = r->x + r->width;
	bbox[3] = r->y + r->height;
	
	// Detect landmarks
	flandmark_detect(frame_bw, bbox, fmodel, landmarks);
	
	//align faces
	double angle[3];
	angle[0] = atan((landmarks[7]-landmarks[9])/(landmarks[6]-landmarks[8]));
	angle[1] = atan((landmarks[11]-landmarks[13])/(landmarks[10]-landmarks[12]));
	angle[2] = atan((landmarks[3]-landmarks[5])/(landmarks[2]-landmarks[4]));
	//cout<<angle[0]*180<<" "<<angle[1]*180<<" "<<angle[2]*180<<endl;
	
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

	//save face to tmp if debug
	if (debug){
		cvRectangle(frame, cvPoint(bbox[0], bbox[1]), cvPoint(bbox[2], bbox[3]), CV_RGB(255,0,0) );
		cvRectangle(frame, cvPoint(fmodel->bb[0], fmodel->bb[1]), cvPoint(fmodel->bb[2], fmodel->bb[3]), CV_RGB(0,0,255) );
		cvCircle(frame, cvPoint((int)landmarks[0], (int)landmarks[1]), 3, CV_RGB(0,0,255), CV_FILLED);
		cvCircle(frame, cvPoint(int(landmarks[2]), int(landmarks[3])), 3, CV_RGB(255,0,0), CV_FILLED);
		cvCircle(frame, cvPoint(int(landmarks[4]), int(landmarks[5])), 3, CV_RGB(255,0,0), CV_FILLED);
		cvCircle(frame, cvPoint(int(landmarks[6]), int(landmarks[7])), 3, CV_RGB(255,0,0), CV_FILLED);
		cvCircle(frame, cvPoint(int(landmarks[8]), int(landmarks[9])), 3, CV_RGB(255,0,0), CV_FILLED);
		cvCircle(frame, cvPoint(int(landmarks[10]), int(landmarks[11])), 3, CV_RGB(255,0,0), CV_FILLED);
		cvCircle(frame, cvPoint(int(landmarks[12]), int(landmarks[13])), 3, CV_RGB(255,0,0), CV_FILLED);
		cvCircle(frame, cvPoint(int(landmarks[14]), int(landmarks[15])), 3, CV_RGB(255,0,0), CV_FILLED);
		Mat face(frame, 0);
		Mat croppedFaceImage = face(faceRect).clone();
		Mat rotated = rotateImage(croppedFaceImage, angle_rotate * 180 / PI);
		resize(rotated, resized, Size(100, 100));
		imwrite( "./tmp/face.jpg" , resized );
	}
	delete [] landmarks;
	storage = cvCreateMemStorage(0);
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&frame_bw);
	cvReleaseImage(&frame);
	
	if (faceRect.height < 50 && faceRect.width < 50){
		printf("Face too small: %d x %d\n", faceRect.height, faceRect.width);
		return resized;
	}
	Mat croppedFaceImage = frame_mat(faceRect).clone();
	Mat rotated = rotateImage(croppedFaceImage, angle_rotate * 180 / PI);
	resize(rotated, resized, Size(100, 100));
	return resized;
}

Mat Detector::rotateImage(const Mat& source, double angle)
{
    Point2f src_center(source.cols/2.0F, source.rows/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    Mat dst;
    warpAffine(source, dst, rot_mat, source.size());
    return dst;
}
