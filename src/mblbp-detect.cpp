#include "mblbp-detect.h"

#include <stdio.h>


#ifdef _OPENMP
static omp_lock_t lock;
#endif 

#define MBLBP_LUTLENGTH  59

#define MBLBP_CALC_SUM(p0, p1, p2, p3, offset_) \
((p0)[offset_] - (p1)[offset_] - (p2)[offset_] + (p3)[offset_])

static uchar MBLBP_LBPTABLE[256] = {1,   2,   3,   4,   5,   0,   6,   7,   8,   0,   0,   0,   9,   0,  10,  11,
	12,   0,   0,   0,   0,   0,   0,   0,  13,   0,   0,   0,  14,   0,  15,  16,
	17,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	18,   0,   0,   0,   0,   0,   0,   0,  19,   0,   0,   0,  20,   0,  21,  22,
	23,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	25,   0,   0,   0,   0,   0,   0,   0,  26,   0,   0,   0,  27,   0,  28,  29,
	30,  31,   0,  32,   0,   0,   0,  33,   0,   0,   0,   0,   0,   0,   0,  34,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  36,
	37,  38,   0,  39,   0,   0,   0,  40,   0,   0,   0,   0,   0,   0,   0,  41,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  42,
	43,  44,   0,  45,   0,   0,   0,  46,   0,   0,   0,   0,   0,   0,   0,  47,
	48,  49,   0,  50,   0,   0,   0,  51,  52,  53,   0,  54,  55,  56,  57,  58};

inline int is_equal( const void* _r1, const void* _r2, void* )
{
    const CvRect* r1 = (const CvRect*)_r1;
    const CvRect* r2 = (const CvRect*)_r2;
    /*
	int distance5x = r1->width ;//int distance = cvRound(r1->width*0.2);

    return r2->x*5 <= r1->x*5 + distance5x &&
           r2->x*5 >= r1->x*5 - distance5x &&
           r2->y*5 <= r1->y*5 + distance5x &&
           r2->y*5 >= r1->y*5 - distance5x &&
           r2->width*5 <= r1->width * 6 &&
           r2->width * 6 >= r1->width*5;
    */
	int delta10x =  MIN(r1->width, r2->width) + MIN(r1->height, r2->height);
	return abs(r1->x - r2->x)*10 <= delta10x &&
           abs(r1->y - r2->y)*10 <= delta10x &&	
           abs(r1->x+r1->width - r2->x-r2->width)*10 <= delta10x &&	
           abs(r1->y+r1->height - r2->y-r2->height)*10 <= delta10x ;
}


MBLBPCascade * LoadMBLBPCascade(const char * filename )
{
    FILE *pFile = fopen(filename, "rb");
  
    if (pFile == NULL) {
        fprintf(stderr, "Can not load detector from file %s\n", filename);
        return NULL;
    }

    MBLBPCascade * pCascade = (MBLBPCascade*)cvAlloc(sizeof(MBLBPCascade));
    memset(pCascade, 0, sizeof(MBLBPCascade));

    if (fread( &(pCascade->win_width), sizeof(int), 1, pFile) != 1)
        goto EXIT_TAG;

    if (fread( &(pCascade->win_height), sizeof(int), 1, pFile) != 1) 
        goto EXIT_TAG;
  
    if (fread(&(pCascade->count), sizeof(int), 1, pFile) != 1)
        goto EXIT_TAG;

    pCascade->stages = (MBLBPStage*)cvAlloc(sizeof(MBLBPStage) * pCascade->count);
    memset(pCascade->stages, 0, sizeof(MBLBPStage) * pCascade->count);

	for (int i = 0; i < pCascade->count; i++) 
    {
        if ( fread( &(pCascade->stages[i].count) , sizeof(int), 1, pFile) != 1) 
            goto EXIT_TAG;

        //float tmp;

        if ( fread( &(pCascade->stages[i].threshold), sizeof(int), 1, pFile) != 1)
            goto EXIT_TAG;

        pCascade->stages[i].weak_classifiers = (MBLBPWeak*)cvAlloc( sizeof(MBLBPWeak) *  pCascade->stages[i].count);
        memset(pCascade->stages[i].weak_classifiers, 0, sizeof(MBLBPWeak) *  pCascade->stages[i].count);

        for(int j = 0; j < pCascade->stages[i].count; j++)
        {
            MBLBPWeak * pWeak = pCascade->stages[i].weak_classifiers + j;

            if (fread(&(pWeak->x), sizeof(int), 1, pFile) != 1) 
                goto EXIT_TAG;
            if (fread(&(pWeak->y), sizeof(int), 1, pFile) != 1) 
                goto EXIT_TAG;
            if (fread(&(pWeak->cellwidth), sizeof(int), 1, pFile) != 1) 
                goto EXIT_TAG;
            if (fread(&(pWeak->cellheight), sizeof(int), 1, pFile) != 1) 
                goto EXIT_TAG;

            for(int k = 0; k < MBLBP_LUTLENGTH; k++) {
                if (fread(pWeak->look_up_table+k, sizeof(int), 1, pFile) != 1) 
                    goto EXIT_TAG;
            }
        }
    }

    fclose(pFile);
	return pCascade;

 EXIT_TAG:
  
	fclose(pFile);
    ReleaseMBLBPCascade(&pCascade);
    return NULL;
}

void ReleaseMBLBPCascade(MBLBPCascade ** ppCascade)
{
    if(!ppCascade )
        return;

    MBLBPCascade * pCascade = *ppCascade;
    if(!pCascade)
        return;

    for(int i = 0; i < pCascade->count && pCascade->stages; i++)
    {
        for(int j = 0; j < pCascade->stages[i].count; j++)
            cvFree(&(pCascade->stages[i].weak_classifiers));
           
    }
    cvFree(&(pCascade->stages));
    cvFree(ppCascade);
}


void myIntegral(const IplImage * image, IplImage *sumImage)
{
    CV_FUNCNAME( "myIntegral" );

    __BEGIN__;

    CvMat src_stub, *src = (CvMat*)image;
    CvMat sum_stub, *sum = (CvMat*)sumImage;
    int src_step, sum_step;
    CvSize size;


    CV_CALL( src = cvGetMat( src, &src_stub ));
    CV_CALL( sum = cvGetMat( sum, &sum_stub ));
    
    if( sum->width != src->width ||
        sum->height != src->height )
        CV_ERROR( CV_StsUnmatchedSizes, "" );

	if(CV_MAT_DEPTH(src->type)!=CV_8U || CV_MAT_CN(src->type)!=1)
		CV_ERROR( CV_StsUnsupportedFormat, "the source array must be 8UC1");

    if( CV_MAT_DEPTH( sum->type ) != CV_32S ||
        !CV_ARE_CNS_EQ( src, sum ))
        CV_ERROR( CV_StsUnsupportedFormat,
        "Sum array must have 32s type in case of 8u source array"
        "and the same number of channels as the source array" );

    size = cvGetMatSize(src);
    src_step = src->step ? src->step : CV_STUB_STEP;
    sum_step = sum->step ? sum->step : CV_STUB_STEP;
    sum_step /= sizeof(int);

    {
        int x,y;
        int s;
        unsigned char * psrc;
        int * psum;

        psrc = (unsigned char*)(src->data.ptr);
        psum = sum->data.i;

        //the top left corner
        psum[0] = psrc[0];
        //the first row
        for(x=1; x < size.width; x++)
            psum[x] = psum[x-1] + psrc[x];
        //the first column
        for(y=1; y < size.height; y++)
            psum[y * sum_step] = psum[ (y-1) * sum_step] + psrc[y * src_step];

        for( y = 1, psrc += src_step, psum += sum_step;
             y < size.height; 
             y++, psrc += src_step, psum += sum_step )
        {                  
            for( x = 1, s = psrc[0]; x < size.width; x++ )            
            {
                s += (psrc[x]);
                psum[x] = psum[x - sum_step] + s;
            }                                                   
        }                                                       
    }

    __END__;
    return ;
}



void UpdateCascade(MBLBPCascade * pCascade, IplImage *sum)
{
    int step;

    CV_FUNCNAME( "UpdateCascade" );

    __BEGIN__;

    if( !sum )
        CV_ERROR( CV_StsNullPtr, "Null integral image pointer" );
    
    if( ! pCascade) 
        CV_ERROR( CV_StsNullPtr, "Invalid classifier cascade" );
    
    step = sum->widthStep;
    pCascade->sum_image_step = step;

    for(int i = 0; i < pCascade->count; i++)
    {
        for(int j = 0; j < pCascade->stages[i].count; j++)
        {
            MBLBPWeak * pw =  pCascade->stages[i].weak_classifiers + j;
            int x = pw->x;
            int y = pw->y;
            int w = pw->cellwidth;
            int h = pw->cellheight;

            pw->p[ 0] = (int*)(sum->imageData + y * step + (x      ) * sizeof(int));
            pw->p[ 1] = (int*)(sum->imageData + y * step + (x + w  ) * sizeof(int));
            pw->p[ 2] = (int*)(sum->imageData + y * step + (x + w*2) * sizeof(int));
            pw->p[ 3] = (int*)(sum->imageData + y * step + (x + w*3) * sizeof(int));

            pw->p[ 4] = (int*)(sum->imageData + (y+h) * step + (x      ) * sizeof(int));
            pw->p[ 5] = (int*)(sum->imageData + (y+h) * step + (x + w  ) * sizeof(int));
            pw->p[ 6] = (int*)(sum->imageData + (y+h) * step + (x + w*2) * sizeof(int));
            pw->p[ 7] = (int*)(sum->imageData + (y+h) * step + (x + w*3) * sizeof(int));

            pw->p[ 8] = (int*)(sum->imageData + (y+h*2) * step + (x      ) * sizeof(int));
            pw->p[ 9] = (int*)(sum->imageData + (y+h*2) * step + (x + w  ) * sizeof(int));
            pw->p[10] = (int*)(sum->imageData + (y+h*2) * step + (x + w*2) * sizeof(int));
            pw->p[11] = (int*)(sum->imageData + (y+h*2) * step + (x + w*3) * sizeof(int));

            pw->p[12] = (int*)(sum->imageData + (y+h*3) * step + (x      ) * sizeof(int));
            pw->p[13] = (int*)(sum->imageData + (y+h*3) * step + (x + w  ) * sizeof(int));
            pw->p[14] = (int*)(sum->imageData + (y+h*3) * step + (x + w*2) * sizeof(int));
            pw->p[15] = (int*)(sum->imageData + (y+h*3) * step + (x + w*3) * sizeof(int));
        }
    }

    __END__;
   return;
}



inline int DetectAt(MBLBPCascade * pCascade, int offset)
{
    if( !pCascade)
        return 0;
    int confidence=0;

	for(int i = 0; i < pCascade->count; i++)
    {
        int stage_sum = 0;
        int code = 0;
        int bit = 0;

        MBLBPWeak * pw =  pCascade->stages[i].weak_classifiers;

        for(int j = 0; j < pCascade->stages[i].count; j++)
        {
            int ** p = pw->p;

            int cval = MBLBP_CALC_SUM( p[5], p[6], p[9], p[10], offset );

            code = ((MBLBP_CALC_SUM( p[0], p[1], p[4], p[5], offset ) >= cval ) << 7 ) |
                ((MBLBP_CALC_SUM( p[1], p[2], p[5], p[6], offset ) >= cval ) << 6) | 
                ((MBLBP_CALC_SUM( p[2], p[3], p[6], p[7], offset ) >= cval ) << 5) |
                ((MBLBP_CALC_SUM( p[6], p[7], p[10], p[11], offset ) >= cval ) << 4) | 
                ((MBLBP_CALC_SUM( p[10], p[11], p[14], p[15], offset ) >= cval ) << 3)| 
                ((MBLBP_CALC_SUM( p[9], p[10], p[13], p[14], offset ) >= cval ) << 2)|  
                ((MBLBP_CALC_SUM( p[8], p[9], p[12], p[13], offset ) >= cval ) << 1)|
                ((MBLBP_CALC_SUM( p[4], p[5], p[8], p[9], offset ) >= cval )   );

			stage_sum += pw->look_up_table[ MBLBP_LBPTABLE[code] ];

            pw++;
        }

        if(stage_sum < pCascade->stages[i].threshold)
            return -i;
        else
            confidence = stage_sum - pCascade->stages[i].threshold;
    }

    return confidence;
}


void MBLBPDetectSingleScale( const IplImage* img,
                             MBLBPCascade * pCascade,
                             CvSeq * positions, 
                             CvSize winStride)
{
    IplImage * sum = 0;
    int ystep, xstep, ymax, xmax;
    
    CV_FUNCNAME( "MBLBPDetectSingleScale" );

    __BEGIN__;


    if( !img )
        CV_ERROR( CV_StsNullPtr, "Null image pointer" );

    if( ! pCascade) 
        CV_ERROR( CV_StsNullPtr, "Invalid classifier cascade" );

    if( !positions )
        CV_ERROR( CV_StsNullPtr, "Null CvSeq pointer" );

    if(pCascade->win_width > img->width || 
       pCascade->win_height > img->height)
        return ;



    CV_CALL( sum = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32S, 1));
    myIntegral(img, sum);
    //cvIntegral(img, sum);
    UpdateCascade(pCascade, sum);

    ystep = winStride.height;
    xstep = winStride.width;
    ymax = img->height - pCascade->win_height -1;
	xmax = img->width  - pCascade->win_width -1;

#ifdef _OPENMP
    #pragma omp parallel for
#endif

	for(int iy = 0; iy < ymax; iy+=ystep)
    {
       for(int ix = 0; ix < xmax; ix+=xstep)
        {
            int w_offset = iy * sum->widthStep / sizeof(int) + ix;
			int result = DetectAt(pCascade, w_offset);
            if( result > 0)
            {
                //since the integral image is different with that of OpenCV,
                //update the position to OpenCV's by adding 1.
                CvPoint pt = cvPoint(ix+1, iy+1);
#ifdef _OPENMP
omp_set_lock(&lock); 
#endif
                cvSeqPush(positions, &pt);
#ifdef _OPENMP
omp_unset_lock(&lock);
#endif
			}
			if(result == 0)
			{
				ix += xstep;
			}
        }
    }

    __END__;

    cvReleaseImage(&sum);
    return ;
}

CvSeq * MBLBPDetectMultiScale( const IplImage* img,
                               MBLBPCascade * pCascade,
                               CvMemStorage* storage, 
                               int scale_factor1024x,
                               int min_neighbors, 
                               int min_size,
							   int max_size)
{
    IplImage stub;
    CvMat mat, *pmat;
    CvSeq* seq = 0;
    CvSeq* seq2 = 0;
    CvSeq* idx_seq = 0;
    CvSeq* result_seq = 0;
    CvSeq* positions = 0;
    CvMemStorage* temp_storage = 0;
    CvAvgComp* comps = 0;
    
    CV_FUNCNAME( "MBLBPDetectMultiScale" );

    __BEGIN__;

    int factor1024x;
    int factor1024x_max;
    int coi;

    if( ! pCascade) 
        CV_ERROR( CV_StsNullPtr, "Invalid classifier cascade" );

    if( !storage )
        CV_ERROR( CV_StsNullPtr, "Null storage pointer" );

    CV_CALL( img = cvGetImage( img, &stub));
    CV_CALL( pmat = cvGetMat( img, &mat, &coi));

    if( coi )
        CV_ERROR( CV_BadCOI, "COI is not supported" );

    if( CV_MAT_DEPTH(pmat->type) != CV_8U )
        CV_ERROR( CV_StsUnsupportedFormat, "Only 8-bit images are supported" );

    if( CV_MAT_CN(pmat->type) > 1 )
    	CV_ERROR( CV_StsUnsupportedFormat, "Only single-channel images are supported" );

    min_size  = MAX(pCascade->win_width,  min_size);
	if(max_size <=0 )
		max_size = MIN(img->width, img->height);
	if(max_size < min_size)
		return NULL;

	CV_CALL( temp_storage = cvCreateChildMemStorage( storage ));
    seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvRect), temp_storage );
    seq2 = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), temp_storage );
    result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), storage );
    positions = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), temp_storage );

    if( min_neighbors == 0 )
        seq = result_seq;

    factor1024x = ((min_size<<10)+(pCascade->win_width/2)) / pCascade->win_width;
	factor1024x_max = (max_size<<10) / pCascade->win_width; //do not round it, to avoid the scan window be out of range

#ifdef _OPENMP
	omp_init_lock(&lock); 
#endif
    for( ; factor1024x <= factor1024x_max;
         factor1024x = ((factor1024x*scale_factor1024x+512)>>10) )
    {
        IplImage * pSmallImage = cvCreateImage( cvSize( ((img->width<<10)+factor1024x/2)/factor1024x, ((img->height<<10)+factor1024x/2)/factor1024x),
                                                IPL_DEPTH_8U, 1);
        try{
			cvResize(img, pSmallImage);
		}
		catch(...)
		{
			cvReleaseImage(&pSmallImage);
			return NULL;
		}
		
		
        CvSize winStride = cvSize( (factor1024x<=2048)+1,  (factor1024x<=2048)+1 );

		cvClearSeq(positions);

        MBLBPDetectSingleScale( pSmallImage, pCascade, positions, winStride);

        for(int i=0; i < (positions ? positions->total : 0); i++)
        {
            CvPoint pt = *(CvPoint*)cvGetSeqElem( positions, i );
            CvRect r = cvRect( (pt.x * factor1024x + 512)>>10,
                               (pt.y * factor1024x + 512)>>10,
                               (pCascade->win_width * factor1024x + 512)>>10,
                               (pCascade->win_height * factor1024x + 512)>>10);

            cvSeqPush(seq, &r);
        }

        cvReleaseImage(&pSmallImage);
    }
#ifdef _OPENMP
	omp_destroy_lock(&lock); 
#endif
  
    if( min_neighbors != 0 )
    {
        // group retrieved rectangles in order to filter out noise 
        int ncomp = cvSeqPartition( seq, 0, &idx_seq, (CvCmpFunc)is_equal, 0 );
        CV_CALL( comps = (CvAvgComp*)cvAlloc( (ncomp+1)*sizeof(comps[0])));
        memset( comps, 0, (ncomp+1)*sizeof(comps[0]));

        // count number of neighbors
        for(int i = 0; i < seq->total; i++ )
        {
            CvRect r1 = *(CvRect*)cvGetSeqElem( seq, i );
            int idx = *(int*)cvGetSeqElem( idx_seq, i );
            assert( (unsigned)idx < (unsigned)ncomp );

            comps[idx].neighbors++;
             
            comps[idx].rect.x += r1.x;
            comps[idx].rect.y += r1.y;
            comps[idx].rect.width += r1.width;
            comps[idx].rect.height += r1.height;
        }

        // calculate average bounding box
        for(int i = 0; i < ncomp; i++ )
        {
            int n = comps[i].neighbors;
            if( n >= min_neighbors )
            {
                CvAvgComp comp;
                comp.rect.x = (comps[i].rect.x*2 + n)/(2*n);
                comp.rect.y = (comps[i].rect.y*2 + n)/(2*n);
                comp.rect.width = (comps[i].rect.width*2 + n)/(2*n);
                comp.rect.height = (comps[i].rect.height*2 + n)/(2*n);
                comp.neighbors = comps[i].neighbors;

                cvSeqPush( seq2, &comp );
            }
        }

        // filter out small face rectangles inside large face rectangles
        for(int i = 0; i < seq2->total; i++ )
        {
            CvAvgComp r1 = *(CvAvgComp*)cvGetSeqElem( seq2, i );
            int j, flag = 1;

            for( j = 0; j < seq2->total; j++ )
            {
                CvAvgComp r2 = *(CvAvgComp*)cvGetSeqElem( seq2, j );
                int distance = (r2.rect.width *2+5)/10;//cvRound( r2.rect.width * 0.2 );
            
                if( i != j &&
                    r1.rect.x >= r2.rect.x - distance &&
                    r1.rect.y >= r2.rect.y - distance &&
                    r1.rect.x + r1.rect.width <= r2.rect.x + r2.rect.width + distance &&
                    r1.rect.y + r1.rect.height <= r2.rect.y + r2.rect.height + distance &&
                    (r2.neighbors > MAX( 3, r1.neighbors ) || r1.neighbors < 3) )
                {
                    flag = 0;
                    break;
                }
            }

            if( flag )
            {
                cvSeqPush( result_seq, &r1 );
                /* cvSeqPush( result_seq, &r1.rect ); */
            }
        }
    }   


    __END__;

    cvReleaseMemStorage( &temp_storage );
    cvFree( &comps );

    return result_seq;
}

