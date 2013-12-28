#ifndef __MBLBP_DETECT__
#define __MBLBP_DETECT__

#include <opencv/cv.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct MBLBPWeak_
{
    int x;
    int y;
    int cellwidth;
    int cellheight;
    int* p[16]; // fast pointer
    int look_up_table[59]; // look up table
}MBLBPWeak;

typedef struct MBLBPStage_
{
    int count;
    int threshold;
    MBLBPWeak * weak_classifiers;
}MBLBPStage;

typedef struct MBLBPCascade_
{
    int count;
    int win_width;
    int win_height;
    int sum_image_step;
    MBLBPStage * stages;
}MBLBPCascade;

MBLBPCascade * LoadMBLBPCascade(const char * filename );
void ReleaseMBLBPCascade(MBLBPCascade ** ppCascade);

CvSeq * MBLBPDetectMultiScale( const IplImage* img, //输入图像
                               MBLBPCascade * pCascade, //分类器
                               CvMemStorage* storage, //内存
                               int scale_factor1024x, //扫描窗口缩放系数，是浮点数的1024倍，如果是1.1，此处应为1024*1.1=1126
                               int min_neighbors, //聚类最小近邻数
                               int min_size, //最小扫描窗口大小（即窗口宽度）
							   int max_size=0); //最大扫描窗口大小（即窗口宽度）
#endif
