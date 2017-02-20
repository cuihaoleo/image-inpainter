#ifndef __INPAINTER_H__
#define __INPAINTER_H__

#include "CL/cl.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/ocl.hpp"
#include <chrono>

//#define USE_QCOM_EXT

#ifdef USE_QCOM_EXT

#include <linux/ion.h>
#include <sys/mman.h>
#include <fcntl.h>

//#include <linux/msm_ion.h>
#define ION_SYSTEM_HEAP_ID 25
#define ION_HEAP(bit) (1 << (bit))

typedef struct {
    int ion_fd;

    size_t len;
    size_t align;

    int fd;
    ion_user_handle_t handle;
    void *ptr;

    size_t pitch;
} OclIonMemory;

#endif

class AbstractInpainter {
private:
    virtual bool detectFillFront() = 0;
    virtual void findBestMatching() = 0;
    virtual void applyPatch() = 0;

    std::chrono::steady_clock::time_point timestamp;
    long measureTimeInMs(bool init=false);

protected:
    const static size_t DEFAULT_HALF_PATCH_WIDTH = 4;

    size_t patchWidth;
    inline size_t halfPatchWidth() {
        return patchWidth >> 1;
    }

public:
    virtual cv::Mat getResult() = 0;
    void inpaint();
    
    virtual ~AbstractInpainter() { };
};


class CpuInpainter: public AbstractInpainter {
private:
    cv::Mat image;
    cv::Mat mask;

    cv::Mat confidence;
    cv::Mat gradient;

    float nextConfidence;
    cv::Point patchCenter;
    cv::Rect patchROI;
    cv::Rect fillROI;

    cv::Mat derivKernelA, derivKernelB;

    bool detectFillFront();
    void findBestMatching();
    void applyPatch();

public:
    cv::Mat getResult();
    CpuInpainter(
            cv::Mat inputImage,
            cv::Mat mask,
            size_t halfPatchWidth = DEFAULT_HALF_PATCH_WIDTH);
};


class OclInpainter: public AbstractInpainter {
private:
    static const size_t workgroupSize = 16;
    static constexpr const char* oclSourceFile = "inpainter.cl";

    cv::Size2i originalSize;
    cv::Size2i size;

    cv::UMat padImage, padMask;

    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue commandQueue;

#ifdef USE_QCOM_EXT
    // must be before cl::Image2D declaration
    struct {
        bool initialized = false;
        size_t devicePageSize = 0;
        size_t padding = 0;

        OclIonMemory image, mask, confidence;
    } ionEnv;

    OclIonMemory allocateIonMemory(
                cl::ImageFormat format,
                cv::Size2i size);
    static void freeIonMemory(const OclIonMemory &ion);
#endif

    cl::Image2D oclImage;
    cl::Image2D oclMask;
    cl::Image2D oclConfidence;

    cv::Point2i patchCenter;
    cv::Rect patchROI;
    cv::Rect fillROI;

    cv::Mat derivKernelA, derivKernelB;

    size_t globalRange[2];
    size_t nWorkgroups;

    struct {
        bool initialized = false;
        cl::Kernel kernel;

        cl::Buffer indexRecords;
        cl::Buffer priorityRecords;
        cl::Event indexRecordsW;
        cl::Event priorityRecordsW;
    } oclEnv_detectFillFront;

    struct {
        bool initialized = false;
        cl::Kernel kernel;

        cl::Buffer indexRecords;
        cl::Buffer errorRecords;
        cl::Event indexRecordsW;
        cl::Event errorRecordsW;
    } oclEnv_findBestMatching;

    struct {
        cl::Event imageW;
        cl::Event maskW;
        cl::Event confidenceW;
    } oclEvents;

    void buildSource();

    bool detectFillFront();
    void findBestMatching();
    void applyPatch();

public:
    cv::Mat getResult();
    OclInpainter(
            cv::Mat inputImage,
            cv::Mat mask,
            size_t halfPatchWidth = DEFAULT_HALF_PATCH_WIDTH);
};



#endif
