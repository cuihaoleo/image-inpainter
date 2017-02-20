#include "inpainter.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <utility>
#include <CL/cl.hpp>

static int roundUp(int n, int base) {
    int r = n % base;
    return r ? n + (base - r) : n;
}


static std::string cvMatToArrayLiteral(const cv::Mat &m) {
    std::stringstream ss;

    ss << std::setprecision(std::numeric_limits<float>::digits10+1);
    for (int i = 0; i < m.cols*m.rows; i++)
        ss << m.at<float>(i) << ",";

    return "{" + ss.str() + "}";
}


static int i2dIndex(const cv::Point2i &point, int rowPitch) {
    return point.y * rowPitch + point.x;
}


void OclInpainter::buildSource() {
    cl_int err;
    size_t localSize1D = workgroupSize * workgroupSize;

    nWorkgroups = (size.area() + localSize1D - 1) / localSize1D;
    globalRange[0] = size.width;
    globalRange[1] = size.height;

    std::ifstream sourceFile(oclSourceFile);
    if (sourceFile.fail())
        throw std::runtime_error(std::string(oclSourceFile) + " not found!");

    std::string kernelSource(
            (std::istreambuf_iterator<char>(sourceFile)),
            std::istreambuf_iterator<char>());

    program = cl::Program(context, kernelSource);

    cv::String buildOption = cv::format("\
            -D HALF_PATCH_WIDTH=%u \
            -D PATCH_WIDTH=%u \
            -D LOCAL_SIZE_1D=%u \
            -D GLOBAL_SIZE_1D=%d \
            -D IMAGE_ROWS=%d \
            -D IMAGE_COLS=%d \
            -D DERIV_KERNEL_A=%s \
            -D DERIV_KERNEL_B=%s",
            halfPatchWidth(),
            patchWidth,
            localSize1D,
            size.area(),
            size.height,
            size.width,
            cvMatToArrayLiteral(derivKernelA).c_str(),
            cvMatToArrayLiteral(derivKernelB).c_str());

    auto devices = std::vector<cl::Device>(1, device);
    err = program.build(devices, buildOption.c_str());
    std::string buildMessage = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);

    if (err) {
        std::cerr << buildMessage << std::endl;
        throw std::runtime_error("detectFillFront.cl failed to build!");
    }
}


OclInpainter::OclInpainter(
        const cv::Mat inputImage,
        const cv::Mat mask,
        size_t halfPatchWidth) {
    cl_int err;

    if (inputImage.type() != CV_8UC3) {
        throw std::runtime_error("inputImage.type() != CV_8UC3");
    }

    if (mask.type() != CV_8UC1) {
        throw std::runtime_error("inputImage.type() != CV_8UC1");
    }

    device = cl::Device((cl_device_id)cv::ocl::Device::getDefault().ptr());
    context = cl::Context((cl_context)cv::ocl::Context::getDefault().ptr());
    //commandQueue = cl::CommandQueue((cl_command_queue)cv::ocl::Queue::getDefault().ptr());
    commandQueue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    originalSize = inputImage.size();
    size.width = roundUp(originalSize.width, workgroupSize);
    size.height = roundUp(originalSize.height, workgroupSize);

    cv::UMat tmp;
    cv::cvtColor(inputImage, tmp, cv::COLOR_BGR2BGRA);
    cv::copyMakeBorder(
            tmp,
            padImage,
            0,
            size.height - originalSize.height,
            0,
            size.width - originalSize.width,
            cv::BORDER_REPLICATE);

    cv::copyMakeBorder(
            mask,
            padMask,
            0,
            size.height - originalSize.height,
            0,
            size.width - originalSize.width,
            cv::BORDER_REPLICATE);

    const size_t origin[3] = { 0, 0, 0 };
    const size_t region[3] = { (size_t)size.width, (size_t)size.height, 1 };

    /*std::vector<cl::ImageFormat> imageFormats;
    context.getSupportedImageFormats(
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_OBJECT_IMAGE2D,
        &imageFormats);
    for (auto format: imageFormats) {
        std::cerr << std::hex
            << format.image_channel_order << ", "
            << format.image_channel_data_type
            << std::endl;
    }*/

    struct {
        cl_mem_flags flags;
        void *ptrImage, *ptrMask, *ptrConfidence;
        size_t pitchImage, pitchMask, pitchConfidence;
    } imageConfig = {};

#ifdef USE_QCOM_EXT
    ionEnv.image = allocateIonMemory(
            cl::ImageFormat(CL_BGRA, CL_UNORM_INT8), size);
    ionEnv.mask = allocateIonMemory(
            cl::ImageFormat(CL_R, CL_UNORM_INT8), size);
    ionEnv.confidence = allocateIonMemory(
            cl::ImageFormat(CL_R, CL_FLOAT), size);

    auto getter = [](OclIonMemory mem) {  
        return cl_mem_ion_host_ptr{
            .ext_host_ptr = {
                .allocation_type = CL_MEM_ION_HOST_PTR_QCOM,
                .host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM
            },
            .ion_filedesc = mem.fd,
            .ion_hostptr = mem.ptr
        };
    };

    auto destructor = [](cl_mem mem, void *userdata) {
        OclIonMemory *p = (OclIonMemory*)userdata;
        freeIonMemory(*p);
        (void)mem;
    };

    struct {
        cl_mem_ion_host_ptr image, mask, confidence;
    } ionHostPtr = {
        getter(ionEnv.image),
        getter(ionEnv.mask),
        getter(ionEnv.confidence)
    };

    imageConfig.flags = CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM;
    imageConfig.ptrImage = &ionHostPtr.image;
    imageConfig.pitchImage = ionEnv.image.pitch;
    imageConfig.ptrMask = &ionHostPtr.mask;
    imageConfig.pitchMask = ionEnv.mask.pitch;
    imageConfig.ptrConfidence = &ionHostPtr.confidence;
    imageConfig.pitchConfidence = ionEnv.confidence.pitch;
#else
    imageConfig.flags = CL_MEM_ALLOC_HOST_PTR;
#endif

    oclImage = cl::Image2D(
            context,
            imageConfig.flags,
            cl::ImageFormat(CL_BGRA, CL_UNORM_INT8),
            size.width,
            size.height,
            imageConfig.pitchImage,
            imageConfig.ptrImage,
            &err);

    if (err) {
        std::cerr << "cl::Image2D initialization failed: " << err << std::endl;
        throw std::runtime_error("Failed to initialize cl::Image2D");
    }

    err = clEnqueueCopyBufferToImage(
        commandQueue(),
        (cl_mem)padImage.handle(cv::ACCESS_READ),
        oclImage(),
        0,
        origin,
        region,
        0,
        NULL,
        &oclEvents.imageW());

    oclMask = cl::Image2D(
            context,
            imageConfig.flags,
            cl::ImageFormat(CL_R, CL_UNORM_INT8),
            size.width,
            size.height,
            imageConfig.pitchMask,
            imageConfig.ptrMask,
            &err);

    err = clEnqueueCopyBufferToImage(
        commandQueue(),
        (cl_mem)padMask.handle(cv::ACCESS_READ),
        oclMask(),
        0,
        origin,
        region,
        0,
        NULL,
        &oclEvents.maskW());

    cv::UMat confidence;
    cv::threshold(padMask, confidence, 0, 1, cv::THRESH_BINARY_INV);
    confidence.convertTo(confidence, CV_32F);
    
    oclConfidence = cl::Image2D(
            context,
            imageConfig.flags,
            cl::ImageFormat(CL_R, CL_FLOAT),
            size.width,
            size.height,
            imageConfig.pitchConfidence,
            imageConfig.ptrConfidence,
            &err);

    err = clEnqueueCopyBufferToImage(
        commandQueue(),
        (cl_mem)confidence.handle(cv::ACCESS_READ),
        oclConfidence(),
        0,
        origin,
        region,
        0,
        NULL,
        &oclEvents.confidenceW());

#ifdef USE_QCOM_EXT
    oclImage.setDestructorCallback(destructor, &ionEnv.image);
    oclMask.setDestructorCallback(destructor, &ionEnv.mask);
    oclConfidence.setDestructorCallback(destructor, &ionEnv.confidence);
#endif

    patchWidth = 2 * halfPatchWidth + 1;
    cv::getDerivKernels(derivKernelA, derivKernelB, 1, 0, patchWidth);

    buildSource();
}


bool OclInpainter::detectFillFront() {
    auto &oclEnv = oclEnv_detectFillFront;
    cl_int err;

    if (!oclEnv.initialized) {
        oclEnv.kernel = cl::Kernel(program, "detectFillFront");
        oclEnv.initialized = true;

        oclEnv.indexRecords = cl::Buffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                nWorkgroups * sizeof(cl_int2));

        oclEnv.priorityRecords = cl::Buffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                nWorkgroups * sizeof(cl_float));
    }

    oclEnv.priorityRecordsW.wait();
    oclEnv.indexRecordsW.wait();

    err = 0;
    err |= oclEnv.kernel.setArg(0, oclImage);
    err |= oclEnv.kernel.setArg(1, oclMask);
    err |= oclEnv.kernel.setArg(2, oclConfidence);
    err |= oclEnv.kernel.setArg(3, oclEnv.indexRecords);
    err |= oclEnv.kernel.setArg(4, oclEnv.priorityRecords);

    if (err) {
        std::cerr << "kernel.setArg failed: " << err << std::endl;
        throw std::runtime_error("Failed to run kernel detectFillFront");
    }

    cl::Event kernelRunningEvent;
    std::vector<cl::Event> eventVector = {
        oclEvents.imageW,
        oclEvents.maskW,
        oclEvents.confidenceW
    };

    err = commandQueue.enqueueNDRangeKernel(
            oclEnv.kernel,
            cl::NullRange,
            cl::NDRange(size.width, size.height),
            cl::NDRange(workgroupSize, workgroupSize),
            &eventVector,
            &kernelRunningEvent);

    if (err) {
        std::cerr << "commandQueue.enqueueNDRangeKernel failed: " << err << std::endl;
        throw std::runtime_error("Failed to run kernel detectFillFront");
    }

    eventVector.clear();
    eventVector.push_back(kernelRunningEvent);

    cl_int2 *indexRecords = (cl_int2*)commandQueue.enqueueMapBuffer(
            oclEnv.indexRecords,
            CL_FALSE,
            CL_MAP_READ,
            0,
            nWorkgroups * sizeof(cl_int2),
            &eventVector,
            &oclEnv.indexRecordsW,
            &err);

    if (err) {
        std::cerr << "commandQueue.enqueueMapBuffer failed: " << err << std::endl;
        throw std::runtime_error("Failed to run kernel detectFillFront");
    }

    cl_float *priorityRecords = (cl_float*)commandQueue.enqueueMapBuffer(
            oclEnv.priorityRecords,
            CL_FALSE,
            CL_MAP_READ,
            0,
            nWorkgroups * sizeof(float),
            &eventVector,
            &oclEnv.priorityRecordsW,
            &err);
    if (err) {
        std::cerr << "commandQueue.enqueueMapBuffer failed: " << err << std::endl;
        throw std::runtime_error("Failed to run kernel detectFillFront");
    }

    oclEnv.priorityRecordsW.wait();
    oclEnv.indexRecordsW.wait();

    float maxPriority = -1.0;
    cl_int2 index = { .s = { 0, 0 } };
    for (size_t i=0; i<nWorkgroups; i++)
        if (priorityRecords[i] > maxPriority) {
            maxPriority = priorityRecords[i];
            index = indexRecords[i];
        }

    commandQueue.enqueueUnmapMemObject(
            oclEnv.priorityRecords,
            priorityRecords,
            nullptr,
            &oclEnv.priorityRecordsW);
    commandQueue.enqueueUnmapMemObject(
            oclEnv.indexRecords,
            indexRecords,
            nullptr,
            &oclEnv.indexRecordsW);

    if (maxPriority < 0)
        return false;

    patchROI = cv::Rect(
            index.s[0] - halfPatchWidth(),
            index.s[1] - halfPatchWidth(),
            patchWidth,
            patchWidth);
    patchCenter = cv::Point2i(index.s[0], index.s[1]);

    std::cerr << patchROI << std::endl;
    std::cerr << "priority: " << maxPriority << std::endl;

    return true;
}


void OclInpainter::findBestMatching() {
    auto &oclEnv = oclEnv_findBestMatching;
    cl_int err;

    if (!oclEnv.initialized) {
        oclEnv.kernel = cl::Kernel(program, "findBestMatching");
        oclEnv.initialized = true;

        oclEnv.indexRecords = cl::Buffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                nWorkgroups * sizeof(cl_int2));

        oclEnv.errorRecords = cl::Buffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                nWorkgroups * sizeof(cl_float));
    }

    oclEnv.errorRecordsW.wait();
    oclEnv.indexRecordsW.wait();

    oclEnv.kernel.setArg(0, oclImage);
    oclEnv.kernel.setArg(1, oclMask);
    oclEnv.kernel.setArg(2, (cl_int2){ .s = { patchROI.x, patchROI.y } });
    oclEnv.kernel.setArg(3, oclEnv.indexRecords);
    oclEnv.kernel.setArg(4, oclEnv.errorRecords);

    cl::Event kernelRunningEvent;
    std::vector<cl::Event> eventVector = {
        oclEvents.imageW,
        oclEvents.maskW,
        oclEvents.confidenceW
    };

    err = commandQueue.enqueueNDRangeKernel(
            oclEnv.kernel,
            cl::NullRange,
            cl::NDRange(size.width, size.height),
            cl::NDRange(workgroupSize, workgroupSize),
            &eventVector,
            &kernelRunningEvent);

    if (err) {
        std::cerr << "commandQueue.enqueueNDRangeKernel failed: " << err << std::endl;
        throw std::runtime_error("Failed to run kernel detectFillFront");
    }

    eventVector.clear();
    eventVector.push_back(kernelRunningEvent);

    cl_int2 *indexRecords = (cl_int2*)commandQueue.enqueueMapBuffer(
            oclEnv.indexRecords,
            CL_FALSE,
            CL_MAP_READ,
            0,
            nWorkgroups * sizeof(cl_int2),
            &eventVector,
            &oclEnv.indexRecordsW,
            &err);

    if (err) {
        std::cerr << "commandQueue.enqueueMapBuffer failed: " << err << std::endl;
        throw std::runtime_error("Failed to run kernel detectFillFront");
    }

    cl_float *errorRecords = (cl_float*)commandQueue.enqueueMapBuffer(
            oclEnv.errorRecords,
            CL_FALSE,
            CL_MAP_READ,
            0,
            nWorkgroups * sizeof(float),
            &eventVector,
            &oclEnv.errorRecordsW,
            &err);
    if (err) {
        std::cerr << "commandQueue.enqueueMapBuffer failed: " << err << std::endl;
        throw std::runtime_error("Failed to run kernel detectFillFront");
    }

    oclEnv.errorRecordsW.wait();
    oclEnv.indexRecordsW.wait();

    float minError = std::numeric_limits<float>::infinity();
    cl_int2 index = { .s = { 0, 0 } };
    for (size_t i=0; i<nWorkgroups; i++)
        if (errorRecords[i] < minError) {
            minError = errorRecords[i];
            index = indexRecords[i];
        }

    commandQueue.enqueueUnmapMemObject(
            oclEnv.errorRecords,
            errorRecords,
            nullptr,
            &oclEnv.errorRecordsW);
    commandQueue.enqueueUnmapMemObject(
            oclEnv.indexRecords,
            indexRecords,
            nullptr,
            &oclEnv.indexRecordsW);

    if (minError < 0)
        return;

    fillROI = cv::Rect(
            index.s[0],
            index.s[1],
            patchWidth,
            patchWidth);

    std::cerr << fillROI << std::endl;
    std::cerr << "error: " << minError << std::endl;
}

void OclInpainter::applyPatch() {
    cl::size_t<3> region;
    region[0] = size.width;
    region[1] = size.height;
    region[2] = 1;

    struct {
        size_t image, mask, confidence;
    } rowPitch;

    cl_uchar4 *imageBuffer = (cl_uchar4*)commandQueue.enqueueMapImage(
            oclImage,
            CL_FALSE,
            CL_MAP_READ | CL_MAP_WRITE,
            cl::size_t<3>(),
            region,
            &rowPitch.image,
            nullptr,
            nullptr,
            &oclEvents.imageW);
    rowPitch.image /= sizeof(cl_uchar4);

    cl_uchar *maskBuffer = (cl_uchar*)commandQueue.enqueueMapImage(
            oclMask,
            CL_FALSE,
            CL_MAP_READ | CL_MAP_WRITE,
            cl::size_t<3>(),
            region,
            &rowPitch.mask,
            nullptr, nullptr,
            &oclEvents.maskW);
    rowPitch.mask /= sizeof(cl_uchar);

    cl_float *confidenceBuffer = (cl_float*)commandQueue.enqueueMapImage(
            oclConfidence,
            CL_FALSE,
            CL_MAP_READ | CL_MAP_WRITE,
            cl::size_t<3>(),
            region,
            &rowPitch.confidence,
            nullptr, nullptr,
            &oclEvents.confidenceW);
    rowPitch.confidence /= sizeof(cl_float);

    oclEvents.imageW.wait();
    oclEvents.maskW.wait();
    oclEvents.confidenceW.wait();

    float conf = 0.0;
    cv::Point2i patchPos, fillPos;
    for (size_t dy=0; dy<patchWidth; dy++)
        for (size_t dx=0; dx<patchWidth; dx++) {
            patchPos = cv::Point2i(patchROI.x + dx, patchROI.y + dy);
            if (!maskBuffer[i2dIndex(patchPos, rowPitch.mask)])
                conf += confidenceBuffer[i2dIndex(patchPos, rowPitch.confidence)];
        }

    conf /= patchWidth * patchWidth;
    for (size_t dy=0; dy<patchWidth; dy++) {
        for (size_t dx=0; dx<patchWidth; dx++) {
            patchPos = cv::Point2i(patchROI.x + dx, patchROI.y + dy);
            fillPos = cv::Point2i(fillROI.x + dx, fillROI.y + dy);

            if (maskBuffer[i2dIndex(patchPos, rowPitch.mask)]) {
                imageBuffer[i2dIndex(patchPos, rowPitch.image)] =
                    imageBuffer[i2dIndex(fillPos, rowPitch.image)];
                confidenceBuffer[i2dIndex(patchPos, rowPitch.confidence)] = conf;
                maskBuffer[i2dIndex(patchPos, rowPitch.mask)] = 0;
            }
        }
    }

    commandQueue.enqueueUnmapMemObject(
            oclImage,
            imageBuffer,
            nullptr,
            &oclEvents.imageW);
    commandQueue.enqueueUnmapMemObject(
            oclMask,
            maskBuffer,
            nullptr,
            &oclEvents.maskW);
    commandQueue.enqueueUnmapMemObject(
            oclConfidence,
            confidenceBuffer,
            nullptr,
            &oclEvents.confidenceW);
}


cv::Mat OclInpainter::getResult() {
    oclEvents.imageW.wait();
    oclEvents.maskW.wait();
    oclEvents.confidenceW.wait();

    cv::Rect2i roi(cv::Point(0, 0), originalSize);
    cv::UMat result;
    cv::ocl::convertFromImage(oclImage(), result);

    cv::Mat resultMat = result.getMat(cv::ACCESS_READ);
    return resultMat(roi).clone();
}
