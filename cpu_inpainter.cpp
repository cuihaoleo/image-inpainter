#include "inpainter.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>

CpuInpainter::CpuInpainter(
        cv::Mat inputImage,
        cv::Mat mask,
        size_t halfPatchWidth) {
    cv::Mat gray, tmp;

    if (inputImage.type() != CV_8UC3) {
        throw std::runtime_error("inputImage.type() != CV_8UC3");
    }

    if (mask.type() != CV_8UC1) {
        throw std::runtime_error("inputImage.type() != CV_8UC1");
    }

    this->image = inputImage.clone();

    cv::cvtColor(this->image, tmp, cv::COLOR_BGR2GRAY);
    tmp.convertTo(gray, CV_32F, 1/255.0);
    tmp.release();

    this->mask = mask.clone();
    this->patchWidth = 2 * halfPatchWidth + 1;

    cv::getDerivKernels(derivKernelA, derivKernelB, 1, 0, patchWidth);

    cv::threshold(this->mask, confidence, 0, 1, cv::THRESH_BINARY_INV);
    confidence.convertTo(confidence, CV_32F);

    cv::Mat gradientX, gradientY;
    //cv::Scharr(gray, gradientX, CV_32F, 1, 0);
    cv::Sobel(gray, gradientX, CV_32F, 1, 0, patchWidth);
    //cv::Scharr(gray, gradientY, CV_32F, 0, 1);
    cv::Sobel(gray, gradientY, CV_32F, 0, 1, patchWidth);

    std::vector<cv::Mat> gradientArray = { gradientX, gradientY };
    cv::merge(gradientArray, gradient);
}


bool CpuInpainter::detectFillFront() {
    cv::Rect whole = cv::Rect(0, 0, image.cols, image.rows);
    float maxPriority = -std::numeric_limits<float>::infinity();

    cv::Mat sobelKernelX, sobelKernelY;
    cv::gemm(derivKernelB, derivKernelA, 1.0, cv::noArray(), 0.0, sobelKernelX, cv::GEMM_2_T);
    cv::gemm(derivKernelA, derivKernelB, 1.0, cv::noArray(), 0.0, sobelKernelY, cv::GEMM_2_T);

    for (int y=0; y<mask.rows; y++) {
        for (int x=0; x<mask.cols; x++) {
            cv::Point p = cv::Point(x, y);

            if (mask.at<uchar>(p) == 0)
                continue;

            cv::Point neigh[4] = {
                cv::Point(p.x-1, p.y),
                cv::Point(p.x+1, p.y),
                cv::Point(p.x, p.y-1),
                cv::Point(p.x, p.y+1)
            };

            for (auto center: neigh)
                if (whole.contains(center) && mask.at<uchar>(center) == 0) {
                    float priority = std::numeric_limits<float>::max();
                    float conf = 0;
                    cv::Rect roi(
                            center.x - halfPatchWidth(),
                            center.y - halfPatchWidth(),
                            patchWidth,
                            patchWidth);

                    if ((roi & whole) == roi) {
                        cv::Mat patchMaskU8 = mask(roi).clone();
                        cv::Mat patchMask(patchMaskU8.size(), CV_32F);
                        patchMaskU8.convertTo(patchMask, CV_32F, 1/255.0);

                        cv::Mat patchConfidence;
                        cv::bitwise_not(patchMaskU8, patchMaskU8);
                        confidence(roi).copyTo(patchConfidence, patchMaskU8);

                        float nx = sobelKernelX.dot(patchMask);
                        float ny = sobelKernelY.dot(patchMask);
                        cv::Vec2f grad = gradient.at<cv::Vec2f>(center);
                        float data = std::fabs(nx * grad[0] + ny * grad[1]);
                        conf = cv::sum(patchConfidence)[0];
                        priority = data * conf;
                    }

                    if (priority > maxPriority) {
                        nextConfidence = conf;
                        maxPriority = priority;
                        patchCenter = center;
                    }
                }
        }
    }

    if (maxPriority < 0) {
        return false;
    }

    patchROI = cv::Rect(
            patchCenter.x - halfPatchWidth(), 
            patchCenter.y - halfPatchWidth(),
            patchWidth,
            patchWidth);
    std::cerr << patchROI << std::endl;
    std::cerr << "priority: " << maxPriority << std::endl;

    return true;
}


void CpuInpainter::findBestMatching() {
    cv::Point topLeft;
    cv::Mat patchImage = image(patchROI);
    cv::Mat patchMask = mask(patchROI);
    float minError = std::numeric_limits<float>::infinity();

    for (int x = 0; x + (int)patchWidth < image.cols; x++)
        for (int y = 0; y + (int)patchWidth < image.rows; y++) {
            bool skip = false;
            cv::Rect roi(x, y, patchWidth, patchWidth);
            cv::Mat fillImage = image(roi);
            cv::Mat fillMask = mask(roi);
            float error = 0.0;

            for (size_t px = 0; !skip && px < patchWidth; px++)
                for (size_t py = 0; py < patchWidth; py++) {
                    if (fillMask.at<uchar>(py, px)) {
                        skip = true;
                        break;
                    }

                    if (patchMask.at<uchar>(py, px))
                        continue;

                    cv::Vec3f fillPixel = fillImage.at<cv::Vec3b>(py, px);
                    cv::Vec3f patchPixel = patchImage.at<cv::Vec3b>(py, px);
                    cv::Vec3f diff = patchPixel - fillPixel;
                    float norm = cv::norm(diff);
                    error += norm * norm;
                }

            if (skip)
                continue;

            if (error < minError) {
                minError = error;
                topLeft = cv::Point(x, y);
            }
        } 

    fillROI = cv::Rect(topLeft.x, topLeft.y, patchWidth, patchWidth);
    std::cerr << fillROI << std::endl;
    std::cerr << "error: " << minError << std::endl;
}

void CpuInpainter::applyPatch() {
    cv::Mat patchImage = image(patchROI);
    cv::Mat patchMask = mask(patchROI);
    cv::Mat patchGradient = gradient(patchROI);
    cv::Mat patchConfidence = confidence(patchROI);

    cv::Mat fillImage = image(fillROI);
    cv::Mat fillMask = mask(fillROI);
    cv::Mat fillGradient = gradient(fillROI);
 
    fillImage.copyTo(patchImage, patchMask);
    fillGradient.copyTo(patchGradient, patchMask);
    patchConfidence.setTo(
            nextConfidence / (patchWidth * patchWidth),
            patchMask);
    patchMask.setTo(0);
}

cv::Mat CpuInpainter::getResult() {
    return image;
}
