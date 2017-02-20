#include <iostream>
#include "inpainter.h"
#include "opencv2/highgui.hpp"

int main(int argc, char* argv[]) {
    const std::string commandLineDescribe = 
        "{ h help   |      | print help message }"
        "{ @image   |      | source image }"
        "{ @mask    |      | binary mask }"
        "{ c opencl |      | enable OpenCL }"
        "{ o output | output.png | output file name }";
    cv::CommandLineParser parser(argc, argv, commandLineDescribe);

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    struct {
        std::string sourceImage;
        std::string maskImage;
        std::string outputPath;
        bool oclMode;
    } arguments;

    arguments.sourceImage = parser.get<std::string>("@image");
    arguments.maskImage = parser.get<std::string>("@mask");
    arguments.outputPath = parser.get<std::string>("output");
    arguments.oclMode = parser.has("opencl");

    cv::Mat src;
    cv::Mat mask;
    cv::Mat result;
    
    src = cv::imread(arguments.sourceImage, cv::IMREAD_COLOR);
    mask = cv::imread(arguments.maskImage, cv::IMREAD_GRAYSCALE);

    AbstractInpainter *inpainter;
    
    if (arguments.oclMode)
        inpainter = new OclInpainter(src, mask);
    else
        inpainter = new CpuInpainter(src, mask);

    inpainter->inpaint();
    result = inpainter->getResult();

    cv::imwrite(arguments.outputPath, result);

    delete inpainter;

    return 0;
}
