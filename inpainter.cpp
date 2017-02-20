#include "inpainter.h"
#include <iostream>
#include <iomanip>

long AbstractInpainter::measureTimeInMs(bool init) {
    using namespace std::chrono;
    steady_clock::time_point current = steady_clock::now();

    if (init) {
        timestamp = current;
        return 0;
    }

    auto duration = current - timestamp;
    return duration_cast<microseconds>(duration).count();
}

void AbstractInpainter::inpaint() {
    float timeDetectFillFront = 0.0,
          timeFindBestMatching = 0.0,
          timeApplyPatch = 0.0;
    int count = 0;

    while (true) {
        measureTimeInMs(true);
        if (!detectFillFront()) break;
        timeDetectFillFront += measureTimeInMs();

        measureTimeInMs(true);
        findBestMatching();
        timeFindBestMatching += measureTimeInMs();

        measureTimeInMs(true);
        applyPatch();
        timeApplyPatch += measureTimeInMs();

        count++;
    }

    std::cerr << std::fixed << std::setprecision(2)
        << "round = " << count << std::endl
        << "detectFillFront: " << timeDetectFillFront / count << std::endl
        << "findBestMatching: " << timeFindBestMatching  / count << std::endl
        << "applyPatch: " << timeApplyPatch / count << std::endl;
}

