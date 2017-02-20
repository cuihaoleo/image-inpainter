__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__constant float derivKernelA[PATCH_WIDTH] = DERIV_KERNEL_A;
__constant float derivKernelB[PATCH_WIDTH] = DERIV_KERNEL_B;


__kernel void detectFillFront(
        __read_only image2d_t image,
        __read_only image2d_t mask,
        __read_only image2d_t confidence,
        __global int2 *globalIndexRecords,
        __global float *globalPriorityRecords) {

    __local int2 indexRecords[LOCAL_SIZE_1D];
    __local float priorityRecords[LOCAL_SIZE_1D];

    const int groupIndex = mad24((int)get_group_id(1), (int)get_num_groups(0), (int)get_group_id(0));
    const int localIndex = mad24((int)get_local_id(1), (int)get_local_size(0), (int)get_local_id(0));
    const int globalIndex = mad24((int)get_global_id(1), (int)get_global_size(0), (int)get_global_id(0));
    const int localSize = get_local_size(0) * get_local_size(1);

    const int2 center = {
        get_global_id(0),
        get_global_id(1)};

    float4 pixel = read_imagef(mask, sampler, center);
    bool isFillFront = false;

    if (all(center >= 0 && center < (int2)(IMAGE_COLS, IMAGE_ROWS)) &&
            pixel.x == 0 && globalIndex < GLOBAL_SIZE_1D) {
        float4 sumPixel = \
            read_imagef(mask, sampler, center + (int2)(-1, 0)) +
            read_imagef(mask, sampler, center + (int2)( 1, 0)) +
            read_imagef(mask, sampler, center + (int2)( 0,-1)) +
            read_imagef(mask, sampler, center + (int2)( 0, 1));

        isFillFront = sumPixel.x > 0;
    }

    float priority = -INFINITY;
    if (isFillFront) {
        float conf;
        float2 dMask = 0.0, dImage = 0.0;

        conf = 0.0;

        for (int dy = 0; dy < PATCH_WIDTH; dy++) {
            for (int dx = 0; dx < PATCH_WIDTH; dx++) {
                int2 current = center + (int2)(dx - HALF_PATCH_WIDTH, dy - HALF_PATCH_WIDTH);

                float4 maskPix = read_imagef(mask, sampler, current);
                float4 imagePix = read_imagef(image, sampler, current);
                float4 confPix = read_imagef(confidence, sampler, current);

                float2 sobel = {
                    derivKernelA[dx] * derivKernelB[dy],
                    derivKernelA[dy] * derivKernelB[dx] };

                dMask += maskPix.x * sobel;
           
                float gray = dot(imagePix.xyz, (float3){0.299, 0.587, 0.114});
                dImage += gray * sobel;

                if (!maskPix.x)
                    conf += confPix.x;
            }
        }

        float data = fabs(dot(dMask, dImage));
        priority = data * conf;
    }

    priorityRecords[localIndex] = priority;
    indexRecords[localIndex] = center;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int offset = localSize >> 1; offset > 0; offset >>= 1) {
        if (localIndex < offset) {
            int otherIndex = localIndex + offset;
            if (priorityRecords[localIndex] < priorityRecords[otherIndex]) {
                priorityRecords[localIndex] = priorityRecords[otherIndex];
                indexRecords[localIndex] = indexRecords[otherIndex];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localIndex == 0) {
        globalIndexRecords[groupIndex] = indexRecords[0];
        globalPriorityRecords[groupIndex] = priorityRecords[0];
    }
}


__kernel void findBestMatching(
        __read_only image2d_t image,
        __read_only image2d_t mask,
        const int2 patchTopLeft,
        __global int2 *globalIndexRecords,
        __global float *globalErrorRecords) {
    __local int2 indexRecords[LOCAL_SIZE_1D];
    __local float errorRecords[LOCAL_SIZE_1D];

    const int groupIndex = mad24((int)get_group_id(1), (int)get_num_groups(0), (int)get_group_id(0));
    const int localIndex = mad24((int)get_local_id(1), (int)get_local_size(0), (int)get_local_id(0));
    const int localSize = get_local_size(0) * get_local_size(1);

    const int2 center = {
        get_global_id(0),
        get_global_id(1)};
    const int2 topLeft = {
        get_global_id(0) - HALF_PATCH_WIDTH,
        get_global_id(1) - HALF_PATCH_WIDTH};

    int px, py, pmx, pmy;
    float error = 0.0;

    for (int i = 0; i < PATCH_WIDTH; i++) {
        px = topLeft.x + i;
        pmx = patchTopLeft.x + i;

        for (int j = 0; j < PATCH_WIDTH; j++) {
            py = topLeft.y + j;
            pmy = patchTopLeft.y + j;

            float4 maskPix = read_imagef(mask, sampler, (int2)(px, py));
            float4 patchMaskPix = read_imagef(mask, sampler, (int2)(pmx, pmy));

            if (maskPix.x) {
                error = HUGE_VALF;
                break;
            } else if (!patchMaskPix.x) {
                float4 fillPixel = read_imagef(image, sampler, (int2)(px, py));
                float4 patchPixel = read_imagef(image, sampler, (int2)(pmx, pmy));
                float norm = fast_length(fillPixel - patchPixel);
                error += norm * norm;
            }
        }

        if (!isfinite(error))
            break;
    }

    errorRecords[localIndex] = error;
    indexRecords[localIndex] = topLeft;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int offset = localSize >> 1; offset > 0; offset >>= 1) {
        if (localIndex < offset) {
            int otherIndex = localIndex + offset;
            if (errorRecords[localIndex] > errorRecords[otherIndex]) {
                errorRecords[localIndex] = errorRecords[otherIndex];
                indexRecords[localIndex] = indexRecords[otherIndex];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localIndex == 0) {
        globalIndexRecords[groupIndex] = indexRecords[0];
        globalErrorRecords[groupIndex] = errorRecords[0];
    }
}
