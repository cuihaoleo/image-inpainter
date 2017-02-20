# 双学位毕业设计

双学位毕业设计项目：移动设备上基于 OpenCL 的 GPGPU 并行图像处理算法设计。

实现了 CPU 串行和 OpenCL 并行版本的图像填补（image inpainting）算法。代码都是自己手撸的，MIT 协议开源。

处理效果一般，可能是算法本身太弱，也可能是哪里有 bug。不过课题主要是研究算法优化的，处理效果不是重点考虑的问题……

算法主要参考：
- Criminisi A, Perez P, Toyama K. Object removal by exemplar-based inpainting[C]//Computer Vision and Pattern Recognition, 2003. Proceedings. 2003 IEEE Computer Society Conference on. IEEE, 2003, 2: II-II.
- Wang G, Xiong Y, Yun J, et al. Computer vision accelerators for mobile systems based on OpenCL GPGPU co-processing[J]. Journal of Signal Processing Systems, 2014, 76(3): 283-299.

## 编译

依赖 OpenCV 库和 OpenCL 头文件。

桌面上编译：
```
g++ $(pkg-config --libs opencv) -lOpenCL cpu_inpainter.cpp ocl_inpainter.cpp inpainter.cpp main.cpp -oinpainter
```

请参考 `Android.mk` 和 `Application.mk` 编译 Android 可执行文件。


## 针对 Qualcomm GPU 的优化

编译时启用 `USE_QCOM_EXT` 宏可以开启 Qualcomm 的 GPU/CPU 内存共享的 OpenCL 扩展。

