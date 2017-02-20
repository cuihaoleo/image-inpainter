ifneq ($(TARGET_SIMULATOR),true)

LOCAL_PATH:= $(call my-dir)

# libOpenCL
include $(CLEAR_VARS)

LOCAL_MODULE := OpenCL-prebuilt
LOCAL_SRC_FILES := libOpenCL.so

include $(PREBUILT_SHARED_LIBRARY)


include $(CLEAR_VARS)

OPENCV_LIB_TYPE := static
include $(LOCAL_PATH)/opencv-native/jni/OpenCV.mk

LOCAL_MODULE := inpainter
LOCAL_SRC_FILES:= main.cpp inpainter.cpp cpu_inpainter.cpp ocl_inpainter.cpp ocl_inpainter_ion.cpp

LOCAL_SHARED_LIBRARIES := OpenCL-prebuilt
LOCAL_CPP_FEATURES := exceptions
LOCAL_CFLAGS += -Wall -std=c++11 -O2
LOCAL_LDLIBS := -Wl,-unresolved-symbols=ignore-in-shared-libs -L$(LOCAL_PATH)/lib -llog -lm -lz
LOCAL_C_INCLUDES := bionic
LOCAL_C_INCLUDES := $(LOCAL_PATH)/include

include $(BUILD_EXECUTABLE)


endif  # TARGET_SIMULATOR != true
