// python_binding.cpp - pybind11 wrapper for CUDA SIFT
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include "cudaImage.h"
#include "cudaSift.h"

namespace py = pybind11;

class PyCudaSift {
private:
    bool initialized;
    int devNum;
    
public:
    PyCudaSift(int device_num = 0) : devNum(device_num), initialized(false) {
        InitCuda(devNum);
        initialized = true;
        
    }
    
    ~PyCudaSift() {
        // Cleanup if needed
    }
    
    // Extract SIFT features from numpy array
    py::dict extract_features(py::array_t<float> input_array, 
                             int max_features = 32768,
                             float init_blur = 1.0f,
                             float thresh = 3.0f,
                             int octaves = 5) {
        
        py::buffer_info buf_info = input_array.request();
        
        if (buf_info.ndim != 2) {
            throw std::runtime_error("Input array must be 2-dimensional");
        }
        
        int height = buf_info.shape[0];
        int width = buf_info.shape[1];
        float* data = static_cast<float*>(buf_info.ptr);
        
        // Initialize CUDA image
        CudaImage img;
        img.Allocate(width, height, iAlignUp(width, 128), false, NULL, data);
        img.Download();
        
        // Initialize SIFT data
        SiftData siftData;
        InitSiftData(siftData, max_features, true, true);
        
        // Allocate temporary memory
        float* memoryTmp = AllocSiftTempMemory(width, height, octaves, false);
        
        // Extract SIFT features
        ExtractSift(siftData, img, octaves, init_blur, thresh, 0.0f, false, memoryTmp);
        
        // Convert results to numpy arrays
        int numPts = siftData.numPts;
        
        // Prepare output arrays
        auto keypoints = py::array_t<float>({numPts, 2});
        auto descriptors = py::array_t<float>({numPts, 128});
        auto scales = py::array_t<float>(numPts);
        auto orientations = py::array_t<float>(numPts);
        auto scores = py::array_t<float>(numPts);
        
        py::buffer_info kp_buf = keypoints.request();
        py::buffer_info desc_buf = descriptors.request();
        py::buffer_info scale_buf = scales.request();
        py::buffer_info orient_buf = orientations.request();
        py::buffer_info score_buf = scores.request();
        
        float* kp_ptr = static_cast<float*>(kp_buf.ptr);
        float* desc_ptr = static_cast<float*>(desc_buf.ptr);
        float* scale_ptr = static_cast<float*>(scale_buf.ptr);
        float* orient_ptr = static_cast<float*>(orient_buf.ptr);
        float* score_ptr = static_cast<float*>(score_buf.ptr);
        
        // Copy data from SIFT results
#ifdef MANAGEDMEM
        SiftPoint* sift_points = siftData.m_data;
#else
        SiftPoint* sift_points = siftData.h_data;
#endif
        
        for (int i = 0; i < numPts; i++) {
            kp_ptr[i * 2] = sift_points[i].xpos;
            kp_ptr[i * 2 + 1] = sift_points[i].ypos;
            scale_ptr[i] = sift_points[i].scale;
            orient_ptr[i] = sift_points[i].orientation;
            score_ptr[i] = sift_points[i].score;
            
            // Copy descriptor
            for (int j = 0; j < 128; j++) {
                desc_ptr[i * 128 + j] = sift_points[i].data[j];
            }
        }
        
        // Cleanup
        FreeSiftTempMemory(memoryTmp);
        FreeSiftData(siftData);
        
        // Return dictionary with results
        py::dict result;
        result["keypoints"] = keypoints;
        result["descriptors"] = descriptors;
        result["scales"] = scales;
        result["orientations"] = orientations;
        result["scores"] = scores;
        result["num_features"] = numPts;
        
        return result;
    }
};

// OpenCV Mat to numpy conversion helper
py::array_t<float> mat_to_numpy(const cv::Mat& mat) {
    return py::array_t<float>(
        {mat.rows, mat.cols},
        {sizeof(float) * mat.cols, sizeof(float)},
        mat.ptr<float>(),
        py::cast(mat)
    );
}

// Convenience function for OpenCV integration
py::dict extract_sift_opencv(py::object img_obj, 
                            int max_features = 32768,
                            float init_blur = 1.0f,
                            float thresh = 3.0f,
                            int octaves = 5,
                            int device_num = 0) {
    
    // Convert from OpenCV Mat (assuming it's passed as numpy array)
    py::array_t<uint8_t> input = img_obj.cast<py::array_t<uint8_t>>();
    py::buffer_info buf_info = input.request();
    
    int height = buf_info.shape[0];
    int width = buf_info.shape[1];
    uint8_t* data = static_cast<uint8_t*>(buf_info.ptr);
    
    // Convert to float
    cv::Mat img_uint8(height, width, CV_8UC1, data);
    cv::Mat img_float;
    img_uint8.convertTo(img_float, CV_32FC1);
    
    // Convert to numpy array
    auto float_array = mat_to_numpy(img_float);
    
    // Extract features
    PyCudaSift extractor(device_num);
    return extractor.extract_features(float_array, max_features, init_blur, thresh, octaves);
}

PYBIND11_MODULE(cuda_sift, m) {
    m.doc() = "CUDA SIFT feature extraction and matching";
    
    py::class_<PyCudaSift>(m, "CudaSift")
        .def(py::init<int>(), py::arg("device_num") = 0)
        .def("extract_features", &PyCudaSift::extract_features,
             py::arg("image"), py::arg("max_features") = 32768,
             py::arg("init_blur") = 1.0f, py::arg("thresh") = 3.0f,
             py::arg("octaves") = 5,
             "Extract SIFT features from image");
    
    // Convenience function
    m.def("extract_sift", &extract_sift_opencv,
          py::arg("image"), py::arg("max_features") = 32768,
          py::arg("init_blur") = 1.0f, py::arg("thresh") = 3.0f,
          py::arg("octaves") = 5, py::arg("device_num") = 0,
          "Extract SIFT features from OpenCV image");
}