#pragma once
//
// Created by dylee on 2020-03-10.
//

#ifndef FACE_TENSORRT_BLACKLISTDETECTION_H
#define FACE_TENSORRT_BLACKLISTDETECTION_H

#endif //FACE_TENSORRT_BLACKLISTDETECTION_H

#include "stdio.h"
//#include "arcface.h"
//#include "base.h"
//#include "mtcnn.h"
//#include "util.h"
//#include "net.h"
#include <experimental/filesystem>

#include "opencv2/opencv.hpp"

#define PI 3.141592

class BlackListDetection{
public:
    BlackListDetection();
    ~BlackListDetection();

    void Init();
    cv::Mat IDCardRegistration(cv::Mat input);
    void MakeBlackListFeatures(std::string path);
    void LoadBlackListFeatures(std::string path);
    std::vector<int> CompareIDCard2BlackList(cv::Mat id_image, float threshold);

//    MtcnnDetector detector;
//    Arcface arc;
//    ncnn::Net m_LandmarkNet;
//    ncnn::Net m_FaceFeatureNet;

private:
    cv::Mat m_InputImage;
    cv::Mat m_IDCardImage;
    std::vector<std::vector<float>> m_BlackListFeature;
    std::vector<std::string> m_BlackListFileName;
    float* m_Similarity2BlackList;

//    cv::Mat ncnn2cv(ncnn::Mat img);
};

