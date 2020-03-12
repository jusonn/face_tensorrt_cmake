//
// Created by dylee on 2020-03-10.
//
#include "blackListDetection.h"

#ifndef _MAX_PATH
#define _MAX_PATH 260
#endif

BlackListDetection::BlackListDetection()
{

}

BlackListDetection::~BlackListDetection()
{

}

void BlackListDetection::Init()
{

}

void BlackListDetection::IDCardRegistration(cv::Mat input)
{
    std::cout<<"input rows: "<<input.dims<<std::endl;
    m_InputImage == input.clone();

    cv::Mat input_gray;
    cv::Mat input_roi;
    cv::Mat input_roi_color;

    input_gray = m_InputImage.clone();
    std::cout<<"check 111"<<std::endl;

    int width = input_gray.cols;
    int height = input_gray.rows;
    int roi_size = 0;
    int roi_startX = 0;
    int roi_startY = 0;

    if (width>height){
        std::cout<<"check 222"<<std::endl;
        roi_size = height;
        roi_startX = 0;
        roi_startY = width /2 -height/2;
        int padd_size = (width-height) /2;
        cv::Mat padd_image;
        cv::Mat padd_image_color;
        cv::copy
    }
    else
    {
        roi_size=width;
        roi_startX=0;
        roi_startY=height/2-width/2;
        input_startY = input_gray(cv::Rect(0,roi_startX,roi_size,roi_size));
        input_roi_color = m_InputImage(cv::Rect(0,roi_startY,roi_size,roi_size));
    }

    std::cout<<"check 333"<<std::endl;

    cv::Mat gray_resized;
    std::vector<cv::Point2f> *WarpCorner = new std::vector<cv::Point2f>[1];
    WarpCorner[0] = std::vector<cv::Point2f>(4);








}