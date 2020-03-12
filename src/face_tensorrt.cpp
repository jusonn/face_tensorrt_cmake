#include "checkBlacklist.h"
#include <iostream>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include "printInfo.h"




//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("/root/optimization/TensorRT-7.0.0.11/data/mnist/");
        params.dataDirs.push_back("/root/optimization/data/TensorRT-7.0.0.11/data/samples/mnist/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.batchSize = 1;
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//checkBlacklist::checkBlacklist() {
//	m_ImageFormat = 0;
//	m_ImageWidth = 1280;
//	m_ImageHeight = 720;
//	m_StableRatio1 = 0.08;        // XY displacement threshold ratio
//	m_StableCount = 10;           // Stable frame count
//	m_StableRatio3 = 0.7;          // Usable detected IDcard size ratio
//	m_CardCheckerFlag = 0;
//	m_FaceRocation = 0;
//	m_LandmarkInputWidth = 64;
//	m_LandmarkInputHeight = 64;
//	m_PrevLocationY = int(m_ImageWidth * 0.5);
//	m_WarpSize = cv::Size(CARD_WIDTH * 2, CARD_HEIGHT * 2);
//	m_WarpCorners = new std::vector<cv::Point2f>[MAX_WARP_CORNER];
//	m_StabilizedCorner = new std::vector<cv::Point2f>[1];
//	m_StabilizedCorner[0] = std::vector<cv::Point2f>(4);
//	m_CardType = 0;
//
//
//    template <typename T>
//    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
//
//
//	for (int i = 0; i < MAX_CHEKER_STATE; i++) { m_CheckerResultState[i] = 0.0; }
//	for (int i = 0; i < MAX_WARP_CORNER; i++) { m_WarpCorners[i] = std::vector<cv::Point2f>(LANDMARK_POINT); }
//
//}

int main(int argc, char** argv)
{
    std::cout << "Hello World!\n";
//    checkBlacklist.Init();

    cv::Mat input_image = cv::imread("/root/optimization/face_tensorrt/drive_license.png",cv::IMREAD_COLOR);

    std::cout<< input_image.size()  <<std::endl;

    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printInfo::printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printInfo::printHelpInfo();
        return EXIT_SUCCESS;
    }
    const std::string gSampleName = "TensorRT.sample_onnx_mnist";

    //report the test result of defineTest
    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);
    gLogger.reportTestStart(sampleTest);

    //initialize the class to check Blacklist
    checkBlacklist checkBlacklist(initializeSampleParams(args));

    gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    if (!checkBlacklist.build())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!checkBlacklist.infer())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);





}



