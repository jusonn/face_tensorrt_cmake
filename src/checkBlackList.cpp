//
// Created by dylee on 2020-03-10.
//

//#include "blackListDetection.h"

//BlackListDetection bld;

#include "checkBlacklist.h"
#include <iostream>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include "logger.h"

bool checkBlacklist::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    assert(mOutputDims.nbDims == 2);
    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool checkBlacklist::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                                       SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                                       SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(
            locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(16_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool checkBlacklist::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool checkBlacklist::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    mNumber = rand() % 10;
    readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // Print an ascii representation
    gLogInfo << "Input:" << std::endl;
    for (int i = 0; i < inputH * inputW; i++)
    {
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    gLogInfo << std::endl;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < inputH * inputW; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    return true;
}



//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool checkBlacklist::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[1];
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0f};
    int idx{0};

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }

        gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
                 << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
    }
    gLogInfo << std::endl;

    return idx == mNumber && val > 0.9f;
}


//checkBlacklist::checkBlacklist() {
//    m_ImageFormat = 0;
//    m_ImageWidth = 1280;
//    m_ImageHeight = 720;
//    m_StableRatio1 = 0.08;        // XY displacement threshold ratio
//    m_StableCount = 10;           // Stable frame count
//    m_StableRatio3 = 0.7;          // Usable detected IDcard size ratio
//    m_CardCheckerFlag = 0;
//    m_FaceRocation = 0;
//    m_LandmarkInputWidth = 64;
//    m_LandmarkInputHeight = 64;
//    m_PrevLocationY = int(m_ImageWidth * 0.5);
//    m_WarpSize = cv::Size(CARD_WIDTH * 2, CARD_HEIGHT * 2);
//    m_WarpCorners = new std::vector<cv::Point2f>[MAX_WARP_CORNER];
//    m_StabilizedCorner = new std::vector<cv::Point2f>[1];
//    m_StabilizedCorner[0] = std::vector<cv::Point2f>(4);
//    m_CardType = 0;
//    for (int i = 0; i < MAX_CHEKER_STATE; i++) { m_CheckerResultState[i] = 0.0; }
//    for (int i = 0; i < MAX_WARP_CORNER; i++) { m_WarpCorners[i] = std::vector<cv::Point2f>(LANDMARK_POINT); }
//
//}


checkBlacklist::~checkBlacklist() {
//    m_LandmarkNet.clear();
//    m_ConditionCheckNet.clear();
//    m_CardCheckNet.clear();
}

int checkBlacklist::Init(int imageFormat, int imageWidth, int imageHeight, int rotateDegree, float imageMargin, float stableRatio1, int stableCount, std::string modelsPath)
{
//#ifdef PRINT_LOG
//    LOGD("checkBlackList Initialization Start")
//#endif
//    std::string
}







