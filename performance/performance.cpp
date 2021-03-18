// @Author: Junaid Afzal

#include <iostream>
#include <fstream>
#include <math.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/dnn.hpp"
#include "../src/rollingAverage.h"



int main(void)
{
	auto begin = std::chrono::high_resolution_clock::now();

	////////////////////////////////////////////////////
	//
	// First lane detection code will be ran without
	// YOLOv4 to see performance impact of YOLOv4
	//
	// Then three variables will be tested:
	// 	   - YOLOv4 vs YOLOv4-tiny
	// 	   - Blob Sizes (608, 512, 416, 320, 288)
	// 	   - CUDA vs CPU
	//
	////////////////////////////////////////////////////

	int numberOfRepetitions = 10;
	int numberOfTests = 21;
	int frameCount = 1155;

	//std::string platformDir = "../tests/Windows 10 Desktop/";
	//std::string platform = "../tests/Linux (Ubuntu 20.04) Desktop/";
	//std::string platform = "../tests/Jetson Nano (4GB)/";

	bool saveResults = true;

	bool recordOuput = false;
	int OuputFPS = 30;
	std::string outputVideoDir = "../tests/All platforms/output vids";


	// Variables that will change
	std::vector<bool> yolo =
	{
		false,
		true,
		true,
		true,
		true,
		true,
		true,
		true,
		true,
		true,
		true,
		true,
		true,
		true,
		true,
		true,
		true,
		true,
		true,
		true,
		true
	};

	std::vector<std::string> yoloTypeConfig =
	{
		"",
		"../yolo/yolov4.cfg",
		"../yolo/yolov4.cfg",
		"../yolo/yolov4.cfg",
		"../yolo/yolov4.cfg",
		"../yolo/yolov4.cfg",
		"../yolo/yolov4-tiny.cfg",
		"../yolo/yolov4-tiny.cfg",
		"../yolo/yolov4-tiny.cfg",
		"../yolo/yolov4-tiny.cfg",
		"../yolo/yolov4-tiny.cfg",
		"../yolo/yolov4.cfg",
		"../yolo/yolov4.cfg",
		"../yolo/yolov4.cfg",
		"../yolo/yolov4.cfg",
		"../yolo/yolov4.cfg",
		"../yolo/yolov4-tiny.cfg",
		"../yolo/yolov4-tiny.cfg",
		"../yolo/yolov4-tiny.cfg",
		"../yolo/yolov4-tiny.cfg",
		"../yolo/yolov4-tiny.cfg"
	};

	std::vector<std::string> yoloTypeWeights =
	{
		"",
		"../yolo/yolov4.weights",
		"../yolo/yolov4.weights",
		"../yolo/yolov4.weights",
		"../yolo/yolov4.weights",
		"../yolo/yolov4.weights",
		"../yolo/yolov4-tiny.weights",
		"../yolo/yolov4-tiny.weights",
		"../yolo/yolov4-tiny.weights",
		"../yolo/yolov4-tiny.weights",
		"../yolo/yolov4-tiny.weights",
		"../yolo/yolov4.weights",
		"../yolo/yolov4.weights",
		"../yolo/yolov4.weights",
		"../yolo/yolov4.weights",
		"../yolo/yolov4.weights",
		"../yolo/yolov4-tiny.weights",
		"../yolo/yolov4-tiny.weights",
		"../yolo/yolov4-tiny.weights",
		"../yolo/yolov4-tiny.weights",
		"../yolo/yolov4-tiny.weights"
	};

	std::vector<int> dnnBackend =
	{
		-1, //"",
		5,  //"cv::dnn::DNN_BACKEND_CUDA",
		5,	//"cv::dnn::DNN_BACKEND_CUDA",
		5,	//"cv::dnn::DNN_BACKEND_CUDA",
		5,	//"cv::dnn::DNN_BACKEND_CUDA",
		5,	//"cv::dnn::DNN_BACKEND_CUDA",
		5,	//"cv::dnn::DNN_BACKEND_CUDA",
		5,	//"cv::dnn::DNN_BACKEND_CUDA",
		5,	//"cv::dnn::DNN_BACKEND_CUDA",
		5,	//"cv::dnn::DNN_BACKEND_CUDA",
		5,	//"cv::dnn::DNN_BACKEND_CUDA",
		3,	//"cv::dnn::DNN_BACKEND_OPENCV",
		3,	//"cv::dnn::DNN_BACKEND_OPENCV",
		3,	//"cv::dnn::DNN_BACKEND_OPENCV",
		3,	//"cv::dnn::DNN_BACKEND_OPENCV",
		3,	//"cv::dnn::DNN_BACKEND_OPENCV",
		3,	//"cv::dnn::DNN_BACKEND_OPENCV",
		3,	//"cv::dnn::DNN_BACKEND_OPENCV",
		3,	//"cv::dnn::DNN_BACKEND_OPENCV",
		3,	//"cv::dnn::DNN_BACKEND_OPENCV",
		3	//"cv::dnn::DNN_BACKEND_OPENCV"
	};

	std::vector<int> dnnTarget =
	{
		-1,	//"",
		6,	//"cv::dnn::DNN_TARGET_CUDA",
		6,	//"cv::dnn::DNN_TARGET_CUDA",
		6,	//"cv::dnn::DNN_TARGET_CUDA",
		6,	//"cv::dnn::DNN_TARGET_CUDA",
		6,	//"cv::dnn::DNN_TARGET_CUDA",
		6,	//"cv::dnn::DNN_TARGET_CUDA",
		6,	//"cv::dnn::DNN_TARGET_CUDA",
		6,	//"cv::dnn::DNN_TARGET_CUDA",
		6,	//"cv::dnn::DNN_TARGET_CUDA",
		6,	//"cv::dnn::DNN_TARGET_CUDA",
		0,	//"cv::dnn::DNN_TARGET_CPU",
		0,	//"cv::dnn::DNN_TARGET_CPU",
		0,	//"cv::dnn::DNN_TARGET_CPU",
		0,	//"cv::dnn::DNN_TARGET_CPU",
		0,	//"cv::dnn::DNN_TARGET_CPU",
		0,	//"cv::dnn::DNN_TARGET_CPU",
		0,	//"cv::dnn::DNN_TARGET_CPU",
		0,	//"cv::dnn::DNN_TARGET_CPU",
		0,	//"cv::dnn::DNN_TARGET_CPU",
		0	//"cv::dnn::DNN_TARGET_CPU"
	};

	std::vector<int> blobSize =
	{
		0,
		608,
		512,
		416,
		320,
		288,
		608,
		512,
		416,
		320,
		288,
		608,
		512,
		416,
		320,
		288,
		608,
		512,
		416,
		320,
		288
	};

	std::vector<std::string> outputFileNames;

	for (int i = 0; i < numberOfTests; i++)
	{
		std::string temp = "";

		if (yolo[i])
		{
			if (yoloTypeConfig[i] == "../yolo/yolov4.cfg")
				temp += "YOLOv4 ";
			else
				temp += "YOLOv4-tiny ";

			temp += std::to_string(blobSize[i]);

			if (dnnBackend[i] == 5)
				temp += " with CUDA";
			else if (dnnBackend[i] == 3)
				temp += " without CUDA";
		}
		else
			temp += "No YOLOv4";

		outputFileNames.push_back(temp);
	}

	cv::VideoWriter frameVideo;

	std::cout << "Number of tests: " << numberOfTests;
	std::cout << "\nNumber of repitions: " << numberOfRepetitions << '\n';

	for (int testNumber = 0; testNumber < numberOfTests; testNumber++)
	{
		std::vector<std::vector<int> > allIterations;

		for (int iterations = 0; iterations < numberOfRepetitions; iterations++)
		{
			std::cout << "\nCurrently on test " << testNumber+1 << "/" << numberOfTests << "\t"  << outputFileNames[testNumber] << " (" << iterations+1 << "/" << numberOfRepetitions << ")";
			std::cout.flush();

			std::vector<int> temp;

			if (recordOuput && iterations == 0)
				frameVideo.open(outputVideoDir + outputFileNames[testNumber] + ".mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), OuputFPS, cv::Size(1920, 1080), true);



			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



			// Create a VideoCapture object and open the input video file
			cv::VideoCapture video("../vids/benchmark.mp4");
			video.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
			video.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
			// Check if camera opened successfully
			if (!video.isOpened()) {
				std::cout << "\nError opening video stream or file\n";
				return -1;
			}

			// Read in the coco names
			// The std::map links model ID with a string and a string with a colour
			std::map<int, std::string> modelIntsAndNames;
			std::map<std::string, cv::Scalar> modelNamesAndColourList;
			std::ifstream modelNamesFile("../yolo/coco.names");
			srand(255);
			if (modelNamesFile.is_open())
			{
				std::string line;
				for (int i = 0; std::getline(modelNamesFile, line); i++)
				{
					#ifdef __linux__
					line.pop_back();
					#endif
					modelNamesAndColourList.insert(std::pair<std::string, cv::Scalar>(line, cv::Scalar(rand() % 256, rand() % 256, rand() % 256)));
					modelIntsAndNames.insert(std::pair<int, std::string>(i, line));
				}

				// Set these as custom colours
				modelNamesAndColourList["car"] = cv::Scalar(255, 64, 64);           // blue
				modelNamesAndColourList["truck"] = cv::Scalar(255, 0, 255);         // red
				modelNamesAndColourList["bus"] = cv::Scalar(255, 255, 255);         // white
				modelNamesAndColourList["traffic light"] = cv::Scalar(0, 0, 255);   // greeny

				modelNamesFile.close();
			}
			else
			{
				std::cout << "\nError opening coco.names file stream or file\n";
				return -2;
			}


			cv::dnn::Net net;
			std::vector<std::string> unconnectedOutLayersNames;
			if (yolo[testNumber])
			{
				// Setup the YOLO CUDA OpenCV DNN
				net = cv::dnn::readNetFromDarknet(yoloTypeConfig[testNumber], yoloTypeWeights[testNumber]);
				net.setPreferableBackend(dnnBackend[testNumber]);
				net.setPreferableTarget(dnnTarget[testNumber]);
				unconnectedOutLayersNames = net.getUnconnectedOutLayersNames();
			}



			// Hard-Coded Parameters
			// Lane Region of interest (ROI)
			constexpr int ROI_TOP_HEIGHT = 660;
			constexpr int ROI_BOTTOM_HEIGHT = 840;
			constexpr int ROI_TOP_WIDTH = 200;
			constexpr int ROI_BOTTOM_WIDTH = 900;

			// Canny edge detection variables
			constexpr int CANNY_LOWER_THRESHOLD = 64;
			constexpr int CANNY_UPPER_THRESHOLD = 128;

			// Canny edge detection variables
			constexpr int HOUGHP_THRESHOLD = 32;
			constexpr int HOUGHP_MIN_LINE_LENGTH = 16;
			constexpr int HOUGHP_MAX_LINE_GAP = 8;

			// Lines lower than this will be horizontal
			constexpr double HORIZONTAL_GRADIENT_THRESHOLD = 0.15;
			// Lines longer than this will be horizontal
			constexpr int HORIZONTAL_LENGTH_THRESHOLD = 50;
			// Line count larger than this will be horizontal
			constexpr int HORIZONTAL_COUNT_THRESHOLD = 10;

			// Lines longer than this will be solid
			constexpr int SOLID_LINE_LENGTH_THRESHOLD = 75;

			// How big of a rolling average to use
			constexpr int HORIZONTAL_LINE_STATE_ROLLING_AVERAGE = 10;
			constexpr int LINE_STATE_ROLLING_AVERAGE = 10;
			constexpr int DRIVING_STATE_ROLLING_AVERAGE = 10;

			// How many frames to wait for until trying to find a difference in car position
			// to determine whether the car is moving left or right. NOTE that this is heavily
			// dependent upon frame rate
			int VIDEO_WIDTH = video.get(cv::CAP_PROP_FRAME_WIDTH);
			int VIDEO_HEIGHT = video.get(cv::CAP_PROP_FRAME_HEIGHT);
			int VIDEO_FPS = video.get(cv::CAP_PROP_FPS);
			int FRAME_COUNT_THRESHOLD = VIDEO_FPS / 3.;

			// YOLO confidence threshold, non-maxima suppression threshold and number of
			// objects that can be detected
			int BLOB_SIZE = blobSize[testNumber];
			constexpr double YOLO_CONFIDENCE_THRESHOLD = 0;
			constexpr double YOLO_NMS_THRESHOLD = 0.4;
			constexpr int BOUNDING_BOX_BUFFER = 5;

			// Font variables
			constexpr int FONT_FACE = cv::FONT_HERSHEY_DUPLEX;
			constexpr double FONT_SCALE = 1;
			constexpr int FONT_THICKNESS = 1;



			// YOLO Variables
			std::vector<cv::Mat> outputBlobs;
			std::vector<cv::Rect> objectBoundingBoxes, preNMSObjectBoundingBoxes;
			std::vector<std::string> objectNames, preNMSObjectNames;
			std::vector<float> objectConfidences, preNMSObjectConfidences;
			double centerX, centerY, width, height, confidence;
			cv::Point classID;
			std::vector<int> indicesAfterNMS;
			std::string trafficLightState;

			// Mat objects
			cv::Mat frame, blobFromImg, ROIFrame, cannyFrame, blankFrame;

			// rolling averages
			rollingAverage horizontalLineStateRollingAverage(HORIZONTAL_LINE_STATE_ROLLING_AVERAGE, 2);
			rollingAverage leftLineTypeRollingAverage(LINE_STATE_ROLLING_AVERAGE, 3);
			rollingAverage middleLineTypeRollingAverage(LINE_STATE_ROLLING_AVERAGE, 3);
			rollingAverage rightLineTypeRollingAverage(LINE_STATE_ROLLING_AVERAGE, 3);
			rollingAverage drivingStateRollingAverage(DRIVING_STATE_ROLLING_AVERAGE, 5);

			// houghProbabilisticLines will hold the results of the Hough line detection
			std::vector<cv::Vec4i> houghLines;
			std::vector<cv::Point> lanePoints;

			// Calculate the mask dimensions
			std::vector<cv::Point> maskDimensions;
			maskDimensions.push_back(cv::Point(VIDEO_WIDTH / 2 - ROI_TOP_WIDTH / 2, ROI_TOP_HEIGHT));
			maskDimensions.push_back(cv::Point(VIDEO_WIDTH / 2 + ROI_TOP_WIDTH / 2, ROI_TOP_HEIGHT));
			maskDimensions.push_back(cv::Point(VIDEO_WIDTH / 2 + ROI_BOTTOM_WIDTH / 2, ROI_BOTTOM_HEIGHT));
			maskDimensions.push_back(cv::Point(VIDEO_WIDTH / 2 - ROI_BOTTOM_WIDTH / 2, ROI_BOTTOM_HEIGHT));

			// Horizontal variables
			int horizontalCount;

			// Line equation variables
			double leftX1, leftX2, leftY1, leftY2, rightX1, rightX2, rightY1, rightY2;
			double mLeftEdgeOfMask, cLeftEdgeOfMask, mRightEdgeOfMask, cRightEdgeOfMask;
			double topMidPoint, bottomOneThird, bottomTwoThird;
			double mLeftThresholdEdge, cLeftThresholdEdge, mRightThresholdEdge, cRightThresholdEdge;

			// Left edge of mask
			mLeftEdgeOfMask = ((double)maskDimensions[0].y - (double)maskDimensions[3].y) / (double)((double)maskDimensions[0].x - (double)maskDimensions[3].x);
			cLeftEdgeOfMask = maskDimensions[0].y - mLeftEdgeOfMask * maskDimensions[0].x;

			// Right edge of mask
			mRightEdgeOfMask = ((double)maskDimensions[1].y - (double)maskDimensions[2].y) / (double)((double)maskDimensions[1].x - (double)maskDimensions[2].x);
			cRightEdgeOfMask = maskDimensions[1].y - mRightEdgeOfMask * maskDimensions[1].x;

			// Find the midpoint of top and the 1/3 point on the bottom of ROI
			topMidPoint = maskDimensions[0].x + ((double)maskDimensions[1].x - (double)maskDimensions[0].x) / 2.;
			bottomOneThird = maskDimensions[3].x + ((double)maskDimensions[2].x - (double)maskDimensions[3].x) / 3.;
			bottomTwoThird = maskDimensions[3].x + 2. * ((double)maskDimensions[2].x - (double)maskDimensions[3].x) / 3.;

			// Then find equation of the lines
			// left threshold
			mLeftThresholdEdge = ((double)ROI_TOP_HEIGHT - (double)ROI_BOTTOM_HEIGHT) / (topMidPoint - bottomOneThird);
			cLeftThresholdEdge = ROI_TOP_HEIGHT - mLeftThresholdEdge * topMidPoint;

			// right threshold
			mRightThresholdEdge = ((double)ROI_TOP_HEIGHT - (double)ROI_BOTTOM_HEIGHT) / (topMidPoint - bottomTwoThird);
			cRightThresholdEdge = ROI_TOP_HEIGHT - mRightThresholdEdge * topMidPoint;

			// Lane equation variables
			double mLeftLaneEdge = 0, cLeftLaneEdge = 0, mRightLaneEdge = 0, cRightLaneEdge = 0;
			bool lineIsInBoundingBox;
			int xLowerRange, xUpperRange, yLowerRange, yUpperRange;
			double dx, dy, gradient;

			// Variables for line type and driving state calculation
			std::deque<int> leftLineTypesForDisplay(5, 0), middleLineTypesForDisplay(5, 0), rightLineTypesForDisplay(5, 0);
			std::vector<cv::Vec4i> leftLines, middleLines, rightLines;
			double leftLineAverageSize, middleLineAverageSize, rightLineAverageSize;
			int leftLineType, middleLineType, rightLineType, drivingState;
			int leftMinY = 0, rightMinY = 0, minY;

			// Steering Input Variables
			double averageDistanceFromLeft = 0, averageDistanceFromRight = 0;
			double withinLaneCurrentDifference;
			double changingLanesCurrentDifference, changingLanesPreviousDifference = 0;
			int changingLanesframeCount = 0;
			std::string turningRequiredToReturnToCenter, currentTurningState;
			int turningRequiredToReturnToCenter_int;

			// Variables for traffic light state detection
			int NonZeroPixelsInGreen, NonZeroPixelsInRed;
			cv::Mat warpedimage, ImageInHSV;
			std::vector<cv::Point2f> srcTrafficLight, dstTrafficLight;

			// Writing text to screen
			std::string titleText = "", rightInfoTitleText = "", giveWayWarningText = "", FPSText = "";
			int baseline = 0;
			cv::Size textSize;
			cv::Point textOrg;
			cv::Rect rightInfoRect(1495, 25, 400, 360);
			cv::Rect FPSRect(0, 0, 145, 45);

			// Misc
			int i, j;



			while (1)
			{
				// Start the stop watch to measure the FPS
				auto start = std::chrono::high_resolution_clock::now();



				// Capture frame
				video >> frame;
				if (frame.empty())
					break;



				// Clear variables that are not over-written but instead added to
				objectBoundingBoxes.clear();
				objectNames.clear();
				objectConfidences.clear();
				preNMSObjectBoundingBoxes.clear();
				preNMSObjectNames.clear();
				preNMSObjectConfidences.clear();
				houghLines.clear();
				leftLines.clear();
				middleLines.clear();
				rightLines.clear();
				lanePoints.clear();



				// Populate blankFrame with zeros (all black) and
				// then create a white mask that is the same size as ROI
				blankFrame = cv::Mat::zeros(VIDEO_HEIGHT, VIDEO_WIDTH, frame.type());
				cv::fillConvexPoly(blankFrame, maskDimensions, cv::Scalar(255, 255, 255), cv::LINE_AA, 0);
				// Then AND blankFrame with frame to extract ROI from frame
				cv::bitwise_and(blankFrame, frame, ROIFrame);

				// Convert to gray scale for canny algorithm
				cv::cvtColor(ROIFrame, ROIFrame, cv::COLOR_BGR2GRAY);

				// Canny algorithm to detect edges
				cv::Canny(ROIFrame, cannyFrame, CANNY_LOWER_THRESHOLD, CANNY_UPPER_THRESHOLD, 3, true);

				// Probabilistic Hough Line Transform to detect lines
				cv::HoughLinesP(cannyFrame, houghLines, 1, CV_PI / 180, HOUGHP_THRESHOLD, HOUGHP_MIN_LINE_LENGTH, HOUGHP_MAX_LINE_GAP);


				if (yolo[testNumber])
				{
					// YOLO Detection for object bounding boxes as they are used in Hough line analysis
					cv::dnn::blobFromImage(frame, blobFromImg, 1 / 255.0, cv::Size(BLOB_SIZE, BLOB_SIZE), cv::Scalar(0), true, false, CV_32F);
					net.setInput(blobFromImg);
					net.forward(outputBlobs, unconnectedOutLayersNames);

					// Go through all output blobs and only allow those with confidence above threshold
					for (i = 0; i < outputBlobs.size(); i++)
					{
						for (j = 0; j < outputBlobs[i].rows; j++)
						{
							// rows represent number/ID of the detected objects (proposed region)
							// so loop over number/ID of detected objects.

							// for each row, the score is from element 5 up
							// to number of classes index (5 - N columns)
							// [x, y, w, h, confidence for class 1, confidence for class 2, ...]
							// minMacLoc gives the max value and its location, i.e. its classID
							cv::minMaxLoc(outputBlobs[i].row(j).colRange(5, outputBlobs[i].cols), NULL, &confidence, NULL, &classID);

							if (confidence > YOLO_CONFIDENCE_THRESHOLD)
							{
								// Get the four int values from output blob for bounding box
								centerX = outputBlobs[i].at<float>(j, 0) * (double)VIDEO_WIDTH;
								centerY = outputBlobs[i].at<float>(j, 1) * (double)VIDEO_HEIGHT;
								width = outputBlobs[i].at<float>(j, 2) * (double)VIDEO_WIDTH + BOUNDING_BOX_BUFFER;
								height = outputBlobs[i].at<float>(j, 3) * (double)VIDEO_HEIGHT + BOUNDING_BOX_BUFFER;

								// Remove object detections on the hood of car
								if (centerY < ROI_BOTTOM_HEIGHT)
								{
									preNMSObjectBoundingBoxes.push_back(cv::Rect(centerX - width / 2, centerY - height / 2, width, height));
									preNMSObjectNames.push_back(modelIntsAndNames[classID.x]);
									preNMSObjectConfidences.push_back(confidence);

								}
							}
						}
					}

					// Apply non-maxima suppression to supress overlapping bounding boxes
					// For objects that overlap, the highest confidence object will be chosen
					cv::dnn::NMSBoxes(preNMSObjectBoundingBoxes, preNMSObjectConfidences, 0.0, YOLO_NMS_THRESHOLD, indicesAfterNMS);

					// boundingBoxes.size() = classIDs.size() = confidences.size()
					// Expect only the objects that dont overlap
					for (i = 0; i < indicesAfterNMS.size(); i++)
					{
						objectBoundingBoxes.push_back(preNMSObjectBoundingBoxes[indicesAfterNMS[i]]);
						objectNames.push_back(preNMSObjectNames[indicesAfterNMS[i]]);
						objectConfidences.push_back(preNMSObjectConfidences[indicesAfterNMS[i]]);

						// Print object bounding boxes, object names and object confidences to the frame
						// This happens first so that the bounding boxes do not go over the UI
						trafficLightState = "";

						if (objectNames.back() == "traffic light")
						{
							srcTrafficLight.clear();
							srcTrafficLight.push_back(cv::Point2f(objectBoundingBoxes.back().x, objectBoundingBoxes.back().y));
							srcTrafficLight.push_back(cv::Point2f(objectBoundingBoxes.back().x + objectBoundingBoxes.back().width, objectBoundingBoxes.back().y));
							srcTrafficLight.push_back(cv::Point2f(objectBoundingBoxes.back().x, objectBoundingBoxes.back().y + objectBoundingBoxes.back().height));
							srcTrafficLight.push_back(cv::Point2f(objectBoundingBoxes.back().x + objectBoundingBoxes.back().width, objectBoundingBoxes.back().y + objectBoundingBoxes.back().height));

							dstTrafficLight.clear();
							dstTrafficLight.push_back(cv::Point2f(0, 0));
							dstTrafficLight.push_back(cv::Point2f(100, 0));
							dstTrafficLight.push_back(cv::Point2f(0, 200));
							dstTrafficLight.push_back(cv::Point2f(100, 200));

							// To warp perspective to only contain traffic light
							cv::warpPerspective(frame, warpedimage, cv::getPerspectiveTransform(srcTrafficLight, dstTrafficLight, 0), cv::Size(100, 200));

							// count the number of green pixels
							cv::cvtColor(warpedimage, ImageInHSV, cv::COLOR_BGR2HSV);
							cv::inRange(ImageInHSV, cv::Scalar(32, 32, 32), cv::Scalar(80, 255, 255), ImageInHSV);
							NonZeroPixelsInGreen = cv::countNonZero(ImageInHSV);

							// count the number of red pixels
							cv::cvtColor(warpedimage, ImageInHSV, cv::COLOR_BGR2HSV);
							cv::inRange(ImageInHSV, cv::Scalar(0, 64, 64), cv::Scalar(10, 255, 255), ImageInHSV);
							NonZeroPixelsInRed = cv::countNonZero(ImageInHSV);

							if (NonZeroPixelsInGreen > NonZeroPixelsInRed)
								trafficLightState = " (Green)";
							else //(NonZeroPixelsInRed > NonZeroPixelsInGreen)
								trafficLightState = " (Red)";
						}

						// Draw rectangle around detected object with the correct colour
						cv::rectangle(frame, objectBoundingBoxes.back(), modelNamesAndColourList[objectNames.back()], 1, cv::LINE_AA);

						// Construct the correct name of object with confidence
						std::string name = objectNames.back() + ": " + std::to_string((int)(100 * objectConfidences.back())) + "%" + trafficLightState;
						int size;
						// This auto adjusts the background box to be the same size as 'name' expect
						// if name is smaller than object rectangle width, where it will be the same
						// size as object rectangle width
						if (objectBoundingBoxes.back().width > name.size() * 9)
							size = objectBoundingBoxes.back().width;
						else
							size = name.size() * 9;
						cv::rectangle(frame, cv::Rect(objectBoundingBoxes.back().x, objectBoundingBoxes.back().y - 15, size, 15), modelNamesAndColourList[objectNames.back()], cv::FILLED, cv::LINE_AA);
						cv::putText(frame, name, cv::Point(objectBoundingBoxes.back().x, objectBoundingBoxes.back().y - 2), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0), 1, cv::LINE_AA);
					}
				}



				// Analyse the Hough lines
				for (i = horizontalCount = leftLineAverageSize = middleLineAverageSize = rightLineAverageSize = 0; i < houghLines.size(); i++)
				{
					// Remove lines that are inside a detected object bounding box
					for (j = 0, lineIsInBoundingBox = false; j < objectBoundingBoxes.size(); j++)
					{
						xLowerRange = objectBoundingBoxes[j].x;
						xUpperRange = objectBoundingBoxes[j].x + objectBoundingBoxes[j].width;
						yLowerRange = objectBoundingBoxes[j].y;
						yUpperRange = objectBoundingBoxes[j].y + objectBoundingBoxes[j].height;

						if (((houghLines[i][0] >= xLowerRange) && (houghLines[i][0] <= xUpperRange)) &&
							((houghLines[i][1] >= yLowerRange) && (houghLines[i][1] <= yUpperRange)))
						{
							lineIsInBoundingBox = true;
							break;
						}

						if (((houghLines[i][2] >= xLowerRange) && (houghLines[i][2] <= xUpperRange)) &&
							((houghLines[i][3] >= yLowerRange) && (houghLines[i][3] <= yUpperRange)))
						{
							lineIsInBoundingBox = true;
							break;
						}
					}
					if (lineIsInBoundingBox)
						continue;

					// Calculate the gradient of Hough line
					dx = (double)houghLines[i][0] - (double)houghLines[i][2];
					dy = (double)houghLines[i][1] - (double)houghLines[i][3];
					// check for divide by zero error and remove
					if (dx == 0)
						continue;
					gradient = dy / dx;

					// Horizontal Lines
					// If lines have a gradient less than HORIZONTAL_GRADIENT_THRESHOLD then possibly horizontal
					if (std::fabs(gradient) < HORIZONTAL_GRADIENT_THRESHOLD)
					{
						// Remove top and bottom edge of mask
						if (((houghLines[i][1] <= ROI_TOP_HEIGHT + 1) && (houghLines[i][3] <= ROI_TOP_HEIGHT + 1)) ||
							((houghLines[i][1] >= ROI_BOTTOM_HEIGHT - 1) && (houghLines[i][3] >= ROI_BOTTOM_HEIGHT - 1)))
							continue;

						// If longer than HORIZONTAL_LENGTH_THRESHOLD then definately horizontal
						if (std::sqrt(dy * dy + dx * dx) > HORIZONTAL_LENGTH_THRESHOLD)
						{
							horizontalCount++;
							continue;
						}
					}

					// Vertical Lines
					else
					{
						// Remove left edge of mask
						leftY1 = mLeftEdgeOfMask * houghLines[i][0] + cLeftEdgeOfMask;
						leftY2 = mLeftEdgeOfMask * houghLines[i][2] + cLeftEdgeOfMask;
						if ((houghLines[i][1] <= leftY1 + 1) && (houghLines[i][3] <= leftY2 + 1))
							continue;

						// left threshold
						leftY1 = mLeftThresholdEdge * houghLines[i][0] + cLeftThresholdEdge;
						leftY2 = mLeftThresholdEdge * houghLines[i][2] + cLeftThresholdEdge;

						if ((houghLines[i][1] < leftY1) && (houghLines[i][3] < leftY2) && gradient < 0)
						{
							leftLines.push_back(houghLines[i]);
							leftLineAverageSize += std::sqrt((houghLines[i][0] - houghLines[i][2]) * (houghLines[i][0] - houghLines[i][2])
								+ (houghLines[i][1] - houghLines[i][3]) * (houghLines[i][1] - houghLines[i][3]));
							continue;
						}

						// Remove right edge of mask
						rightY1 = mRightEdgeOfMask * houghLines[i][0] + cRightEdgeOfMask;
						rightY2 = mRightEdgeOfMask * houghLines[i][2] + cRightEdgeOfMask;
						if ((houghLines[i][1] <= rightY1 + 1) && (houghLines[i][3] <= rightY2 + 1))
							continue;

						// right threshold
						rightY1 = mRightThresholdEdge * houghLines[i][0] + cRightThresholdEdge;
						rightY2 = mRightThresholdEdge * houghLines[i][2] + cRightThresholdEdge;

						if ((houghLines[i][1] < rightY1) && (houghLines[i][3] < rightY2) && gradient > 0)
						{
							rightLines.push_back(houghLines[i]);
							rightLineAverageSize += std::sqrt((houghLines[i][0] - houghLines[i][2]) * (houghLines[i][0] - houghLines[i][2])
								+ (houghLines[i][1] - houghLines[i][3]) * (houghLines[i][1] - houghLines[i][3]));
							continue;
						}

						// else must be in middle
						middleLines.push_back(houghLines[i]);

						middleLineAverageSize += std::sqrt((houghLines[i][0] - houghLines[i][2]) * (houghLines[i][0] - houghLines[i][2])
							+ (houghLines[i][1] - houghLines[i][3]) * (houghLines[i][1] - houghLines[i][3]));
					}
				}



				// If horizontal line exists
				if (horizontalCount > HORIZONTAL_COUNT_THRESHOLD)
				{
					if (horizontalLineStateRollingAverage.calculateRollingAverage(1))
					{
						giveWayWarningText = "WARNING: Giveway ahead";
						baseline = 0;
						textSize = cv::getTextSize(giveWayWarningText, FONT_FACE, FONT_SCALE, FONT_THICKNESS, &baseline);
						baseline += FONT_THICKNESS;

						// center the text
						textOrg = cv::Point(((double)VIDEO_WIDTH - (double)textSize.width) / 2., 225 + baseline + textSize.height);

						// draw the box
						cv::rectangle(frame, textOrg + cv::Point(0, baseline), textOrg + cv::Point(textSize.width, -textSize.height - baseline), cv::Scalar(0), cv::FILLED);

						// then put the text itself
						cv::putText(frame, giveWayWarningText, textOrg, FONT_FACE, FONT_SCALE, cv::Scalar::all(255), FONT_THICKNESS, cv::LINE_AA);
					}
				}
				// Else input 0 and carry on
				else
					horizontalLineStateRollingAverage.calculateRollingAverage(0);

				// Average size of each line
				leftLineAverageSize /= (double)leftLines.size();
				middleLineAverageSize /= (double)middleLines.size();
				rightLineAverageSize /= (double)rightLines.size();



				// If above certain length solid if not dashed if neither then no line detected
				// this value is then inputted to the rolling average
				if (isinf(leftLineAverageSize))
					leftLineType = leftLineTypeRollingAverage.calculateRollingAverage(0);
				else if (leftLineAverageSize < SOLID_LINE_LENGTH_THRESHOLD)
					leftLineType = leftLineTypeRollingAverage.calculateRollingAverage(1);
				else
					leftLineType = leftLineTypeRollingAverage.calculateRollingAverage(2);

				// If above certain length solid if not dashed if neither then no line detected
				// this value is then inputted to the rolling average
				if (isinf(middleLineAverageSize))
					middleLineType = middleLineTypeRollingAverage.calculateRollingAverage(0);
				else if (middleLineAverageSize < SOLID_LINE_LENGTH_THRESHOLD)
					middleLineType = middleLineTypeRollingAverage.calculateRollingAverage(1);
				else
					middleLineType = middleLineTypeRollingAverage.calculateRollingAverage(2);

				// If above certain length solid if not dashed if neither then no line detected
				// this value is then inputted to the rolling average
				if (isinf(rightLineAverageSize))
					rightLineType = rightLineTypeRollingAverage.calculateRollingAverage(0);
				else if (rightLineAverageSize < SOLID_LINE_LENGTH_THRESHOLD)
					rightLineType = rightLineTypeRollingAverage.calculateRollingAverage(1);
				else
					rightLineType = rightLineTypeRollingAverage.calculateRollingAverage(2);

				// Determine which driving state the car is currently in
				// and then input this value to the rolling averages
				// Within Lanes
				if ((leftLines.size() != 0) && (middleLines.size() != 0) && (rightLines.size() != 0))
					drivingState = drivingStateRollingAverage.calculateRollingAverage(0);

				else if ((leftLines.size() != 0) && (middleLines.size() == 0) && (rightLines.size() != 0))
					drivingState = drivingStateRollingAverage.calculateRollingAverage(0);

				// Changing Lanes
				else if ((leftLines.size() == 0) && (middleLines.size() != 0) && (rightLines.size() == 0))
					drivingState = drivingStateRollingAverage.calculateRollingAverage(1);

				else if ((leftLines.size() != 0) && (middleLines.size() != 0) && (rightLines.size() == 0))
					drivingState = drivingStateRollingAverage.calculateRollingAverage(1);

				else if ((leftLines.size() == 0) && (middleLines.size() != 0) && (rightLines.size() != 0))
					drivingState = drivingStateRollingAverage.calculateRollingAverage(1);

				// Only left road marking detected
				else if ((leftLines.size() != 0) && (middleLines.size() == 0) && (rightLines.size() == 0))
					drivingState = drivingStateRollingAverage.calculateRollingAverage(2);

				// Only right road marking detected
				else if ((leftLines.size() == 0) && (middleLines.size() == 0) && (rightLines.size() != 0))
					drivingState = drivingStateRollingAverage.calculateRollingAverage(3);

				// No road marking detected
				else //((leftLines.size() == 0) && (middleLines.size() == 0) && (rightLines.size() == 0))
					drivingState = drivingStateRollingAverage.calculateRollingAverage(4);



				// Draw a filled black rectangle for the information on RHS
				cv::rectangle(frame, rightInfoRect, cv::Scalar(0), cv::FILLED, cv::LINE_AA, 0);

				// These statements add the current line state to the beginning of a
				// STL deque container and then remove the end value, thus keeping it a size of 5
				leftLineTypesForDisplay.push_front(leftLineType);
				leftLineTypesForDisplay.pop_back();

				middleLineTypesForDisplay.push_front(middleLineType);
				middleLineTypesForDisplay.pop_back();

				rightLineTypesForDisplay.push_front(rightLineType);
				rightLineTypesForDisplay.pop_back();

				// Left line state on RHS box
				for (i = 0; i < leftLineTypesForDisplay.size(); i++)
					cv::rectangle(frame, cv::Rect(1595, 80 + i * 50, 4, 25 * leftLineTypesForDisplay[i]), cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
				// Right line state on RHS box
				for (i = 0; i < rightLineTypesForDisplay.size(); i++)
					cv::rectangle(frame, cv::Rect(1795, 80 + i * 50, 4, 25 * rightLineTypesForDisplay[i]), cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);



				// Execute the appropriate code for the current driving state
				switch (drivingState)
				{
				case 0: // Within lanes
				{
					// Check to prevent divide by zero error
					// Calculate the average distance to left edge, minimum y,
					// and average mLeftLaneEdge and cLeftLaneEdge
					if (leftLines.size() != 0)
					{
						// distance to left edge
						for (i = averageDistanceFromLeft = mLeftLaneEdge = cLeftLaneEdge = 0, leftMinY = leftLines[i][1]; i < leftLines.size(); i++)
						{
							leftX1 = (leftLines[i][1] - cLeftEdgeOfMask) / mLeftEdgeOfMask;
							leftX2 = (leftLines[i][3] - cLeftEdgeOfMask) / mLeftEdgeOfMask;

							averageDistanceFromLeft += std::fabs(leftLines[i][0] - leftX1);
							averageDistanceFromLeft += std::fabs(leftLines[i][2] - leftX2);

							if (leftLines[i][1] < leftMinY)
								leftMinY = leftLines[i][1];

							if (leftLines[i][3] < leftMinY)
								leftMinY = leftLines[i][3];

							// Find average m and c values for left lane
							mLeftLaneEdge += ((double)leftLines[i][1] - (double)leftLines[i][3]) / (double)((double)leftLines[i][0] - (double)leftLines[i][2]);
							cLeftLaneEdge += leftLines[i][1] - (((double)leftLines[i][1] - (double)leftLines[i][3]) / (double)((double)leftLines[i][0] - (double)leftLines[i][2])) * leftLines[i][0];
						}

						averageDistanceFromLeft /= (double)(leftLines.size() * 2);
						mLeftLaneEdge /= (double)(leftLines.size());
						cLeftLaneEdge /= (double)(leftLines.size());
					}

					// Check to prevent divide by zero error
					// Calculate the average distance to right edge, minimum y,
					// and average mRightLaneEdge and cRightLaneEdge
					if (rightLines.size() != 0)
					{
						// distance to right edge
						for (i = averageDistanceFromRight = mRightLaneEdge = cRightLaneEdge = 0, rightMinY = ROI_BOTTOM_HEIGHT; i < rightLines.size(); i++)
						{
							rightX1 = (rightLines[i][1] - cRightEdgeOfMask) / mRightEdgeOfMask;
							rightX2 = (rightLines[i][3] - cRightEdgeOfMask) / mRightEdgeOfMask;

							averageDistanceFromRight += std::fabs(rightLines[i][0] - rightX1);
							averageDistanceFromRight += std::fabs(rightLines[i][2] - rightX2);

							if (rightLines[i][1] < rightMinY)
								rightMinY = rightLines[i][1];

							if (rightLines[i][3] < rightMinY)
								rightMinY = rightLines[i][3];

							// Find average m and c values for right lane
							mRightLaneEdge += ((double)rightLines[i][1] - (double)rightLines[i][3]) / (double)((double)rightLines[i][0] - (double)rightLines[i][2]);
							cRightLaneEdge += rightLines[i][1] - (((double)rightLines[i][1] - (double)rightLines[i][3]) / (double)((double)rightLines[i][0] - (double)rightLines[i][2])) * rightLines[i][0];
						}

						averageDistanceFromRight /= (double)(rightLines.size() * 2);
						mRightLaneEdge /= (double)(rightLines.size());
						cRightLaneEdge /= (double)(rightLines.size());
					}

					// Next determine position of car using distances from left and right lane to the left and right edge
					if ((averageDistanceFromLeft - averageDistanceFromRight) > 200)
						withinLaneCurrentDifference = 1;
					else if ((averageDistanceFromLeft - averageDistanceFromRight) < -200)
						withinLaneCurrentDifference = -1;
					else
						withinLaneCurrentDifference = (averageDistanceFromLeft - averageDistanceFromRight) / 200;

					// Calculate the turning needed to return to centre to the nearest 10%
					turningRequiredToReturnToCenter_int = (withinLaneCurrentDifference * 100) - (int)(withinLaneCurrentDifference * 100) % 10;

					// Calculate the direction of turning needed
					if (turningRequiredToReturnToCenter_int == 0)
						turningRequiredToReturnToCenter = "In Centre";
					else if (turningRequiredToReturnToCenter_int < 0)
						turningRequiredToReturnToCenter = "Turn Left " + std::to_string(-turningRequiredToReturnToCenter_int) + "%";
					else
						turningRequiredToReturnToCenter = "Turn Right " + std::to_string(turningRequiredToReturnToCenter_int) + "%";

					// Draw the yellow box that signifies the position of car with respect to the lanes detected
					blankFrame = cv::Mat::zeros(VIDEO_HEIGHT, VIDEO_WIDTH, frame.type());
					cv::rectangle(blankFrame, cv::Rect(1695 - turningRequiredToReturnToCenter_int - 75, 205 - 100, 150, 200), cv::Scalar(0, 200, 200), cv::FILLED, cv::LINE_AA);
					cv::add(frame, blankFrame, frame);

					// Draw the lane overlay
					// To avoid divide by zero error
					if ((mLeftLaneEdge != 0) && (mRightLaneEdge != 0))
					{
						// Then plot line from ROI_BOTTOM_HEIGHT to the lowest minY
						if (leftMinY < rightMinY)
							minY = leftMinY;
						else
							minY = rightMinY;

						// Make blank frame a blank black frame
						blankFrame = cv::Mat::zeros(VIDEO_HEIGHT, VIDEO_WIDTH, frame.type());

						// Add the four points of the quadrangle
						lanePoints.push_back(cv::Point((minY - cLeftLaneEdge) / mLeftLaneEdge, minY));
						lanePoints.push_back(cv::Point((minY - cRightLaneEdge) / mRightLaneEdge, minY));
						lanePoints.push_back(cv::Point((ROI_BOTTOM_HEIGHT - cRightLaneEdge) / mRightLaneEdge, ROI_BOTTOM_HEIGHT));
						lanePoints.push_back(cv::Point((ROI_BOTTOM_HEIGHT - cLeftLaneEdge) / mLeftLaneEdge, ROI_BOTTOM_HEIGHT));

						cv::fillConvexPoly(blankFrame, lanePoints, cv::Scalar(0, 64, 0), cv::LINE_AA, 0);

						// Can simply add the two images as the background in blankFrame
						// is black (0,0,0) and so will not affect the frame image
						// while still being able to see tarmac
						cv::add(frame, blankFrame, frame);
					}

					// Write the turning needed to the screen
					baseline = 0;
					textSize = cv::getTextSize(turningRequiredToReturnToCenter, FONT_FACE, FONT_SCALE, FONT_THICKNESS, &baseline);
					baseline += FONT_THICKNESS;
					// center the text
					textOrg = cv::Point((rightInfoRect.x + rightInfoRect.width / 2.) - textSize.width / 2., rightInfoRect.y + rightInfoRect.height + baseline - textSize.height);
					// then put the text itself
					cv::putText(frame, turningRequiredToReturnToCenter, textOrg, FONT_FACE, FONT_SCALE, cv::Scalar::all(255), FONT_THICKNESS, cv::LINE_AA);


					titleText = "Within Detected Lanes";
					rightInfoTitleText = "Detected Lanes";
					// Reset these to prevent errors
					changingLanesPreviousDifference = 0;
					changingLanesframeCount = 0;
					currentTurningState.clear();
					break;
				}

				case 1: // Changing lanes
				{
					// Check to prevent divide by zero error
					if (middleLines.size() != 0)
					{
						// Calculate the average distance to the left and right edge of the middle lane
						for (i = averageDistanceFromLeft = averageDistanceFromRight = 0; i < middleLines.size(); i++)
						{
							leftY1 = (middleLines[i][1] - cLeftEdgeOfMask) / mLeftEdgeOfMask;
							leftY2 = (middleLines[i][3] - cLeftEdgeOfMask) / mLeftEdgeOfMask;

							averageDistanceFromLeft += std::fabs(middleLines[i][0] - leftY1);
							averageDistanceFromLeft += std::fabs(middleLines[i][2] - leftY2);

							rightY1 = (middleLines[i][1] - cRightEdgeOfMask) / mRightEdgeOfMask;
							rightY2 = (middleLines[i][3] - cRightEdgeOfMask) / mRightEdgeOfMask;

							averageDistanceFromRight += std::fabs(middleLines[i][0] - rightY1);
							averageDistanceFromRight += std::fabs(middleLines[i][2] - rightY2);
						}

						averageDistanceFromLeft /= (double)(middleLines.size() * 2);
						averageDistanceFromRight /= (double)(middleLines.size() * 2);

						changingLanesCurrentDifference = averageDistanceFromLeft - averageDistanceFromRight;

						// To determine the direction the car is moving, multiple frames that are many frames apart need to be compared
						// to see a difference in lane position; thus, a frame count is used

						// Increment frame count and then check if the threshold met. If so, the current turning state is compared to the previous
						// turning state - which occurred FRAME_COUNT_THRESHOLD number of frames ago - and then determine the car's turning state and
						// update the previous difference and reset the counter.

						// this for if coming from a different driving state
						if (changingLanesframeCount == 0)
							changingLanesPreviousDifference = changingLanesCurrentDifference;

						changingLanesframeCount++;

						if (changingLanesframeCount == FRAME_COUNT_THRESHOLD)
						{
							// Returns whether the car is turning left, right, or not turning based on
							// a current and previous difference, which is a value that represents the
							// difference between the distances from the left and right edge with respect
							// to the left and right road markings. The threshold value defines how big of
							// a difference between current and previous for the car to be detected as turning
							if ((changingLanesPreviousDifference - changingLanesCurrentDifference) == 0)
								currentTurningState = "(Currently Not Turning)";
							else if ((changingLanesPreviousDifference - changingLanesCurrentDifference) < 0)
								currentTurningState = "(Currently Turning Left)";
							else
								currentTurningState = "(Currently Turning Right)";

							// Update previous difference
							if (changingLanesCurrentDifference != 0)
								changingLanesPreviousDifference = changingLanesCurrentDifference;

							changingLanesframeCount = 0;
						}
					}

					titleText = "WARNING: Changing lanes";
					rightInfoTitleText = "Detected Lanes";

					// Middle line type on RHS information box
					for (i = 0; i < middleLineTypesForDisplay.size(); i++)
						cv::rectangle(frame, cv::Rect(1695, 80 + i * 50, 4, 25 * middleLineTypesForDisplay[i]), cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);

					if (!currentTurningState.empty())
					{
						// Write the current turning state to screen
						baseline = 0;
						textSize = cv::getTextSize(currentTurningState, FONT_FACE, FONT_SCALE, FONT_THICKNESS, &baseline);
						baseline += FONT_THICKNESS;
						// center the text
						textOrg = cv::Point(((double)VIDEO_WIDTH - (double)textSize.width) / 2., 125 + baseline + textSize.height);
						// draw the box
						cv::rectangle(frame, textOrg + cv::Point(0, baseline), textOrg + cv::Point(textSize.width, -textSize.height - baseline), cv::Scalar(0), cv::FILLED);
						// then put the text itself
						cv::putText(frame, currentTurningState, textOrg, FONT_FACE, FONT_SCALE, cv::Scalar::all(255), FONT_THICKNESS, cv::LINE_AA);
					}
					break;
				}

				case 2: // Only left road marking detected
				{
					titleText = "WARNING: Only left road marking detected";
					rightInfoTitleText = "Detected Lanes";
					// Reset these to prevent errors
					changingLanesPreviousDifference = 0;
					changingLanesframeCount = 0;
					currentTurningState.clear();
					break;
				}

				case 3: // Only right road marking detected
				{
					titleText = "WARNING: Only right road marking detected";
					rightInfoTitleText = "Detected Lanes";
					// Reset these to prevent errors
					changingLanesPreviousDifference = 0;
					changingLanesframeCount = 0;
					currentTurningState.clear();
					break;
				}

				case 4: // No road markings detected
				{
					titleText = "WARNING: No road markings detected";
					rightInfoTitleText = "No Lanes Detected";
					// Reset these to prevent errors
					changingLanesPreviousDifference = 0;
					changingLanesframeCount = 0;
					currentTurningState.clear();
					break;
				}

				default:
					std::cout << "\ncurrentDrivingState switch statement error: " << drivingState;
					break;
				}



				// Write title to screen
				baseline = 0;
				textSize = cv::getTextSize(titleText, FONT_FACE, FONT_SCALE, FONT_THICKNESS, &baseline);
				baseline += FONT_THICKNESS;
				// center the text
				textOrg = cv::Point(((double)VIDEO_WIDTH - (double)textSize.width) / 2., 25 + baseline + textSize.height);
				// draw the box
				cv::rectangle(frame, textOrg + cv::Point(0, baseline), textOrg + cv::Point(textSize.width, -textSize.height - baseline), cv::Scalar(0), cv::FILLED);
				// then put the text itself
				cv::putText(frame, titleText, textOrg, FONT_FACE, FONT_SCALE, cv::Scalar::all(255), FONT_THICKNESS, cv::LINE_AA);

				// Write right info box title to screen
				baseline = 0;
				textSize = cv::getTextSize(rightInfoTitleText, FONT_FACE, FONT_SCALE, FONT_THICKNESS, &baseline);
				baseline += FONT_THICKNESS;
				// center the text
				textOrg = cv::Point((rightInfoRect.x + rightInfoRect.width / 2.) - textSize.width / 2., rightInfoRect.y + baseline + textSize.height);
				// then put the text itself
				cv::putText(frame, rightInfoTitleText, textOrg, FONT_FACE, FONT_SCALE, cv::Scalar::all(255), FONT_THICKNESS, cv::LINE_AA);

				// Display the current FPS
				cv::rectangle(frame, FPSRect, cv::Scalar(0), cv::FILLED);
				FPSText = std::to_string((int)(1000 / (double)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count())) + " FPS";
				baseline = 0;
				textSize = cv::getTextSize(FPSText, FONT_FACE, FONT_SCALE, FONT_THICKNESS, &baseline);
				baseline += FONT_THICKNESS;
				textOrg = cv::Point(5, baseline + textSize.height);
				cv::putText(ROIFrame, FPSText, textOrg, FONT_FACE, FONT_SCALE, cv::Scalar::all(255), FONT_THICKNESS, cv::LINE_AA);

				// Display the resulting frame
				cv::imshow("frame", frame);

				// Required to display the frame
				cv::waitKey(1);

				//temp.push_back(
					std::cout << '\n' << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

				// Write the frame to a video file
				if (recordOuput && iterations == 0)
					frameVideo << frame;
			}

			// When everything done, release the video capture objects
			video.release();
			frameVideo.release();

			// Closes all the frames
			cv::destroyAllWindows();



			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



			allIterations.push_back(temp);
		}

		if (saveResults == true)
		{
			// save the results
			std::ofstream results(platformDir + outputFileNames[testNumber]);
			if (results.is_open())
			{
				for (int i = 0; i < frameCount; i++)
				{
					int temp = 0;

					for (int j = 0; j < allIterations.size(); j++)
					{
						temp += allIterations[j][i];
					}

					results << std::to_string(temp / (double)numberOfRepetitions) << '\n';
				}
			}
		}
	}

	int total = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - begin).count();
	double hours = total / 3600.;
	double minutes = (hours - floor(hours)) * 60;
	double seconds = (minutes - floor(minutes)) * 60;

	hours = floor(hours);
	minutes = floor(minutes);
	seconds = floor(seconds);

	std::cout << "\n\nAll test complete!\nTotal time taken = " << hours << "hrs " << minutes << "mins " << seconds << "secs\n\n";

	return 0;
}
