#Book-Spine Detection by YOLO
🧠 Introduction
This project implements an improved Hough Transform algorithm for book spine detection in smart libraries. The proposed method addresses common challenges in traditional spine segmentation, such as low detection accuracy, limited robustness, and high computational cost.

We present a lightweight and resource-efficient pipeline that enhances traditional Hough line detection by integrating the following techniques:

Weighted Hough Space Accumulation

Angle and Spatial Quantization Optimization

Adaptive Thresholding Mechanism

Geometry-Driven Line Segment Merging Strategy

Secondary Optimization Based on Geometric Constraints

This approach significantly improves the accuracy of book spine localization while ensuring compatibility with real-time processing and deployment on low-resource embedded systems.

🔧 How to Use the Code

To run this project, please ensure the following environment configuration:

Python: 3.7

CUDA: 12.4

PyTorch: 1.12.1 (available from official PyTorch website)

PaddlePaddle: 2.5.2

PaddleHub: 2.3.1

Note: PaddlePaddle and PaddleHub can be installed from the official PaddlePaddle site.

Within the main function, comments are provided to guide you through the process. Simply replace the value of directory_path with the path to your input images.

To use the visualization tool, please ensure that the current image is closed and saved before moving on to the next one (i.e., exit the current window to proceed).
