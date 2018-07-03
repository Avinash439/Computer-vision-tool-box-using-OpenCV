#include "histogram1d.h"
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<QFileDialog>
#include<QtCore>
#include<iostream>

using namespace cv;
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
Histogram1D::Histogram1D()
{

}


Histogram1D h;

// Compute the histogram
cv::MatND histo= h.getHistogram(image);

for(int i=0; i<256; i++)

    cout << "Value " << i << " = " << histo.at<float>(i) << endl;

// Display a histogram as an image
cv::namedWindow("Histogcv::MatND histo= h.getHistogram(image);ram");
cv::imshow("Histogram",h.getHistogramImage(image));



