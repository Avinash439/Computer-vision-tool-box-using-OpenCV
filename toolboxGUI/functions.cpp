#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<QFileDialog>
#include<QtCore>
#include<iostream>

using namespace cv;
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void MainWindow:: salt(cv::Mat &image, int n) {


    int i,j;
    for (int k=0; k<n; k++) {

        // rand() is the MFC random number generator
        i= rand()%image.cols;
        j= rand()%image.rows;


       // if (image.channels() == 1) { // gray-level image
             image.at<uchar>(j,i)= 0;

        }

for (int k=0; k<n; k++) {

    // rand() is the MFC random number generator
    i= rand()%image.cols;
    j= rand()%image.rows;


   // if (image.channels() == 1) { // gray-level image
         image.at<uchar>(j,i)= 255;

    }
}

void MainWindow:: sharpen(const cv::Mat &image, cv::Mat &result) {
// allocate if necessary
result.create(image.size(), image.type());
for (int j= 1; j<image.rows-1; j++) { // for all rows
// (except first and last)
const uchar* previous=
image.ptr<const uchar>(j-1); // previous row
const uchar* current=
image.ptr<const uchar>(j); // current row
const uchar* next=
image.ptr<const uchar>(j+1); // next row
uchar* output= result.ptr<uchar>(j); // output row
for (int i=1; i<image.cols-1; i++) {
*output++= cv::saturate_cast<uchar>(
5*current[i]-current[i-1]
-current[i+1]-previous[i]-next[i]);
}
}
// Set the unprocess pixels to 0
result.row(0).setTo(cv::Scalar(0));
result.row(result.rows-1).setTo(cv::Scalar(0));
result.col(0).setTo(cv::Scalar(0));
result.col(result.cols-1).setTo(cv::Scalar(0));
}

// Apply probabilistic Hough Transform
std::vector<cv::Vec4i> MainWindow:: findLines(cv::Mat& binary) {
lines.clear();
cv::HoughLinesP(binary,lines,
1, PI/180, 80,
100, 20);
return lines;
}
void MainWindow:: drawDetectedLines(cv::Mat &image,cv::Scalar color) {
    // Draw the lines
    std::vector<cv::Vec4i>::const_iterator it2= lines.begin();
    while (it2!=lines.end()) {
    cv::Point pt1((*it2)[0],(*it2)[1]);
    cv::Point pt2((*it2)[2],(*it2)[3]);
    cv::line( image, pt1, pt2, color);
    ++it2;
    }
    }


