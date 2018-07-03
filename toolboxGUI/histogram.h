#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <QMainWindow>
#include <QObject>
#include <QWidget>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Histogram
{

private:

    cv::MatND hist;
    int histSize[1];
    float hranges[2];
    const float* ranges[1];
    int channels[1];

public:
 cv::MatND getHistogram(const cv::Mat &image);
  cv::Mat getHistogramImage(const cv::Mat &image);
  cv::Mat equalize(const cv::Mat &image);
  Histogram();


};

#endif // HISTOGRAM_H
