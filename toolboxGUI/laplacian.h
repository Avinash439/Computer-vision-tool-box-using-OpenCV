#ifndef LAPLACIAN_H
#define LAPLACIAN_H

#include <QMainWindow>
#include <QObject>
#include <QWidget>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Laplacian
{
private:
// original image
cv::Mat img;

// 32-bit float image containing the Laplacian
cv::Mat laplace;
// Aperture size of the laplacian kernel
int aperture;
public:

cv::Mat computeLaplacian(const cv::Mat& image);
cv::Mat getLaplacianImage(double scale);
void setAperture(int a);
//   Laplacian();//defining the constructor
Laplacian():aperture(3){}
};



#endif // LAPLACIAN_H
