#include "laplacian.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;

cv::Mat aperture;






void Laplacian::setAperture(int a) {
aperture= a;
}

cv::Mat Laplacian::computeLaplacian(const cv::Mat& image) {
// Compute Laplacian
cv::Laplacian(image,laplace,CV_32F,aperture);
// Keep local copy of the image
// (used for zero-crossings)
img= image.clone();
return laplace;
}


cv::Mat Laplacian::getLaplacianImage(double scale=-1.0) {
if (scale<0) {
double lapmin, lapmax;
cv::minMaxLoc(laplace,&lapmin,&lapmax);
scale= 127/ std::max(-lapmin,lapmax);
}
cv::Mat laplaceImage;
laplace.convertTo(laplaceImage,CV_8U,scale,128);
return laplaceImage;
}
