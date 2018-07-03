#ifndef ROBUSTMATCHER_H
#define ROBUSTMATCHER_H

#include <QMainWindow>
#include <QObject>
#include <QWidget>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iomanip>
#include<vector>


class RobustMatcher
{

private:
// pointer to the feature point detector object
cv::Ptr<cv::FeatureDetector> detector;
// pointer to the feature descriptor extractor object
cv::Ptr<cv::DescriptorExtractor> extractor;
float ratio; // max ratio between 1st and 2nd NN
bool refineF; // if true will refine the F matrix
double distance; // min distance to epipolar
double confidence; // confidence level (probability)
public:
RobustMatcher() : ratio(0.65f), refineF(true),
confidence(0.99), distance(3.0) {
// SURF is the default feature

}
// Set the feature detector
void setFeatureDetector(
cv::Ptr<cv::FeatureDetector>& detect) {
detector= detect;
}
// Set the descriptor extractor
void setDescriptorExtractor(
cv::Ptr<cv::DescriptorExtractor>& desc) {
extractor= desc;
}
cv::Mat match(cv::Mat &image1, cv::Mat &image2, std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2);
};


#endif // ROBUSTMATCHER_H
