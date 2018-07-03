#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include<opencv2/calib3d/calib3d.hpp>
#include<QPixmap>
#include<QLabel>
#include"histogram.h"
#include"laplacian.h"
#include"cameracalibrator.h"
#include"robustmatcher.h"
namespace Ui{
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);

    ~MainWindow();
private:
    Ui::MainWindow *ui;
    cv::Mat image,image1,image2; // the image variable

    cv::Mat imageROI,img2,out,imageMatches,points1,points2;
    cv::Mat imgGrayscale;
    cv::Mat logo,bw;
    cv::Mat histo;
    cv::Mat E,n1;
    Histogram h;
    cv::Mat result;
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    RobustMatcher rmatcher;
    int saltAndPepper_value;
    double sobmin, sobmax;
    cv::Mat outs;
    cv::Mat flap,laplace;
    cv::Mat sobelX, sobelY;
    cv::Mat sobel;
    cv::Mat sobelImage;
    cv::Mat contours,THRESH_BINARY;
    cv::Mat results,result1,result2,result3,result4;
    cv::Mat HoughLines;
    cv::Mat finder,drawing;
    cv::Mat setMinVote;
    cv::Mat setLineLengthAndGap;
    cv::Mat Point,val,img,BORDER_DEFAULT, NORM_MINMAX;
    cv::Mat vector,size,pointIndexes1,pointIndexes2,selpoints1,selpoints2,Fundamental,fundamental,it;
    cv:: Mat dst, dst_norm,img1;
    cv::Mat Orig,temp;

    Laplacian l;
    double PI=3.14;

    int blockSize = 3;
    int apertureSize = 5;
    double k = 0.04;
    std::vector<cv::Vec4i> lines;
    cv::Mat canny_output,src_gray;
    cv::Mat Scalar, color,thresh;
    std::vector<std::vector<cv::Point>> contour;
    std::vector<cv::Vec4i>hierarchy;
    std::vector<cv::KeyPoint> keypoints1;
    cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SIFT::create(2000.0);
    cv::Ptr<cv::Feature2D> ptrFeature2DSurf;

private slots:
    void on_pushButton_clicked();
    void salt(cv::Mat &image, int n);
    void sharpen(const cv::Mat &image, cv::Mat &result);
    std::vector<cv::Vec4i> findLines(cv::Mat& binary);
    void drawDetectedLines(cv::Mat &image,cv::Scalar color=cv::Scalar(255,255,255));
    // void shape_Des(int, void* );
    void on_comboBox_activated(int index);

    void on_comboBox_2_activated(int index);
    void on_comboBox_3_activated(int index);
    void on_comboBox_4_activated(int index);
    void on_comboBox_5_activated(int index);
    void on_pushButton_2_clicked();

    void on_horizontalSlider_valueChanged(int value);
    //void on_pushButton_3_clicked();
};




#endif // MAINWINDOW_H


