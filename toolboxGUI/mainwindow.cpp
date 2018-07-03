#include "mainwindow.h"
#include "ui_mainwindow.h"
#include"histogram.h"
#include "cameracalibrator.h"
#include"robustmatcher.h"
#include<iostream>
#include<QFileDialog>
#include<QtCore>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    //opening an image
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Image"), ".",
                                                    tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));
    image= cv::imread(fileName.toAscii().data());
    // cv::cvtColor(image,image,CV_BGR2RGB);
    cv::cvtColor(image,imgGrayscale,CV_BGR2GRAY);


    QImage img= QImage((const unsigned char*)(imgGrayscale.data),imgGrayscale.cols,imgGrayscale.rows,QImage::Format_Indexed8);
    ui->label->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::on_comboBox_activated(int index)
{

    if( index==1 )
    {
        //createTrackbar( "Min Threshold", MainWindow, 1, 100, salt );
       salt(imgGrayscale, saltAndPepper_value);

       QImage img= QImage((const unsigned char*)(imgGrayscale.data),imgGrayscale.cols,imgGrayscale.rows,QImage::Format_Indexed8);
       ui->label_2->setPixmap(QPixmap::fromImage(img));
    }
    else if( index==2 )
    {

        QString fileName = QFileDialog::getOpenFileName(this,
                                                        tr("Open Image"), ".",
                                                        tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));
        logo=cv::imread(fileName.toAscii().data());
        cv::cvtColor(logo,bw,CV_BGR2GRAY);

        // define image ROI at image bottom-right
        cv::Mat imageROI = imgGrayscale( cv::Rect(imgGrayscale.cols-logo.cols,imgGrayscale.rows-logo.rows,logo.cols,logo.rows));// ROI size

        // insert logo
        bw.copyTo(imageROI,bw);

        QImage img= QImage((const unsigned char*)(imgGrayscale.data),imgGrayscale.cols,imgGrayscale.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));


    }
    else if( index==3 )
    {
        //  cv::cvtColor(image, imgGrayscale, CV_BGR2Luv);
        // imgGrayscale= 150./255;
        cv::cvtColor(imgGrayscale, imgGrayscale, CV_GRAY2BGR);
      //  cv::cvtColor(imgGrayscale,imgGrayscale,CV_BGR2GRAY);
        QImage img= QImage((const unsigned char*)(imgGrayscale.data),imgGrayscale.cols,imgGrayscale.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));
    }


    else if(index==4)
    {
        //histogram of the image
        QImage img= QImage((const unsigned char*)(h.getHistogramImage(imgGrayscale).data),h.getHistogramImage(imgGrayscale).cols,h.getHistogramImage(imgGrayscale).rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));
        //ui->label_2->resize(ui->label->pixmap()->size());
    }


    else if(index==5)
    {
        //histogram equalization
        cv::Mat E= h.equalize(imgGrayscale);
        //           cv::namedWindow("Equalized Image");
        //           cv::imshow("Equalized Image",E);

        QImage img= QImage((const unsigned char*)(E.data),E.cols,E.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));

    }

    else if(index==6)
    {
        //blurringthe image
        cv::blur(imgGrayscale,result,cv::Size(5,5));
        QImage img= QImage((const unsigned char*)(result.data),result.cols,result.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));

    }



}

void MainWindow::on_comboBox_2_activated(int index)
{
    if(index==1)
    {
        // Erode the image
        cv::Mat eroded; // the destination image
        cv::erode(imgGrayscale,eroded,cv::Mat());
        // Display the eroded image
        QImage img= QImage((const unsigned char*)(eroded.data),eroded.cols,eroded.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));

    }
    else if(index==2)
    {
        // Dilate the image
        cv::Mat dilated; // the destination image
        cv::dilate(imgGrayscale,dilated,cv::Mat());
        // Display the dilated image
        QImage img= QImage((const unsigned char*)(dilated.data),dilated.cols,dilated.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));
    }
    else if(index==3)
    {
        cv::Mat element5(5,5,CV_8U,cv::Scalar(1));
        cv::Mat opened;// open the image
        cv::morphologyEx(imgGrayscale,opened,cv::MORPH_OPEN,element5);
        QImage img= QImage((const unsigned char*)(opened.data),opened.cols,opened.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));
    }
    else if(index==4)
    {
        cv::Mat element5(5,5,CV_8U,cv::Scalar(1));
        cv::Mat closed;// close operation the image
        cv::morphologyEx(imgGrayscale,closed,cv::MORPH_CLOSE,element5);
        QImage img= QImage((const unsigned char*)(closed.data),closed.cols,closed.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));
    }
}
void MainWindow::on_comboBox_3_activated(int index)
{
    if(index==1)
    {
        cv::Sobel(imgGrayscale,sobelX,CV_16S,1,0);
        cv::Sobel(imgGrayscale,sobelY,CV_16S,0,1);
        cv::Mat sobel;
        //compute the L1 norm
        sobel= abs(sobelX)+abs(sobelY);
        double sobmin, sobmax;
        cv::minMaxLoc(sobel,&sobmin,&sobmax);
        cv::Mat sobelImage;//applying sobbel function to the image
        sobel.convertTo(sobelImage,CV_8U,-255./sobmax,255);
        QImage img= QImage((const unsigned char*)(sobelImage.data),sobelImage.cols,sobelImage.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));
    }
    else if(index==2)
    {
//laplacian function
        l.setAperture(7);
        cv::Mat flap= l.computeLaplacian(imgGrayscale);
        laplace= l.getLaplacianImage(-1.0);

        QImage img= QImage((const unsigned char*)(laplace.data),laplace.cols,laplace.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));
    }
    else if(index==3)
    {
        //sharpen the image
        sharpen(imgGrayscale,result2);
        QImage img= QImage((const unsigned char*)(result2.data),result2.cols,result2.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));
    }
    else if(index==4)
    {
        //finding contours of the image
        cv::Mat contours;
        cv::Canny(image, // gray-level image
                  contours, // output contours
                  125, // low threshold
                  350); // high threshold
        QImage img= QImage((const unsigned char*)(contours.data),contours.cols,contours.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));
    }

    else if(index==5)
    {
        //extracting lines
        cv::Mat contours;
        cv::Canny(imgGrayscale,contours,125,350);
        std::vector<cv::Vec4i> lines= findLines(contours);
        drawDetectedLines(imgGrayscale);
        QImage img= QImage((const unsigned char*)(imgGrayscale.data),imgGrayscale.cols,imgGrayscale.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));
    }

    else if(index==6)
    {
       //extracting cirlces from the image.
        cv::GaussianBlur(imgGrayscale, imgGrayscale, cv::Size(5, 5), 1.5);
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(imgGrayscale, circles, cv::HOUGH_GRADIENT,
                         2,   // accumulator resolution (size of the image / 2)
                         20,  // minimum distance between two circles
                         200, // Canny high threshold
                         60, // minimum number of votes
                         15, 50); // min and max radius

        std::cout << "Circles: " << circles.size() << std::endl;

        // Draw the circles


        std::vector<cv::Vec3f>::const_iterator itc = circles.begin();

        while (itc!=circles.end()) {

            cv::circle(imgGrayscale,
                       cv::Point((*itc)[0], (*itc)[1]), // circle centre
                    (*itc)[2], // circle radius
                    cv::Scalar(255), // color
                    2); // thickness

            ++itc;
        }

        QImage img= QImage((const unsigned char*)(imgGrayscale.data),imgGrayscale.cols,imgGrayscale.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));
    }
    else if(index==7)
    {
        cv:: RNG rng(12345);
        threshold(imgGrayscale, result, 200, 255,cv:: THRESH_BINARY );
        Canny( result, canny_output, 25, 150, 3 );
        findContours( canny_output, contour, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
        // Draw contours
        cv::Mat results(canny_output.size(),CV_8U,cv::Scalar(255));

//drawing contours for the image
        cv::drawContours(results,contour,
                         -1, // draw all contours
                         cv::Scalar(0), // in black
                         2); // with a thickness of 2
        // Show in a window

        QImage img= QImage((const unsigned char*)(results.data),results.cols,results.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));
    }

    else if(index==8)
    {
        // cv::Mat threshold_output;
        std:: vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;


        cv:: RNG rng(12345);
        blur( imgGrayscale, imgGrayscale, cv::Size(3,3) );

        threshold(imgGrayscale, result4, 100, 255,cv:: THRESH_BINARY );

        findContours( result4, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );


        // Approximate contours to polygons + get bounding rects and circles
        std:: vector<std::vector<cv::Point> > contours_poly( contours.size() );
        std::vector<cv::Rect> boundRect( contours.size() );
        std::vector<cv::Point2f>center( contours.size() );
        std::vector<float>radius( contours.size() );

        for( int i = 0; i < contours.size(); i++ )
        { approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect(cv:: Mat(contours_poly[i]) );
            minEnclosingCircle( (cv::Mat)contours_poly[i], center[i], radius[i] );
        }


        //Draw polygonal contour + bonding rects + circles
        // cv:: Mat drawing b= cv::Mat::zeros( result4.size(), CV_8UC3 );
        imgGrayscale.copyTo(drawing);
        for( int i = 0; i< contours.size(); i++ )
        {
            cv:: Scalar color =cv:: Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            cv::drawContours( drawing, contours_poly, i, color, 5, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
            cv:: rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 4, 8, 0 );
            cv:: circle( drawing, center[i], (int)radius[i], color,3 , 8, 0 );
        }
        QImage img= QImage((const unsigned char*)(drawing.data),drawing.cols,drawing.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));

    }


    else if(index==9)
    {

        cv::Mat dst, dst_norm;
        dst =cv:: Mat::zeros(imgGrayscale.rows,imgGrayscale.cols, CV_32FC1 );


        cv::cornerHarris(imgGrayscale, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT );
        cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
        imgGrayscale.copyTo(img);

        for( int j = 0; j < dst_norm.rows ; j++ )
        {
            for( int i = 0; i < dst_norm.cols; i++ )
            {
                if( (int) dst_norm.at<float>(j,i) >100)
                {

                    circle( img, cv::Point( i, j ), 2,  cv::Scalar(0,255,0), 1, 8, 0 );
                }
            }
        }
        QImage img1= QImage((const unsigned char*)(img.data),img.cols,img.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img1));
    }

}


void MainWindow::on_comboBox_4_activated(int index)
{
    if(index==1)
    {
        // vector of keypoints

        std::vector<cv::KeyPoint> keypoints;

        //
        // Construct the BRISK feature detector object
        cv::Ptr<cv::BRISK> ptrBRISK = cv::BRISK::create();
        // detect the keypoints
        ptrBRISK->detect(imgGrayscale, keypoints);


        cv::drawKeypoints(imgGrayscale, // original image
                          keypoints, // vector of keypoints
                          out, // the resulting image
                          cv::Scalar(255,255,255), // color of the points
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); //flag
        cv::cvtColor(out,out,CV_BGR2GRAY);
        QImage img= QImage((const unsigned char*)(out.data),out.cols,out.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));




    }
    else if(index==2)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> ptrSURF =
                cv::xfeatures2d::SurfFeatureDetector::create(2000.0);
        // detect the keypoints
        ptrSURF->detect(imgGrayscale, keypoints);
        // Draw the keypoints with scale and orientation

        cv::drawKeypoints(imgGrayscale, keypoints, out,cv::Scalar(255,255,255),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::cvtColor(out,out,CV_BGR2GRAY);
        QImage img= QImage((const unsigned char*)(out.data),out.cols,out.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img));


    }


    else if(index==3)
    {

        std::vector<cv::KeyPoint> keypoints;


        cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> ptrSIFT =cv::xfeatures2d::SiftFeatureDetector::create();
        // detect the keypoints
        ptrSIFT->detect(imgGrayscale, keypoints);

        cv::Scalar keypointColor = cv::Scalar(255,0, 0);     // Blue keypoints.
        drawKeypoints(imgGrayscale, keypoints, out, keypointColor);
        cv::cvtColor(out,out,CV_BGR2GRAY);
        QImage img1= QImage((const unsigned char*)(out.data),out.cols,out.rows,QImage::Format_Indexed8);
        ui->label_2->setPixmap(QPixmap::fromImage(img1));
    }
    else if(index==4)
    {

        cv::Mat image;

        std::vector<std::string> filelist;

        QStringList filename = QFileDialog::getOpenFileNames(this,tr("Open File"));

        foreach( QString str, filename) {
            filelist.push_back(str.toStdString());
        }

        image= cv::imread( filelist[1],0);
        // Create calibrator object
        cameracalibrator cameraCalibrator;
        // add the corners from the chessboard
        cv::Size boardSize(6,4);
        cameraCalibrator.addChessboardPoints(
                    filelist,	// filenames of chessboard image
                    boardSize, "Detected points");	// size of chessboard

        // calibrate the camera
        cameraCalibrator.setCalibrationFlag(true,true);
        cameraCalibrator.calibrate(image.size());

        // Exampple of Image Undistortion
        image= cv::imread( filelist[1],0);
        cv::Size newSize(static_cast<int>(image.cols*1.5), static_cast<int>(image.rows*1.5));
        cv::Mat uImage= cameraCalibrator.remap(image, newSize);

        // display camera matrix
        cv::Mat cameraMatrix= cameraCalibrator.getCameraMatrix();
        std::cout << " Camera intrinsic: " << cameraMatrix.rows << "x" << cameraMatrix.cols << std::endl;
        std::cout << cameraMatrix.at<double>(0,0) << " " << cameraMatrix.at<double>(0,1) << " " << cameraMatrix.at<double>(0,2) << std::endl;
        std::cout << cameraMatrix.at<double>(1,0) << " " << cameraMatrix.at<double>(1,1) << " " << cameraMatrix.at<double>(1,2) << std::endl;
        std::cout << cameraMatrix.at<double>(2,0) << " " << cameraMatrix.at<double>(2,1) << " " << cameraMatrix.at<double>(2,2) << std::endl;

        cv::namedWindow("Original Image");
        cv::imshow("Original Image", image);
        cv::namedWindow("Undistorted Image");
        cv::imshow("Undistorted Image", uImage);

        // Store everything in a xml file
        cv::FileStorage fs("calib.xml", cv::FileStorage::WRITE);
        fs << "Intrinsic" << cameraMatrix;
        fs << "Distortion" << cameraCalibrator.getDistCoeffs();

        cv::waitKey();

    }
    else if(index==5)
    {

        std::vector<cv::KeyPoint> keypoints1;
        std::vector<cv::KeyPoint> keypoints2;
        // Read input images
        QString fileName = QFileDialog::getOpenFileName(this,
                                                        tr("Open Image"), ".",
                                                        tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));
        image1= cv::imread(fileName.toAscii().data());
        // cv::cvtColor(image,image,CV_BGR2RGB);
        cv::cvtColor(image1,image1,CV_BGR2GRAY);

        fileName = QFileDialog::getOpenFileName(this,
                                                tr("Open Image"), ".",
                                                tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));

        image2= cv::imread(fileName.toAscii().data());
        // cv::cvtColor(image,image,CV_BGR2RGB);
        cv::cvtColor(image2,image2,CV_BGR2GRAY);

        ptrFeature2D->compute(image1,keypoints1,descriptors1);
        ptrFeature2D->compute(image2,keypoints2,descriptors2);
        cv::BFMatcher matcher(cv::NORM_L2);
        // to test with crosscheck (symmetry) test
        // note: must not be used in conjunction with ratio test
        // cv::BFMatcher matcher(cv::NORM_L2, true); // with crosscheck
        // Match the two image descriptors
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors1,descriptors2, matches);


        std::cout << "Number of SIFT keypoints (image 1): " << keypoints1.size() << std::endl;
        std::cout << "Number of SIFT keypoints (image 2): " << keypoints2.size() << std::endl;

        // Extract the keypoints and descriptors
        ptrFeature2D = cv::xfeatures2d::SIFT::create();
        ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
        ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

        // Match the two image descriptors
        matcher.match(descriptors1,descriptors2, matches);

        // extract the 50 best matches
        std::nth_element(matches.begin(),matches.begin()+50,matches.end());
        matches.erase(matches.begin()+50,matches.end());

        // draw matches
        cv::drawMatches(
                    image1, keypoints1, // 1st image and its keypoints
                    image2, keypoints2, // 2nd image and its keypoints
                    matches,            // the matches
                    imageMatches,      // the image produced
                    cv::Scalar(255, 255, 255),  // color of lines
                    cv::Scalar(255, 255, 255), // color of points
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS| cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        // Display the image of matches


//        QImage img1= QImage((const unsigned char*)(imageMatches.data),imageMatches.cols,imageMatches.rows,QImage::Format_Indexed8);
//        ui->label_2->setPixmap(QPixmap::fromImage(img1));
        cv::namedWindow("Multi-scale SIFT Matches");
        cv::imshow("Multi-scale SIFT Matches",imageMatches);

        std::cout << "Number of matches: " << matches.size() << std::endl;

        cv::waitKey();
    }


}


void MainWindow::on_comboBox_5_activated(int index)
{
    if(index==1)
    {
        std::vector<cv::KeyPoint> keypoints1;
        std::vector<cv::KeyPoint> keypoints2;
        // Read input images
        QString fileName = QFileDialog::getOpenFileName(this,
                                                        tr("Open Image"), ".",
                                                        tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));
        image1= cv::imread(fileName.toAscii().data());
        // cv::cvtColor(image,image,CV_BGR2RGB);
        cv::cvtColor(image1,image1,CV_BGR2GRAY);

        fileName = QFileDialog::getOpenFileName(this,
                                                tr("Open Image"), ".",
                                                tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));

        image2= cv::imread(fileName.toAscii().data());
        // cv::cvtColor(image,image,CV_BGR2RGB);
        cv::cvtColor(image2,image2,CV_BGR2GRAY);
        ptrFeature2DSurf = cv::xfeatures2d::SURF::create();
//        ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
//        ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
        ptrFeature2DSurf->detect(image1, keypoints1);
        ptrFeature2DSurf->detect(image2, keypoints2);

        keypoints1.resize(8);
        keypoints2.resize(8);


        std::vector<cv::Point2f> selPoints1, selPoints2;
        cv::KeyPoint::convert(keypoints1,selPoints1,pointIndexes1);
        cv::KeyPoint::convert(keypoints2,selPoints2,pointIndexes2);
        cv::Mat fundemental= cv::findFundamentalMat(
                    cv::Mat(selPoints1), // points in first image
                    cv::Mat(selPoints2), // points in second image
                    CV_FM_7POINT); // //calculating fundamental matrix by using 7-point method



        std::vector<cv::Vec3f> lines1;
        cv::computeCorrespondEpilines(cv::Mat(selPoints1),1,fundemental,lines1);

        for (std::vector<cv::Vec3f>::const_iterator it= lines1.begin();
             it!=lines1.end(); ++it) {

            cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
                    cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
                    cv::Scalar(255,255,255));
        }

        std::vector<cv::Vec3f> lines2;
        cv::computeCorrespondEpilines(cv::Mat(selPoints2),2,fundemental,lines2);

        for (std::vector<cv::Vec3f>::const_iterator it= lines2.begin();
             it!=lines2.end(); ++it) {

            cv::line(image1,cv::Point(0,-(*it)[2]/(*it)[1]),
                    cv::Point(image1.cols,-((*it)[2]+(*it)[0]*image1.cols)/(*it)[1]),
                    cv::Scalar(255,255,255));
        }

       // Display the images with epipolar lines
        cv::namedWindow("Right Image Epilines ");
        cv::imshow("Right Image Epilines",image1);
        cv::namedWindow("Left Image Epilines");
        cv::imshow("Left Image Epilines",image2);
}
        else if(index==2)
        {
            std::vector<cv::KeyPoint> keypoints1;
            std::vector<cv::KeyPoint> keypoints2;
            // Read input images
            QString fileName = QFileDialog::getOpenFileName(this,
                                                            tr("Open Image"), ".",
                                                            tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));
            image1= cv::imread(fileName.toAscii().data());
            // cv::cvtColor(image,image,CV_BGR2RGB);
            cv::cvtColor(image1,image1,CV_BGR2GRAY);

            fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Image"), ".",
                                                    tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));

            image2= cv::imread(fileName.toAscii().data());
            // cv::cvtColor(image,image,CV_BGR2RGB);
            cv::cvtColor(image2,image2,CV_BGR2GRAY);
            ptrFeature2DSurf = cv::xfeatures2d::SURF::create();
    //        ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    //        ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
            ptrFeature2DSurf->detect(image1, keypoints1);
            ptrFeature2DSurf->detect(image2, keypoints2);

            keypoints1.resize(8);
            keypoints2.resize(8);


            std::vector<cv::Point2f> selPoints1, selPoints2;
            cv::KeyPoint::convert(keypoints1,selPoints1,pointIndexes1);
            cv::KeyPoint::convert(keypoints2,selPoints2,pointIndexes2);
            cv::Mat fundemental= cv::findFundamentalMat(
                        cv::Mat(selPoints1), // points in first image
                        cv::Mat(selPoints2), // points in second image
                        CV_FM_8POINT); // calculating fundamental matrix by using8-point method

            std::vector<cv::Vec3f> lines1;
            cv::computeCorrespondEpilines(cv::Mat(selPoints1),1,fundemental,lines1);

            for (std::vector<cv::Vec3f>::const_iterator it= lines1.begin();
                 it!=lines1.end(); ++it) {

                cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
                        cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
                        cv::Scalar(255,255,255));
            }

            std::vector<cv::Vec3f> lines2;
            cv::computeCorrespondEpilines(cv::Mat(selPoints2),2,fundemental,lines2);

            for (std::vector<cv::Vec3f>::const_iterator it= lines2.begin();
                 it!=lines2.end(); ++it) {

                cv::line(image1,cv::Point(0,-(*it)[2]/(*it)[1]),
                        cv::Point(image1.cols,-((*it)[2]+(*it)[0]*image1.cols)/(*it)[1]),
                        cv::Scalar(255,255,255));
            }

           // Display the images with epipolar lines
            cv::namedWindow("Right Image Epilines ");
            cv::imshow("Right Image Epilines",image1);
            cv::namedWindow("Left Image Epilines");
            cv::imshow("Left Image Epilines",image2);


    }

    else if(index==3)
    {
        std::vector<cv::KeyPoint> keypoints1;
        std::vector<cv::KeyPoint> keypoints2;
        // Read input images
        QString fileName = QFileDialog::getOpenFileName(this,
                                                        tr("Open Image"), ".",
                                                        tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));
        image1= cv::imread(fileName.toAscii().data());
        // cv::cvtColor(image,image,CV_BGR2RGB);
        cv::cvtColor(image1,image1,CV_BGR2GRAY);

        fileName = QFileDialog::getOpenFileName(this,
                                                tr("Open Image"), ".",
                                                tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));

        image2= cv::imread(fileName.toAscii().data());
        // cv::cvtColor(image,image,CV_BGR2RGB);
        cv::cvtColor(image2,image2,CV_BGR2GRAY);
        ptrFeature2DSurf = cv::xfeatures2d::SURF::create();
//        ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
//        ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
        ptrFeature2DSurf->detect(image1, keypoints1);
        ptrFeature2DSurf->detect(image2, keypoints2);

        keypoints1.resize(8);
        keypoints2.resize(8);
        std::vector<cv::Point2f> selPoints1, selPoints2;
        cv::KeyPoint::convert(keypoints1,selPoints1,pointIndexes1);
        cv::KeyPoint::convert(keypoints2,selPoints2,pointIndexes2);
        cv::Mat fundemental= cv::findFundamentalMat(//calculating fundamental matrix by using ransac
                    cv::Mat(selPoints1), // points in first image
                    cv::Mat(selPoints2), // points in second image
                    CV_FM_RANSAC); // RANSAC



        std::vector<cv::Vec3f> lines1;
        cv::computeCorrespondEpilines(cv::Mat(selPoints1),1,fundemental,lines1);

        for (std::vector<cv::Vec3f>::const_iterator it= lines1.begin();
             it!=lines1.end(); ++it) {

            cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
                    cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
                    cv::Scalar(255,255,255));
        }

        std::vector<cv::Vec3f> lines2;
        cv::computeCorrespondEpilines(cv::Mat(selPoints2),2,fundemental,lines2);

        for (std::vector<cv::Vec3f>::const_iterator it= lines2.begin();
             it!=lines2.end(); ++it) {

            cv::line(image1,cv::Point(0,-(*it)[2]/(*it)[1]),
                    cv::Point(image1.cols,-((*it)[2]+(*it)[0]*image1.cols)/(*it)[1]),
                    cv::Scalar(255,255,255));
        }

       // Display the images with epipolar lines
        cv::namedWindow("Right Image Epilines ");
        cv::imshow("Right Image Epilines",image1);
        cv::namedWindow("Left Image Epilines");
        cv::imshow("Left Image Epilines",image2);


}
    else if(index=4)
    {
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    // Read input images
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Image"), ".",
                                                    tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));
    image1= cv::imread(fileName.toAscii().data());
    // cv::cvtColor(image,image,CV_BGR2RGB);
    cv::cvtColor(image1,image1,CV_BGR2GRAY);

    fileName = QFileDialog::getOpenFileName(this,
                                            tr("Open Image"), ".",
                                            tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));

    image2= cv::imread(fileName.toAscii().data());
    // cv::cvtColor(image,image,CV_BGR2RGB);
    cv::cvtColor(image2,image2,CV_BGR2GRAY);
    ptrFeature2DSurf = cv::xfeatures2d::SURF::create();
    cv::Ptr<cv::DescriptorExtractor> extractor= cv::xfeatures2d::SurfDescriptorExtractor::create();


//        ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
//        ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
    ptrFeature2DSurf->detect(image1, keypoints1);
    ptrFeature2DSurf->detect(image2, keypoints2);

    keypoints1.resize(8);
    keypoints2.resize(8);


    cv::Mat descriptors1;
    extractor->compute(image1,keypoints1,descriptors1);

    cv::Mat descriptors2;
    extractor->compute(image2,keypoints2,descriptors2);

    cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L2, true));
    std::vector<cv::DMatch> matches;
               matcher->match(descriptors1,descriptors2, matches);
    std::vector<cv::Point2f> points1, points2;
    for (std::vector<cv::DMatch>::
    const_iterator it= matches.begin();
    it!= matches.end(); ++it) {
    // Get the position of left keypoints
    float x= keypoints1[it->queryIdx].pt.x;
    float y= keypoints1[it->queryIdx].pt.y;
    points1.push_back(cv::Point2f(x,y));
    // Get the position of right keypoints
    x= keypoints2[it->trainIdx].pt.x;
    y= keypoints2[it->trainIdx].pt.y;
    points2.push_back(cv::Point2f(x,y));
    }

    cv::KeyPoint::convert(keypoints1,points1,pointIndexes1);
       cv::KeyPoint::convert(keypoints2,points2,pointIndexes2);

        std::vector<uchar> inliers(points1.size(),0);
        cv::Mat homography= cv::findHomography(//homography
                    cv::Mat(points1), // corresponding
                    cv::Mat(points2), // points
                    inliers, // outputted inliers matches
                    CV_RANSAC, // RANSAC method
                    1.); // max distance to reprojection point

        std::vector<cv::Point2f>::const_iterator itPts=
                points1.begin();
        std::vector<uchar>::const_iterator itIn= inliers.begin();
        while (itPts!=points1.end()) {
            // draw a circle at each inlier location
            if (*itIn)
                cv::circle(image1,*itPts,3,
                           cv::Scalar(255,255,255),2);
            ++itPts;
            ++itIn;
        }
        itPts= points2.begin();

        cv::circle(image2,*itPts,3,
                   cv::Scalar(255,255,255),2);
    ++itPts;
    ++itIn;


cv::warpPerspective(image1, // input image
                    result, // output image
                    homography, // homography
                    cv::Size(2*image1.cols,
                             image1.rows)); // size of output image

cv::Mat half(result,cv::Rect(0,0,image2.cols,image2.rows));
image2.copyTo(half); // copy image2 to image1 roi
QImage img1= QImage((const unsigned char*)(result.data),result.cols,result.rows,QImage::Format_Indexed8);
ui->label_2->setPixmap(QPixmap::fromImage(img1));
       }
        }


void MainWindow::on_pushButton_2_clicked()
{
    //saving the image
    QString fileName = QFileDialog::getSaveFileName(this,
                tr("Save Address Book"), "",
                tr("JPG Image (*.jpg);;All Files (*)"));
        if (fileName.isEmpty())
            return;
        else {
            const QPixmap* pixmap = ui->label_2->pixmap();
            if ( pixmap )
            {
                QImage image( pixmap->toImage() );
                image.save(fileName,"jpg");
            }
        }
}


void MainWindow::on_horizontalSlider_valueChanged(int value)
{
    ui->textEdit->setText(QString::number(value));
    saltAndPepper_value = value;

}

//void MainWindow::on_pushButton_3_clicked()
//{
//    Orig = temp;
//    QImage image = cv::convertOpenCVMatToQtQImage(Orig);
//    ui->label_3->setPixmap(QPixmap::fromImage(image));
//}
