#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int lower_red, upper_red;

//threshold values
int hmin = 0, smin = 0, vmin = 0;
int hmax = 179, smax = 255, vmax = 255;

void onTrackbarChange(int , void*){} //updated globally

void maskTrackBar(Mat frame1){
    //to find the correct range of values for the mask use of a track bar:
    namedWindow("TrackBars", WINDOW_AUTOSIZE);
    
    createTrackbar("Hue Min", "TrackBars", &hmin, 179, onTrackbarChange); //179 is the max value of the hue value
    createTrackbar("Hue Max", "TrackBars", &hmax, 179, onTrackbarChange);
    createTrackbar("Sat Min", "TrackBars", &smin, 255, onTrackbarChange);
    createTrackbar("Sat Max", "TrackBars", &smax, 255, onTrackbarChange);
    createTrackbar("Val Min", "TrackBars", &vmin, 255, onTrackbarChange);
    createTrackbar("Val Max", "TrackBars", &vmax, 255, onTrackbarChange);

    setTrackbarPos("Hue max", "TrackBars", 179);
    setTrackbarPos("Sat max", "TrackBars", 255);
    setTrackbarPos("Val max", "TrackBars", 255);

    while(true){
        //to detect cones using computer vision [test]
        Mat frame_HSV, mask_red, mask_yellow, mask_blue, result_mask_r;
        cvtColor(frame1, frame_HSV, COLOR_BGR2HSV); //convert image to HSV

        hmin = getTrackbarPos("Hue Min", "TrackBars");
        hmax = getTrackbarPos("Hue Max", "TrackBars");
        smin = getTrackbarPos("Satu Min", "TrackBars");
        smax = getTrackbarPos("Sat Max", "TrackBars");
        vmin = getTrackbarPos("Val Min", "TrackBars");
        vmax = getTrackbarPos("Val Max", "TrackBars");


        Scalar lower_red(hmin, smin, vmin);
        Scalar upper_red(hmax, smax, vmax);
        inRange(frame_HSV, lower_red, upper_red, mask_red);

        bitwise_and(frame1, frame1, result_mask_r, mask_red);


        //imshow("racetrack 1", frame1);
        //imshow("racetrack HSV", frame_HSV);
        imshow("red mask", mask_red);
        imshow("applied mask", result_mask_r);
        //waitKey(1);

        char key = (char)waitKey(30); //image update every 30ms, when esc is pressed it stops
        if (key == 27) break;
        
    }
}

int main(){
    //level 1: load and display the image
    string path_image1 = "../images/frame_1.png";
    Mat frame1 = imread(path_image1); //matrix from opencv library where we store the image, read from the path given 

    if (frame1.empty()){
        cerr << "Could not load the image from path: " << path_image1 << endl;
        return -1;
    }

    imshow("racetrack 1", frame1);
    
    //to create a trackbar to find the useful values for creating the different masks
    maskTrackBar(frame1);

    char key = (char)waitKey(0);
    if (key == 27){
        return 0;
    }


    return 0;
}
