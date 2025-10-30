#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;


//threshold values
int hmin = 0, smin = 0, vmin = 0;
int hmax = 179, smax = 255, vmax = 255;
int lower, upper; //used in the trackbar



void onTrackbarChange(int , void*){} //updated globally

void maskTrackBar(Mat frame1, Mat frame_HSV){
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
        Mat trial_mask, result_mask;

        hmin = getTrackbarPos("Hue Min", "TrackBars");
        hmax = getTrackbarPos("Hue Max", "TrackBars");
        smin = getTrackbarPos("Satu Min", "TrackBars");
        smax = getTrackbarPos("Sat Max", "TrackBars");
        vmin = getTrackbarPos("Val Min", "TrackBars");
        vmax = getTrackbarPos("Val Max", "TrackBars");


        Scalar lower(hmin, smin, vmin);
        Scalar upper(hmax, smax, vmax);
        inRange(frame_HSV, lower, upper, trial_mask);

        bitwise_and(frame1, frame1, result_mask, trial_mask);


        //imshow("racetrack 1", frame1);
        //imshow("racetrack HSV", frame_HSV);
        imshow("mask", trial_mask);
        imshow("mask applied", result_mask);
        //waitKey(1);

        char key = (char)waitKey(30); //image update every 30ms, when esc is pressed it stops
        if (key == 27) break;
        
    }
}

void redConesDetection(Mat frame1, Mat frame_HSV){
    //after having found correct range with trackbar, hardcode the values
    Scalar lower_red(109, 0, 175);
    Scalar upper_red(179, 255, 255);

    Mat mask_red, result;
    inRange(frame_HSV, lower_red, upper_red, mask_red);

    //remove noise
    Mat kernel = getStructuringElement(MORPH_RECT, Size(7,7)); //creates a rectangular kernel
    morphologyEx(mask_red, mask_red, MORPH_OPEN, kernel);
    morphologyEx(mask_red, mask_red, MORPH_CLOSE, kernel);
    morphologyEx(mask_red, mask_red, MORPH_DILATE, kernel);

    bitwise_and(frame1, frame1, result, mask_red); //result is a 3 channel color image
    imshow("removing noise", result);


    //detect the geometry using contours
    vector <vector <Point> > contours;
    vector <Vec4i> hierarchy;

    findContours(mask_red, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); //expect a single channel image

    drawContours(frame1, contours, -1, Scalar(0, 255, 0), 2);
    imshow("contours found", frame1);


    vector <vector <Point> > approx(contours.size()); //needed for the approximated polygon

    //approximate to triangle and detect cone
    for (size_t j=0; j< contours.size(); j++){
        double area = contourArea(contours[j]);
        if (area < 500)  // filter out small noise
            continue;

        double peri = arcLength(contours[j], true); //true is for open
        
        approxPolyDP(contours[j], approx[j], 0.02 * peri, true); // 0.02 is the epsilon and it's the approximation parameter, higher the epsilon less precise

        // bounding box which is used for labelling
        Rect box = boundingRect(approx[j]);
        rectangle(frame1, box, Scalar(0, 0, 255), 2);

        int vertices = (int)approx[j].size();

         if (vertices >= 3 && vertices <= 6) {
            putText(frame1, "RED CONE", Point(box.x, box.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
        } else {
            putText(frame1, "UNKNOWN", Point(box.x, box.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        }

    }

    imshow("red cones", frame1);


    char key = (char)waitKey(0);
    if (key == 27){
        return;
    }
}

void blueConesDetection(Mat frame1, Mat frame_HSV){
    Scalar lower_blue(77, 0, 112); //select a wider range to help detecting the further cones
    Scalar upper_blue(125, 255, 192);

    Rect roi( 0, 150, 400, 200);
    //rectangle(frame1, roi, Scalar(255, 0, 0), 2);

    Mat cropped_HSV = frame_HSV(roi); //crop the HSV so to compute the mask on the cropped part

    Mat mask_blue, result_blue;
    inRange(cropped_HSV, lower_blue, upper_blue, mask_blue);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(1,1));
    morphologyEx(mask_blue, mask_blue, MORPH_OPEN, kernel);
    //morphologyEx(mask_blue, mask_blue, MORPH_CLOSE, kernel);
    morphologyEx(mask_blue, mask_blue, MORPH_DILATE, kernel);    


    // 3. Find contours (candidate cones)
    vector<vector<Point>> contours;
    vector <Vec4i> hierarchy;

    findContours(mask_blue, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto& contour : contours) { //offset the counters so that they're on the right position
        for (auto& point : contour) {
            point.x += roi.x;
            point.y += roi.y;
        }
    }

    drawContours(frame1, contours, -1, Scalar(0, 255, 0), 2);
    imshow("contours found", frame1);

    int imgHeight = frame1.rows;

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        Rect box = boundingRect(contour);
        double aspect = (double)box.height / (double)box.width;

        // Filter weird shapes
        if (aspect < 0.5 || aspect > 4.0) continue;

        rectangle(frame1, box, Scalar(255, 0, 0), 2);
    }

    imshow("detected blue cones", frame1);


    char key = (char)waitKey(0);
    if (key == 27){
        return;
    }
} 

void yellowConesDetection(Mat frame1, Mat frame_HSV){
    Scalar lower_yellow(10, 90, 165); //select a wider range to help detecting the further cones
    Scalar upper_yellow(20, 255, 192);

    Rect roi( 100, 150, 400, 200);
    rectangle(frame1, roi, Scalar(255, 0, 0), 2);

    Mat cropped_HSV = frame_HSV(roi); //crop the HSV so to compute the mask on the cropped part

    Mat mask_yellow, result_yellow;
    inRange(cropped_HSV, lower_yellow, upper_yellow, mask_yellow);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(1,1));
    morphologyEx(mask_yellow, mask_yellow, MORPH_OPEN, kernel);
    morphologyEx(mask_yellow, mask_yellow, MORPH_CLOSE, kernel);
    morphologyEx(mask_yellow, mask_yellow, MORPH_DILATE, kernel);    


    // 3. Find contours (candidate cones)
    vector<vector<Point>> contours;
    vector <Vec4i> hierarchy;

    findContours(mask_yellow, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto& contour : contours) { //offset the counters so that they're on the right position
        for (auto& point : contour) {
            point.x += roi.x;
            point.y += roi.y;
        }
    }

    drawContours(frame1, contours, -1, Scalar(0, 255, 0), 2);
    imshow("contours found", frame1);

    int imgHeight = frame1.rows;

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        Rect box = boundingRect(contour);
        double aspect = (double)box.height / (double)box.width;

        // Filter weird shapes
        if (aspect < 0.5 || aspect > 4.0) continue;

        rectangle(frame1, box, Scalar(255, 0, 255), 2);
    }

    imshow("detected yellow cones", frame1);


    char key = (char)waitKey(0);
    if (key == 27){
        return;
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

    Mat frame_HSV;
    cvtColor(frame1, frame_HSV, COLOR_BGR2HSV); //convert image to HSV
    
    //to create a trackbar to find the useful values for creating the different masks
    //maskTrackBar(frame1, frame_HSV);

    //function to detect the red cones
    //redConesDetection(frame1, frame_HSV);
    //blueConesDetection(frame1, frame_HSV);
    yellowConesDetection(frame1, frame_HSV);

    char key = (char)waitKey(0);
    if (key == 27){
        return 0;
    }


    return 0;
}
