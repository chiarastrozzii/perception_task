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

void drawLegend(Mat& frame) {
    int width = 200;
    int height = 100;
    Rect legend(10, 10, width, height);

    Mat overlay;
    frame.copyTo(overlay);
    rectangle(overlay, legend, Scalar(0, 0, 0), FILLED); 
    addWeighted(overlay, 0.4, frame, 0.6, 0, frame); //0.4 of transparency

    int startX = 20, startY = 30, boxSize = 15, gap = 25;

    rectangle(frame, Point(startX, startY), Point(startX + boxSize, startY + boxSize), Scalar(255, 0, 0), FILLED);
    putText(frame, "Blue cone", Point(startX + 25, startY + 12), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

    rectangle(frame, Point(startX, startY + gap), Point(startX + boxSize, startY + gap + boxSize), Scalar(0, 255, 255), FILLED);
    putText(frame, "Yellow cone", Point(startX + 25, startY + gap + 12), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

    rectangle(frame, Point(startX, startY + 2 * gap), Point(startX + boxSize, startY + 2 * gap + boxSize), Scalar(0, 0, 255), FILLED);
    putText(frame, "Red cone", Point(startX + 25, startY + 2 * gap + 12), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
}

void redConesDetection(Mat frame1, Mat frame_HSV, vector<Point>& red_centers){
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
    //imshow("removing noise", result);


    //detect the geometry using contours
    vector <vector <Point> > contours;
    vector <Vec4i> hierarchy;

    findContours(mask_red, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); //expect a single channel image

    //drawContours(frame1, contours, -1, Scalar(0, 255, 0), 2);
    //imshow("contours found", frame1);

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
        Point center(box.x + box.width/2, box.y + box.height/2);
        red_centers.push_back(center);
        rectangle(frame1, box, Scalar(0, 0, 255), 2);
        

        int vertices = (int)approx[j].size();

         if (vertices >= 3 && vertices <= 6) {
            putText(frame1, "RED CONE", Point(box.x, box.y - 10), //inferring the class!
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
        } else {
            putText(frame1, "UNKNOWN", Point(box.x, box.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        }

    }

    //drawLegend(frame1);
    //imshow("red cones", frame1);


    /* char key = (char)waitKey(0); //need to uncomment in case of testing the single function
    if (key == 27){
        return;
    } */
}

void blueConesDetection(Mat frame1, Mat frame_HSV, vector<Point>& blue_centers){
    Scalar lower_blue(50, 0, 70); //select a wider range to help detecting the further cones
    Scalar upper_blue(150, 255, 255);

    Rect roi( 0, 210, 400, 80);
    //rectangle(frame1, roi, Scalar(255, 0, 0), 2);

    Mat cropped_HSV = frame_HSV(roi); //crop the HSV so to compute the mask on the cropped part

    Mat mask_blue, result_blue;
    inRange(cropped_HSV, lower_blue, upper_blue, mask_blue);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(2,2));
    morphologyEx(mask_blue, mask_blue, MORPH_OPEN, kernel);
    morphologyEx(mask_blue, mask_blue, MORPH_CLOSE, kernel);
    morphologyEx(mask_blue, mask_blue, MORPH_DILATE, kernel);    


    vector<vector<Point>> contours;
    vector <Vec4i> hierarchy;

    findContours(mask_blue, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto& contour : contours) { //offset the counters so that they're on the right position
        for (auto& point : contour) {
            point.x += roi.x;
            point.y += roi.y;
        }
    }

    //drawContours(frame1, contours, -1, Scalar(0, 255, 0), 2);
    //imshow("contours found", frame1);

    int imgHeight = frame1.rows;
    vector <Rect> boxes;

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        Rect box = boundingRect(contour);
        boxes.push_back(box);
        double aspect = (double)box.height / (double)box.width;

        // Filter weird shapes
        if (aspect < 0.3 || aspect > 3.0) continue;
        rectangle(frame1, box, Scalar(255, 0, 0), 2);

        Point center(box.x + box.width/2, box.y + box.height/2);
        blue_centers.push_back(center);
    }

    //drawLegend(frame1);
    //imshow("detected blue cones", frame1);


    /* char key = (char)waitKey(0);
    if (key == 27){
        return;
    } */
} 

void yellowConesDetection(Mat frame1, Mat frame_HSV, vector<Point>& yellow_centers){
    Scalar lower_yellow(0, 90, 190); //select a wider range to help detecting the further cones
    Scalar upper_yellow(40, 255, 255);

    Rect roi( 0, 200, 600, 110);
    //rectangle(frame1, roi, Scalar(0, 255, 255), 2);

    Mat cropped_HSV = frame_HSV(roi); //crop the HSV so to compute the mask on the cropped part

    Mat mask_yellow, result_yellow;
    inRange(cropped_HSV, lower_yellow, upper_yellow, mask_yellow);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(1,1));
    morphologyEx(mask_yellow, mask_yellow, MORPH_OPEN, kernel);
    morphologyEx(mask_yellow, mask_yellow, MORPH_CLOSE, kernel);
    morphologyEx(mask_yellow, mask_yellow, MORPH_DILATE, kernel);    


    // contours
    vector<vector<Point>> contours;
    vector <Vec4i> hierarchy;

    findContours(mask_yellow, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto& contour : contours) { //offset the counters so that they're on the right position
        for (auto& point : contour) {
            point.x += roi.x;
            point.y += roi.y;
        }
    }

    //drawContours(frame1, contours, -1, Scalar(0, 255, 0), 2);
    //imshow("contours found", frame1);

    int imgHeight = frame1.rows;

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        Rect box = boundingRect(contour);
        double aspect = (double)box.height / (double)box.width;

        Point center(box.x + box.width/2, box.y + box.height/2);

        if (aspect < 0.5|| aspect > 3.5) continue;
        rectangle(frame1, box, Scalar(0, 255, 255), 2);

      
        yellow_centers.push_back(center);
    }

    //drawLegend(frame1);
    //imshow("detected yellow cones", frame1);


    /* char key = (char)waitKey(0);
    if (key == 27){
        return;
    } */

}

void drawBlueEdge(Mat frame1, vector<Point>& blue_centers ){
    vector<Point> sorted_cones = blue_centers;
    sort(sorted_cones.begin(), sorted_cones.end(), [](const Point&a, const Point& b){
        return a.x < b.x;
    });

    size_t mid_index = sorted_cones.size() / 2;
    vector<Point> leftCones(sorted_cones.begin(), sorted_cones.begin() + mid_index - 2);
    vector<Point> rightCones(sorted_cones.begin() + mid_index -2, sorted_cones.end());

    for (size_t i=1; i<leftCones.size(); ++i){
        line(frame1, leftCones[i-1], leftCones[i], Scalar(255,0,0), 2, LINE_AA);
    }

    vector <Point> sorted_right = rightCones;
    sort(sorted_right.begin(), sorted_right.end(), [] (const Point& a, const Point& b ){
        return a.y< b.y;
    });

    for (size_t i=1; i<sorted_right.size(); ++i){
        line(frame1, sorted_right[i-1], sorted_right[i], Scalar(255, 0, 0), 2, LINE_AA);
    }

    line(frame1, leftCones.back(), sorted_right.front(), Scalar(255, 0, 0), 2, LINE_AA);
}

void drawYellowEdge(Mat frame1, vector<Point>& yellow_centers ){
    vector<Point> sorted_cones = yellow_centers;
    sort(sorted_cones.begin(), sorted_cones.end(), [](const Point&a, const Point& b){
        return a.x < b.x;
    });

    for(size_t i = 1; i<sorted_cones.size(); ++i){
        line(frame1, sorted_cones[i-1], sorted_cones[i], Scalar(255,255,0), 2, LINE_AA );
    }

    /* size_t mid_index = sorted_cones.size() / 2;
    vector<Point> leftCones(sorted_cones.begin(), sorted_cones.begin() + mid_index);
    vector<Point> rightCones(sorted_cones.begin() + mid_index, sorted_cones.end());
 */
   
}

void drawRedLine(Mat frame1, vector<Point>& red_centers){
    for(size_t i=1; i<red_centers.size(); ++i){
        line(frame1, red_centers[i-1], red_centers[i], Scalar(0, 0, 255), 2, LINE_AA);
    }

     Point p1 = red_centers.front();
    Point p2 = red_centers.back();
    Point mid((p1.x + p2.x) / 2, (p1.y + p2.y) / 2); //middle point of the line

    string text = "START";
    int baseline = 0;
    Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline); //measuring the size of the text to know where to position it

    Point textOrg(mid.x - textSize.width / 2, mid.y + textSize.height - 20); //position the text slightly below the line

    putText(frame1, text, textOrg, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2, LINE_AA);
}

int main(){
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

    //to find center points of the cones, used for track trace
    vector <Point> blue_centers;
    vector <Point> yellow_centers;
    vector <Point> red_centers;


    //function to detect the red cones
    redConesDetection(frame1, frame_HSV, red_centers);
    //blue
    blueConesDetection(frame1, frame_HSV, blue_centers);
    //yellow
    yellowConesDetection(frame1, frame_HSV, yellow_centers);

    //draws a legend on top left corner for classification
    drawLegend(frame1);
    
    imshow("detection", frame1);


    //LEVEL 4
    //left edge of the track are the blue cones (find the center of the cones so that you can then use it to draw the lines)
    //right edge of track the yellow cones
    //start given by the red cones

    //to draw lines fitLine from the opencv lib

    //drawBlueEdge(frame1, blue_centers);
    //drawYellowEdge(frame1, yellow_centers);
    drawRedLine(frame1, red_centers);
    imshow("race track edges", frame1 );


    char key = (char)waitKey(0);
    if (key == 27){
        return 0;
    }


    return 0;
}
