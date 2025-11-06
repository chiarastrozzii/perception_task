#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
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

void redConesDetection(Mat& first_frame, Mat& frame_HSV, vector<Point>& red_centers){
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

    bitwise_and(first_frame, first_frame, result, mask_red); //result is a 3 channel color image
    //imshow("removing noise", result);


    //detect the geometry using contours
    vector <vector <Point> > contours;
    vector <Vec4i> hierarchy;

    findContours(mask_red, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); //expect a single channel image

    //drawContours(first_frame, contours, -1, Scalar(0, 255, 0), 2);
    //imshow("contours found", first_frame);

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
        rectangle(first_frame, box, Scalar(0, 0, 255), 2);
        

        int vertices = (int)approx[j].size();

         if (vertices >= 3 && vertices <= 6) {
            putText(first_frame, "RED CONE", Point(box.x, box.y - 10), //inferring the class!
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
        } else {
            putText(first_frame, "UNKNOWN", Point(box.x, box.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        }

    }

    //drawLegend(first_frame);
    //imshow("red cones", first_frame);


    /* char key = (char)waitKey(0); //need to uncomment in case of testing the single function
    if (key == 27){
        return;
    } */
}

void blueConesDetection(Mat& first_frame, Mat& frame_HSV, vector<Point>& blue_centers){
    Scalar lower_blue(50, 0, 70); //select a wider range to help detecting the further cones
    Scalar upper_blue(150, 255, 255);

    Rect roi( 0, 210, 400, 80);
    //rectangle(first_frame, roi, Scalar(255, 0, 0), 2);

    Mat cropped_HSV = frame_HSV(roi); //crop the HSV so to compute the mask on the cropped part

    Mat mask_blue, result_blue;
    inRange(cropped_HSV, lower_blue, upper_blue, mask_blue);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
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

    //drawContours(first_frame, contours, -1, Scalar(0, 255, 0), 2);
    //imshow("contours found", first_frame);

    int imgHeight = first_frame.rows;
    vector <Rect> boxes;

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        Rect box = boundingRect(contour);
        boxes.push_back(box);
        double aspect = (double)box.height / (double)box.width;

        // Filter weird shapes
        if (aspect < 0.3 || aspect > 3.0) continue;
        rectangle(first_frame, box, Scalar(255, 0, 0), 2);

        Point center(box.x + box.width/2, box.y + box.height/2);
        blue_centers.push_back(center);
    }

    //drawLegend(first_frame);
    //imshow("detected blue cones", first_frame);


    /* char key = (char)waitKey(0);
    if (key == 27){
        return;
    } */
} 

void yellowConesDetection(Mat& first_frame, Mat& frame_HSV, vector<Point>& yellow_centers){
    Scalar lower_yellow(0, 90, 190); //select a wider range to help detecting the further cones
    Scalar upper_yellow(40, 255, 255);

    Scalar lower_yellow2(0, 90, 174);
    Scalar upper_yellow2(90, 255, 255);

    Rect roi( 0, 200, 600, 100);
    //rectangle(first_frame, roi, Scalar(0, 255, 255), 2);

    //create 2 rectangle region for 2 separate masks, so that i can use different kernels
    Rect further_cones(0, 200, 390, 40);
    //rectangle(first_frame, further_cones, Scalar(0, 255, 255), 2);
    Rect right_cones(390, 200, 100, 100);
    //rectangle(first_frame, right_cones, Scalar(0, 255, 255), 2);


    Mat cropped_HSV1 = frame_HSV(further_cones); //crop the HSV so to compute the mask on the cropped part
    Mat cropped_HSV2 = frame_HSV(right_cones);

    Mat further_mask, right_mask;

    inRange(cropped_HSV1, lower_yellow, upper_yellow, further_mask);
    Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(1,1));
    morphologyEx(further_mask, further_mask, MORPH_OPEN, kernel1);
    morphologyEx(further_mask, further_mask, MORPH_CLOSE, kernel1);
    morphologyEx(further_mask, further_mask, MORPH_DILATE, kernel1);    

    inRange(cropped_HSV2, lower_yellow2, upper_yellow2, right_mask);
    Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, Size(6,6));
    morphologyEx(right_mask, right_mask, MORPH_CLOSE, kernel2);
    morphologyEx(right_mask, right_mask, MORPH_OPEN, kernel2);
    morphologyEx(right_mask, right_mask, MORPH_DILATE, kernel2);  


    // contours
    vector<vector<Point>> further_contours, right_contours;
    vector <Vec4i> hierarchy1, hierarchy2;

    findContours(further_mask, further_contours, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(right_mask, right_contours, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto& contour : further_contours) { //offset the counters so that they're on the right position
        for (auto& point : contour) {
            point.x += further_cones.x;
            point.y += further_cones.y;
        }
    }

    for (auto& contour : right_contours) {
        for (auto& point : contour) {
            point.x += right_cones.x;
            point.y += right_cones.y;
        }
    }

    //merge contours together
    vector<vector<Point>> contours;
    contours.insert(contours.end(), further_contours.begin(), further_contours.end());
    contours.insert(contours.end(), right_contours.begin(), right_contours.end());

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        Rect box = boundingRect(contour);
        double aspect = (double)box.height / (double)box.width;

        Point center(box.x + box.width/2, box.y + box.height/2);

        if (aspect < 0.3|| aspect > 4) continue;
        rectangle(first_frame, box, Scalar(0, 255, 255), 2);
        yellow_centers.push_back(center);
    }

    //drawLegend(first_frame);
    //imshow("detected yellow cones", first_frame);


    /* char key = (char)waitKey(0);
    if (key == 27){
        return;
    } */

}

void drawBlueEdge(Mat& first_frame, vector<Point>& blue_centers ){
    vector<Point> sorted_cones = blue_centers;
    sort(sorted_cones.begin(), sorted_cones.end(), [](const Point&a, const Point& b){
        return a.x < b.x;
    });

    float mean_x = 0;
    for (auto& p : sorted_cones) mean_x += p.x;
        mean_x /= sorted_cones.size();

    vector<Point> leftCones, rightCones;
    for (auto& p : sorted_cones) {
        if (p.x < mean_x) leftCones.push_back(p);
        else rightCones.push_back(p);
    }

    for (size_t i=1; i<leftCones.size(); ++i){
        line(first_frame, leftCones[i-1], leftCones[i], Scalar(255,0,0), 2, LINE_AA);
    }

    vector <Point> sorted_right = rightCones;
    sort(sorted_right.begin(), sorted_right.end(), [] (const Point& a, const Point& b ){
        return a.y< b.y;
    });

    for (size_t i=1; i<sorted_right.size(); ++i){
        line(first_frame, sorted_right[i-1], sorted_right[i], Scalar(255, 0, 0), 2, LINE_AA);
    }

    line(first_frame, leftCones.back(), sorted_right.front(), Scalar(255, 0, 0), 2, LINE_AA);
}

void drawYellowEdge(Mat& first_frame, vector<Point>& yellow_centers ){
    vector<Point> sorted_cones = yellow_centers;
    sort(sorted_cones.begin(), sorted_cones.end(), [](const Point&a, const Point& b){
        return a.x < b.x;
    });

    for(size_t i = 1; i<sorted_cones.size(); ++i){
        line(first_frame, sorted_cones[i-1], sorted_cones[i], Scalar(0,255,255), 2, LINE_AA );
    }
   
}

void drawRedLine(Mat& first_frame, vector<Point>& red_centers){
    for(size_t i=1; i<red_centers.size(); ++i){
        line(first_frame, red_centers[i-1], red_centers[i], Scalar(0, 0, 255), 2, LINE_AA);
    }

     Point p1 = red_centers.front();
    Point p2 = red_centers.back();
    Point mid((p1.x + p2.x) / 2, (p1.y + p2.y) / 2); //middle point of the line

    string text = "START";
    int baseline = 0;
    Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline); //measuring the size of the text to know where to position it

    Point textOrg(mid.x - textSize.width / 2, mid.y + textSize.height - 20); //position the text slightly below the line

    putText(first_frame, text, textOrg, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2, LINE_AA);
}


//---- LEVEL 5 functions -----
Mat create_mask1(Mat& first_frame){
    Mat mask_car1 = Mat::ones(first_frame.size(), CV_8UC1) * 255; // initially create a mask full of 1s (white), 8 bit unsigned since we're looking at an area of interest (greyscale)
    Rect middle_rect(190, 320, 280, 300);
    Rect left_tire1(30, 390, 160, 100);
    Rect right_tire1(470, 390, 160, 100);

    rectangle(mask_car1, middle_rect, Scalar(0), FILLED);
    rectangle(mask_car1, left_tire1, Scalar(0), FILLED);
    rectangle(mask_car1, right_tire1, Scalar(0), FILLED);

    return mask_car1;
}

Mat create_mask2(Mat& second_frame){
    Mat mask_car2 = Mat::ones(second_frame.size(), CV_8UC1) * 255;
    Rect middle_rect(190, 320, 280, 300);
    Rect left_tire2(30, 410, 160, 80);
    Rect right_tire2(470, 430, 160, 70);

    rectangle(mask_car2, middle_rect, Scalar(0), FILLED);
    rectangle(mask_car2, left_tire2, Scalar(0), FILLED);
    rectangle(mask_car2, right_tire2, Scalar(0), FILLED);

    return mask_car2;
}

Mat odometry(Mat &first_frame, Mat& second_frame){
    //create a binary mask to hide the car for both frames
    Mat mask_car1 = create_mask1(first_frame);
    Mat mask_car2 = create_mask2(second_frame);

    Mat masked_first_frame, masked_frame2;
    first_frame.copyTo(masked_first_frame, mask_car1);
    second_frame.copyTo(masked_frame2, mask_car2);
    //imshow("masked car 1", masked_first_frame);
    //imshow("masked car 2", masked_frame2);


    //find the features and their descriptors
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptor1, descriptor2; //local neighbours of each keypoint

    orb-> detectAndCompute(first_frame, mask_car1, keypoints1, descriptor1); //returns a vector of keypoints and a matrix
    orb-> detectAndCompute(second_frame, mask_car2, keypoints2, descriptor2);


    //match the features
    BFMatcher matcher(NORM_HAMMING, true); //using hamming distance to check, true means that all the matches found are symmetric, more reliable
    vector <DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches); //single best matches based on the smallest distance

    //extract the 2d points
    vector<Point2f> pt1, pt2; //points in image 1 with corresponding points (in second vector) in the second image
    for(auto &m: matches){
        pt1.push_back(keypoints1[m.queryIdx].pt); //.pt stores the x and y coordinates of the points
        pt2.push_back(keypoints2[m.trainIdx].pt);
    }

    //intrinsic matrix
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        387.3502807617188, 0,                   317.7719116210938,
        0,                 387.3502807617188,   242.4875946044922,
        0,                 0,                   1);   
    
    
    //compute the Essential Matrix
    Mat inLierMask;
    Mat E = findEssentialMat(pt1, pt2, K, RANSAC, 0.999, 1.0, inLierMask); //the mask is an output, it tells how many matches were consistent

    //how the camera actually moved (rotation matrix and translation vector)
    Mat R, t;
    recoverPose(E, pt1, pt2, K, R, t, inLierMask);

    //testing prints to see if the rotational matrix and the translation vector have plausible values
    //cout << "R" << R << endl;
    //cout << "t: " << t << endl;

    // Draw inlier matches
    vector<DMatch> inLierMatches;
    for (size_t i = 0; i < matches.size(); i++) {
        if (inLierMask.at<uchar>(i)) {
            inLierMatches.push_back(matches[i]);
        }
    }

    Mat imgMatches;
    drawMatches(first_frame, keypoints1, second_frame, keypoints2, inLierMatches, imgMatches,
                Scalar(255, 0, 0), Scalar(255, 255, 0), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    return imgMatches;
}

void menu(Mat &first_frame){
    vector<string> lines = {
        "CONTROLS",
        "[d] cones detection",
        "[e] racetrack edges",
        "[o] odometry",
        "[t] mask trackbar",
        "[r] reset",
        "[ESC] exit"
    };

    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.5;
    int thickness = 1;
    int lineSpacing = 30;

    int baseline = 0;
    int maxWidth = 0;
    for (const auto &line : lines) {
        Size textSize = getTextSize(line, fontFace, fontScale, thickness, &baseline);
        if (textSize.width > maxWidth) maxWidth = textSize.width;
    }

    int totalHeight = lines.size() * lineSpacing + 20;

    int rectX = (first_frame.cols - maxWidth) / 2 - 20;
    int rectY = (first_frame.rows - totalHeight) / 2 - 10;
    int rectWidth = maxWidth + 40;
    int rectHeight = totalHeight + 20;


    Mat overlay;
    first_frame.copyTo(overlay);
    rectangle(overlay, Rect(rectX, rectY, rectWidth, rectHeight), Scalar(30, 30, 30), FILLED);
    addWeighted(overlay, 0.5, first_frame, 0.3, 0, first_frame);

    // Draw the text lines
    int y = rectY + 40;
    for (size_t i = 0; i < lines.size(); ++i) {
        Scalar color;
        if (i == 0) color = Scalar(255, 255, 0);
        else if (lines[i].find("ESC") != string::npos) color = Scalar(0, 0, 255);
        else color = Scalar(255, 255, 255);

        putText(first_frame, lines[i],
                Point(rectX + 20, y),
                fontFace, fontScale, color, thickness + (i == 0 ? 1 : 0));
        y += lineSpacing;
    }
}



int main(){
    Mat first_frame = imread("../images/frame_1.png");
    Mat second_frame= imread("../images/frame_2.png");

    Mat frame_odometry = first_frame.clone();

    if (first_frame.empty()){
        return -1;
    }


    Mat frame_HSV;
    cvtColor(first_frame, frame_HSV, COLOR_BGR2HSV); //convert image to HSV
    
    //to create a trackbar to find the useful values for creating the different masks
    //maskTrackBar(first_frame, frame_HSV);

    //to find center points of the cones, used for track trace
    vector <Point> blue_centers;
    vector <Point> yellow_centers;
    vector <Point> red_centers;

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

    Mat odometry_result;

    bool show_detection = false;
    bool show_edges = false;
    bool show_odometry = false;
    bool show_trackbar = false;
    bool show_menu = true;

    while (true){
        Mat display = first_frame.clone();

        if (show_detection){
            //function to detect the red cones
            redConesDetection(display, frame_HSV, red_centers);
            //blue
            blueConesDetection(display, frame_HSV, blue_centers);
            //yellow
            yellowConesDetection(display, frame_HSV, yellow_centers);

            //draws a legend on top left corner for classification
            drawLegend(display);
        }

        if (show_edges){
            drawBlueEdge(display, blue_centers);
            drawYellowEdge(display, yellow_centers);
            drawRedLine(display, red_centers);
        }

        if (show_odometry){
            if (odometry_result.empty()){
                odometry_result = odometry(display, second_frame);
            }
            imshow("matches", odometry_result);
        }else{
            destroyWindow("matches");
            odometry_result.release();
        }

        if (show_trackbar){
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

            bitwise_and(first_frame, first_frame, result_mask, trial_mask);

            imshow("mask applied", result_mask);
        }else{
            destroyWindow("mask applied");
        }

        if (show_menu){
            menu(display);
        }

        imshow("Perception Task", display);

        char key = (char)waitKey(30);
        if (key == 27) break;

        switch (key){
            case 'd':
                show_detection = !show_detection;
                show_menu = !show_menu; //when detection is off show the main menu
                break;
            case 'e':
                show_edges = !show_edges;
                show_menu = !show_menu;
                break;
            case 'o':
                show_odometry = !show_odometry;
                show_menu = !show_menu;
                break;
            case 't':
                show_trackbar = !show_trackbar;
                show_menu = !show_menu;
                break;
            case 'r':
                show_detection = false;
                show_edges = false;
                show_odometry = false;
                show_trackbar = false;
                red_centers.clear();
                blue_centers.clear();
                yellow_centers.clear();
                show_menu = true;
                break;
        }
    }

    destroyAllWindows();
    return 0;
}
