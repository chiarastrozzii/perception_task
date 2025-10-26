#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;


//importing imgs

int main(){

    string path_image1 = "../images/frame_1.png";
    Mat frame1 = imread(path_image1); //matrix from opencv library where we store the image, read from the path given 

    if (frame1.empty()){
        cerr << "Could not load the image from path: " << path_image1 << endl;
        return -1;
    }
    imshow("frame 1", frame1); //with no delay, the image will be shown but closed automatically
    waitKey(0); //the image will be shown and not closed until we click the close button


    return 0;
}
