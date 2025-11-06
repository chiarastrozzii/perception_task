# 🏎️ Perception Pipeline - Formula Student Driverless Car

This project simulates a **perception pipeline** for a Formula Student Driverless Vehicle. Its main objectives are to **detect** the cones present on the racetrack, visualize the **edges** of the track and perform **odometry** using 2 images.

---


## 🖼️ Example Outputs

| Mode | Visualization |
|------|----------------|
| 🟦 **Cone Detection** | ![Cone Detection](../documents/cone_detection.png) |
| 🟨 **Edge Visualization** | ![Edge Detection](../documents/edge_visualization.png) |
| 🎚️ **Trackbar Mask Tuning** | ![Trackbar](../documents/trackbar.gif) |

---

## ⚙️ Installation & Build

### **Requirements**
- C++17 or newer  
- OpenCV 4.0+  
- CMake 3.10+  

### **Building**
```bash
git clone <your-repo-url>
cd perception_project
mkdir build && cd build
cmake ..
make
./PerceptionTask
```

Your project should have a `CMakeLists.txt` similar to:
```cmake
cmake_minimum_required(VERSION 3.10)
project(PerceptionTask)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

add_executable(PerceptionTask main.cpp)
target_link_libraries(PerceptionTask ${OpenCV_LIBS})
```

---

## 🎛️ GUI Controls

When running the program, a **main menu overlay** appears in the center of the screen.  
Use the following keyboard controls to toggle different modes, press once to active and another time to disactivate:

| Key | Mode | Description |
|-----|------|--------------|
| **s** |  *Load Data* | Loads and shows both frames given |
| **d** |  *Cone Detection* | Detects and labels red, blue, and yellow cones on the track, using the first frame. |
| **e** |  *Edge Visualization* | Shows track edges and the start line. |
| **o** |  *Odometry* | Estimates and displays motion between the 2 frames using ORB features. |
| **t** |  *Trackbar Mode* | Opens an HSV trackbar window for dynamic tuning of color thresholds. |
| **r** |  *Reset* | Resets all modes. |
| **ESC** |  *Exit* | Closes all windows |

---

## 🧩 Pipeline Architecture

###  **Level 1 — Load Data**
- Loads both images
- Visualize both

### **Level 2 — Cones Detection**
- Converts frame 1 from BGR to HSV
- Applies color thresholds with inRange().
- Performs filtering such as MORPH_OPEN, MORPH_CLOSE MORPH_DILATE.
- Extracts contours and bounding boxes for red, yellow, and blue cones.
- Computes cone centers for later edge processing.


###  **Level 3 — Edge Detection**
- Sorts detected cone centers by coordinates.  
- Connects blue and yellow cones to form racetrack boundaries.  
- Connects red cones to mark the starting line.

###  **Level 4 — Odometry**
- Masks out static regions (the car).  
- Detects **ORB** keypoints and computes descriptors on two frames.  
- Uses **Brute Force (Hamming)** matcher to find correspondences.  
- Estimates motion:
  ```cpp
  Mat E = findEssentialMat(points1, points2, K, RANSAC);
  recoverPose(E, points1, points2, K, R, t);
  ```
- Displays inliers and matched features.

###  **Level 5 — Interactive GUI**
- Uses OpenCV **HighGUI** (`imshow`, `waitKey`, `createTrackbar`).  
- Implements toggle states for visualization.  
- Trackbars allow dynamic tuning of HSV ranges:
  - `Hue Min / Max`
  - `Saturation Min / Max`
  - `Value Min / Max`
- Updates masks and display in real time.

---

## 🧪 Example of the GUI Menu Overlay

*(You can screenshot your own menu overlay here)*  

```text
-----------------------------
         CONTROLS
[d] cones detection
[e] racetrack edges
[o] odometry
[t] mask trackbar
[r] reset
[ESC] exit
-----------------------------
```

---



