# Currency Detection using OpenCV


<p align="center">
  <a target="_blank"><img src="https://www.meshcookie.com/wp-content/uploads/2015/03/opencv_logo-1.png" /></a>&nbsp;&nbsp;&nbsp;&nbsp;
</p>

---

OpenCV to detect and idnetify the denomination of Indian Currency from web cam stream. It uses ORB to Detect Keypoints and BEBLID to compute the Images and FLANN based Matcher to match the images in video stream.

## Installation

    pip install -r requirements.txt

## Usage

    from Helpers.Detector import Detector
    detect_obj = Detector() # Object Creation
    detect_obj.start() # Initialising the Detector
    while True:
        detect_obj.detect() # Detecting Images in video Stream
