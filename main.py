from sympy import det
from Helpers.Detector import Detector

detect_obj = Detector()
detect_obj.start()
while True:
    detect_obj.detect()
    print("[INFO] Detected Currency :{}".format(
        detect_obj.detectedCurrency))
    print("[INFO] Matching Points :{}".format(
        detect_obj.maxMatching))
 
 