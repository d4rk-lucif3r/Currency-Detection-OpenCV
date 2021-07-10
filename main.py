from Detection.Detector import Detector

detect_obj = Detector()
detect_obj.start()
while True:
    try:
        detect_obj.detect()
        print("[INFO] Detected Currency :{}".format(
            detect_obj.detectedCurrency))
        print("[INFO] Matching Points :{}".format(
            detect_obj.maxMatching))
    except KeyboardInterrupt:
        detect_obj.stop()