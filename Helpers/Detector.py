import math
import os
import time


import cv2
from imutils.video import FPS


from Helpers.config import *
from Helpers.Stream import Stream


class Detector:
    def __init__(self):
        """[
            Samples the dataset and fetches all features in the images then detects the denomination of currency in video stream.
            Uses ORB to Detect Keypoints and BEBLID to compute the Images.
            Uses FLANN based Matcher to match the images in video stream.
            ]
        """
        self.cache = None
        self.featureExtractor = cv2.ORB_create()
        self.featureExtractorSupport = cv2.xfeatures2d.BEBLID_create(0.80)
        self.videoCapture = None
        self.frame = None
        self.maxMatchingsData = None
        self.maxMatchings = 0
        self.detectedCurrency = None
        self.maxMatching = None
        self.kp = None
        self.des = None
        self.matchesSummary = {}
        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.videoCapture = Stream(src=0).start()
        self.fps = FPS().start()
        # self.matcher = cv2.DescriptorMatcher_create(
        #     cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

    def start(self):
        """[Starts Sampling and Video Capture]
        """
        start = time.time()
        print('Started Sampling')
        self.getSampleData()
        end = time.time()
        print('Done sampling :', end-start)
        self.frame = self.videoCapture.read()

    def getSampleData(self):
        """[Samples the images in dataset and stores features and description of each image in a dictionary with key as the currency value]
        """
        self.cache = {}
        for currencyValue in os.listdir(BASE_DIR):
            data = []
            for currencySampleImageName in os.listdir(BASE_DIR + os.sep + currencyValue):
                currencyTrainImagePath = BASE_DIR + os.sep + \
                    currencyValue + os.sep + currencySampleImageName
                currencyTrainImage = cv2.imread(
                    currencyTrainImagePath, cv2.IMREAD_GRAYSCALE)
                self.__getFeatures(currencyTrainImage)
                data.append([currencyTrainImage, (
                    self.kp, self.des)])
            self.cache[currencyValue] = data

    def __getFeatures(self, image):
        """[Uses ORB to detect keypoints and BEBLID to compute the image]

        Args:
            image ([cv2.img]): [Image for detecting and computing keypoints]
        """
        start = time.time()
        self.image = image
        keypoints = self.featureExtractor.detect(self.image, None)
        self.kp, self.des = self.featureExtractor.compute(
            self.image, keypoints)

        end = time.time()

    def __filterFalsePositives(self, foundMatchings):
        self.foundMatchings = foundMatchings
        if not self.foundMatchings:
            return []

        self.good = []
        try:
            for m, n in self.foundMatchings:
                if m.distance < 0.7 * n.distance:
                    self.good.append(m)

        except ValueError:
            pass
        return self.good

    def __getMatchingPoints(self, queryImageDes, trainImageDes):
        """[
            Matches the queryimage description with trainimage descrption using KNN Matching.
        ]

        Args:
            queryImageDes ([array]): [Description of the image frame from the video stream which is to be detected]
            trainImageDes ([array]]): [Description of the training images]

        Returns:
            [list]: [List of Matching Points]
        """
        start = time.time()
        self.queryImageDes = queryImageDes
        self.trainImageDes = trainImageDes
        if self.queryImageDes is None or len(self.queryImageDes) == 0:
            return []
        if self.trainImageDes is None or len(self.trainImageDes) == 0:
            return []
        end = time.time()
        # print('[DEBUG] Get Matching Points :', end-start)
        try:
            return self.__filterFalsePositives(self.matcher.knnMatch(self.queryImageDes, self.trainImageDes, k=2))
        except cv2.error:
            pass

    def __buildHomographyInputData(self):
        """[Builds Homography Data]
        """
        if not (self.maxMatchingsData is None):
            queryImageData = self.maxMatchingsData[1]
            self.queryImageKeypoints = queryImageData[1][0]
            self.matches = self.maxMatchingsData[0]
            self.queryImageShape = queryImageData[0].shape

    def __guessCurrency(self):
        """[Guesses the Currency in video stream]
        """
        try:
            self.maxMatchingsCurrency = 0
            self.maxMatchings = 0
            self.detectedCurrency = 0
            self.maxMatching = 0

            self.__getFeatures(self.frame)

            for currencyValue, images in self.cache.items():
                totalMatches = 0
                for imageData in images:
                    matches = self.__getMatchingPoints(
                        imageData[1][1], self.des)
                    countMatches = len(matches)

                    if countMatches > self.maxMatchings:
                        self.maxMatchings = countMatches
                        self.maxMatchingsCurrency = currencyValue
                        self.maxMatchingsData = (matches, imageData)

                    totalMatches += countMatches
                    self.matchesSummary[currencyValue] = totalMatches

            self.__buildHomographyInputData()
            if not (self.maxMatchingsCurrency is None) and self.maxMatchings > MIN_MATCH_COUNT:
                self.detectedCurrency = self.maxMatchingsCurrency
                self.maxMatching = self.maxMatchings
            self.homographyData = (self.queryImageShape,
                                   self.queryImageKeypoints, self.kp, self.matches)
            self.matchingSummary = self.matchesSummary
        except Exception as error:
            pass

    def __processFrame(self):
        """[Processes Frame image from stream by converting it to grayScale and the calls Currency Guesser]
        """
        self.colorFrame = self.frame
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.__guessCurrency()

    def detect(self):
        """[Detects the currency in the Video Stream]
        """
        self.__processFrame()
        cv2.imshow("frame", self.frame)
        key = cv2.waitKey(1)
        self.fps.update()

    def stop(self):
        """[Exits the Currency Detector]
        """
        self.fps.stop()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
        self.videoCapture.stop()
        cv2.destroyAllWindows()
