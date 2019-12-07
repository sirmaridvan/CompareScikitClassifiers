import numpy as np
import cv2
import mahotas
import sift
from skimage import color
from skimage.feature import hog
import matplotlib.pyplot as plt

class FeatureExtractor:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.images_gray = self.convert_gray_scale(self.images)

    def convert_gray_scale(self, data):
        return [cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY) for pic in data]

    def get_hog_feature_array(self, data):
        ppc = 16
        hog_features = []
        hist_features = []
        for image in data:
            fd_hog,hog_image = hog(image, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualise=True)
            '''hist  = cv2.calcHist([image],[0],None,[256],[0,256])
            cv2.normalize(hist, hist)
            fd_hist =  hist.flatten()'''
            hog_features.append(fd_hog)
            #hist_features.append(fd_hist)

        return np.array(hog_features)

    def feature_hog(self):
        return self.get_hog_feature_array(self.images_gray)

    def get_histogram_feature_array(self, data):
        hist_features = []
        for image in data:
            hist  = cv2.calcHist([image],[0],None,[256],[0,256])
            cv2.normalize(hist, hist)
            fd_hist =  hist.flatten()
            hist_features.append(fd_hist)

        return np.array(hist_features)

    def feature_histogram(self):
        return self.get_histogram_feature_array(self.images_gray)

    def get_humoments_feature_array(self, data):
        hum_features = []
        for image in data:
            fd_humoments = cv2.HuMoments(cv2.moments(image)).flatten()
            hum_features.append(fd_humoments)

        return np.array(hum_features)

    def feature_humoments(self):
        return self.get_humoments_feature_array(self.images_gray)

    def get_haralick_feature_array(self, data):
        hera_features = []
        for image in data:
            fd_heralick = mahotas.features.texture.haralick(image.astype(int)).mean(axis=0)
            hera_features.append(fd_heralick)

        return np.array(hera_features)

    def feature_haralick(self):
        return self.get_haralick_feature_array(self.images_gray)

    def get_sift_feature_array(self,data):
        sift_feature = []
        sift = cv2.xfeatures2d.SIFT_create()
        #sift = cv2.xfeatures2d_SURF()
        #sift = cv2.ORB_create(nfeatures=1500)
        for image in data:
            kp, des = sift.detectAndCompute(image, None)
            #des.flatten()
            sift_feature.append(kp)
        return sift_feature

    def feature_sift(self):
        return self.get_sift_feature_array(self.images_gray)

    def get_all_feature_descriptors(self):
        feature_descriptor = []
        feature_descriptor.append(self.feature_hog())
        feature_descriptor.append(self.feature_haralick())
        feature_descriptor.append(self.feature_humoments())
        feature_descriptor.append(self.feature_histogram())
        return feature_descriptor
