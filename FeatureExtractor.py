import numpy as np
import cv2
from skimage import color
from skimage.feature import hog
import mahotas

class FeatureExtractor:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def convert_gray_scale(self, data):
        return [ color.rgb2gray(i) for i in data]

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
        data_gray = self.convert_gray_scale(self.images)
        return self.get_hog_feature_array(data_gray)

    def get_hum_feature_array(self, data):
        hum_features = []
        for image in data:
            fd_humoments = cv2.HuMoments(cv2.moments(image)).flatten()
            hum_features.append(fd_humoments)

        return np.array(hum_features)

    def feature_hum(self):
        data_gray = self.convert_gray_scale(self.images)
        return self.get_hum_feature_array(data_gray)

    def get_heralick_feature_array(self, data):
        hera_features = []
        for image in data:
            fd_heralick = mahotas.features.texture.haralick(image.astype(int)).mean(axis=0)
            hera_features.append(fd_heralick)

        return np.array(hera_features)

    def feature_heralick(self):
        data_gray = self.convert_gray_scale(self.images)
        return self.get_heralick_feature_array(data_gray)
