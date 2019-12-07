This project aims to compare ML classifiers on the data set described below.

Classifiers:
    *Nearest Neighbors
    *Linear SVM
    *RBF SVM
    *Gaussian Process
    *Decision Tree
    *Random Forest
    *Neural Net
    *AdaBoost
    *Naive Bayes
    *QDA

Feature Descriptors:
    *Histogram of Oriented Gradients
    *HuMoments
    *Haralick
    
Dataset:
    The dataset consists of image chips extracted from Planet satellite imagery collected over the San Francisco Bay and San Pedro Bay areas of California.
    It includes 4000 80x80 RGB images labeled with either a "ship" or "no-ship" classification.
    Image chips were derived from PlanetScope full-frame visual scene products, which are orthorectified to a 3 meter pixel size.

Related Kaggle Problem:
    https://www.kaggle.com/rhammell/ships-in-satellite-imagery

Start point kernel:
    https://www.kaggle.com/manikg/training-svm-classifier-with-hog-features
