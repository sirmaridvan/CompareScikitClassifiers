from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
class Classifiers:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "Logistic Regression", "QDA"]

        self.classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            LogisticRegression(),
            QuadraticDiscriminantAnalysis()]

    def run_all(self):
        accuracies = []
        for name, clf in zip(self.names, self.classifiers):
            clf.fit(self.x_train, self.y_train)
            y_pred = clf.predict(self.x_test)
            print("These results are belongs to: " + name)
            self.print_info(y_pred)
            accuracies.append(accuracy_score(self.y_test, y_pred))
        return  accuracies

    def print_info(self, y_pred):
        print("Accuracy: "+str(accuracy_score(self.y_test, y_pred)))
        print('\n')
        print(classification_report(self.y_test, y_pred))
