
from sklearn.kernel_approximation import Nystroem
from sklearn import datasets, svm

def main():
    print("hello world")
    X, y = datasets.load_digits(n_class=9, return_X_y=True)
    data = X / 16.
    clf = svm.LinearSVC()
    feature_map_nystroem = Nystroem(gamma=.2, random_state=1, n_components=300)
    data_transformed = feature_map_nystroem.fit_transform(data)
    clf.fit(data_transformed, y)
    score = clf.score(data_transformed, y)
    print(score)

if __name__ == "__main__":
    main()