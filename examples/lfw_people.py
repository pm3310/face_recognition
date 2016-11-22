from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from face_recognition import FaceRecogniser

if __name__ == '__main__':
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    X = lfw_people.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = lfw_people.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    face_detector = FaceRecogniser(
        X_train,
        y_train,
        images_height=lfw_people.images.shape[1],
        images_width=lfw_people.images.shape[2],
        num_evals=30
    )

    face_detector.train()

    y_pred = [face_detector.predict(item) for item in X_test]

    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
