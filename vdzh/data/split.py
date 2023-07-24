import os

from sklearn.model_selection import train_test_split



def split(root, train_folder="trainset", validation_folder="valset", validation_size=0.2, random_state=42):
    """
    Splits the dataset in training and validation
    """
    images = "JPEGImages"
    annotations = "Annotations"

    os.makedirs(os.path.join(root, validation_folder), exist_ok=True)
    os.makedirs(os.path.join(root, validation_folder, images), exist_ok=True)
    os.makedirs(os.path.join(root, validation_folder, annotations), exist_ok=True)

    tot_list = os.listdir(os.path.join(root, train_folder, images))
    train_list, val_list = train_test_split(tot_list, test_size=0.2, random_state=42)

    val_annotations = [x.split(".")[0] + ".xml" for x in val_list]

    for image, annotation in zip(val_list, val_annotations):
        os.rename(os.path.join(root, train_folder, images, image), os.path.join(root, validation_folder, images, image))
        os.rename(os.path.join(root, train_folder, annotations, annotation), os.path.join(root, validation_folder, annotations, annotation))