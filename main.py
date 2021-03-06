from data_helper import prepare_files, BASE_PATH, TRAIN_PATH, VALID_PATH, TEST_PATH, get_nb_training_files, get_nb_validation_files
from CNN import CNN
import os

prepare_files()

n_training_files = get_nb_training_files()
n_validation_files = get_nb_validation_files()

if __name__ == '__main__':
    model = CNN(
        train_data_path=str(TRAIN_PATH),
        valid_data_path=str(VALID_PATH),
        test_data_path=str(TEST_PATH),
        learning_rate=0.001,
        image_size=(128,128),
        batch_size=64,
        n_classes=12,
        base_path=BASE_PATH,
        n_training_files=n_training_files,
        n_validation_files=n_validation_files
    )
    model.train(n_epochs=10)
    model.test(os.path.join(str(BASE_PATH), 'submission.csv'))