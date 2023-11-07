from data_utils import DataLoader, GenerateDataset
from model import ImageToTextModel

name_of_encoder = "google/vit-base-patch16-224-in21k"
name_tokenizer = "bert-base-uncased"
name_of_decoder = "bert-base-uncased"
dataset_image_path = 'image_tegging/data/train2014'
dataset_label_path = 'image_tegging/data/dataset_coco.json'


if __name__ == '__main__':

    model = ImageToTextModel()
    encoder, tokenizer = model.init_model(name_of_encoder, name_tokenizer, name_tokenizer)

    dataset_val, dataset_train = GenerateDataset(dataset_image_path,
                                                  dataset_label_path,
                                                  test_size=0.25
                                                  ).get_splited_dataset()

    dataset_train = DataLoader(dataset_train, encoder, tokenizer)

    name_test_image = dataset_val[0]['image_paths']

    dataset_test = DataLoader([dataset_val[0]], encoder, tokenizer)
    dataset_val = DataLoader(dataset_val, encoder, tokenizer)

    model.train(dataset_train, dataset_val)

    print(f'name_test_image: {name_test_image}\npredict label: {model.predict(dataset_test)}')







