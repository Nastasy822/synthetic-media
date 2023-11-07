from PIL import Image
import torch
import glob
import json
import tqdm
import math


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, dataset, image_processor, tokenizer):
        self.dataset = dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        image = Image.open(self.dataset[idx]['image_paths'])
        image = image.resize((100, 100))
        image_features = self.image_processor(image, retun_tensors="pt")['pixel_values'][0]

        labels = self.tokenizer(self.dataset[idx]["label"],
                                return_tensors="pt",
                                max_length=46,
                                pad_to_max_length=True,
                                return_token_type_ids=True,
                                truncation=True)['input_ids']

        return {'pixel_values': image_features, 'labels': labels.squeeze(0)}

    def __len__(self):
        return len(self.dataset)


class GenerateDataset:
    def __init__(self,
                 dataset_image_path,
                 dataset_label_path,
                 test_size=0.25,
                 ):
        self.dataset_image_path = dataset_image_path
        self.dataset_label_path = dataset_label_path
        self.test_size = test_size

    def __get_images_paths(self, folder_name, type_files='jpg'):
        return [x for x in glob.glob(f'{folder_name}/*.{type_files}')]

    def __generate_dataset(self, images_paths):
        with open(self.dataset_label_path) as json_file:
            labels = json.load(json_file)

        dataset = []
        for image_path in tqdm.tqdm(images_paths[:10]):
            image_name = image_path.split("/")[-1]

            for label in labels['images']:
                if label['filename'] == image_name:
                    dataset.append({'image_paths': image_path, 'label': label['sentences'][0]['raw']})

        return dataset

    def __split_names_of_files(self, data_files_names):

        count_of_data_files = len(data_files_names)

        test_files_count = math.ceil(count_of_data_files * self.test_size)

        test_files_names = data_files_names[:test_files_count]
        train_files_names = data_files_names[test_files_count:]

        return test_files_names, train_files_names

    def get_splited_dataset(self):
        images_paths = self.__get_images_paths(self.dataset_image_path)
        data_files_names = self.__generate_dataset(images_paths)
        test_files_names, train_files_names = self.__split_names_of_files(data_files_names)

        return test_files_names, train_files_names



