from video_utils import get_sequence, get_paths_of_files_in_folder
from model import predict, get_most_popular, clean


if __name__ == '__main__':
    get_sequence('data/dansing.mp4', 'data/sequence/')
    images_paths = get_paths_of_files_in_folder('data/sequence/', type_files ='jpg')
    result = predict(images_paths)

    result = get_most_popular(result)
    result = clean(result)


    print(result)

