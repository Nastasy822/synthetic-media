from pytube import YouTube
import os
import cv2
import glob

def get_and_cut_video(url,
              from_time,
              to_time,
              sequence_path,
              ):
  yt = YouTube(url)
  yt.streams.first().download('data/')
  os.rename('data/'+yt.streams.first().default_filename, 'data/tmp_video.mp4')

  load_movie_path = 'data/tmp_video.mp4'
  out_movie_path = 'data/tmp_video_new.mp4'
  os.system(f'ffmpeg -i {load_movie_path} -ss {from_time} -to {to_time} -c copy {out_movie_path}')


def get_sequence(video_file,sequence_path,):

  vidcap = cv2.VideoCapture(video_file)
  success,image = vidcap.read()
  count = 0
  frame = 0
  success = True
  if not os.path.exists(sequence_path):
            os.makedirs(sequence_path)
  while success:
      cv2.imwrite(f"{sequence_path}/frame{count}.jpg", image)
      count += 1
      success,image = vidcap.read()
      frame += 1


def get_paths_of_files_in_folder(folder_name, type_files ='csv'):
    return [x for x in glob.glob(f'{folder_name}/*.{type_files}')]

