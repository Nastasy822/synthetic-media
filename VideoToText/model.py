from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import numpy as np
import nltk
nltk.download('averaged_perceptron_tagger')

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}



def predict(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)

    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]

  return preds


def get_most_popular(preds):

  predicts = ''
  for predict in preds:
    predicts += predict

  return list(set(predicts.split(' ')))

def clean(words):
  functors_pos = {'NN', 'VBG'}  # function words
  return [word for word, pos in nltk.pos_tag(words) if pos in functors_pos]
