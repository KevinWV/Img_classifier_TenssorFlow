import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import os
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def process_image(img):
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, (224, 224), method = 'nearest')
    img = tf.cast(img, tf.float32)
    img /= 255
    img = img.numpy()
    return img
                
def predict(image_path, model, top_k):
    img = Image.open(image_path)
    img = np.asarray(img)
    img_proc = process_image(img)
    img_proc = np.expand_dims(img_proc, axis=0)
    p = model.predict(img_proc)
    classes = np.argpartition(p[0], -top_k)[-top_k:]
    r=classes[np.argsort(p[0][classes])][::-1]
    return (r+1).astype(str), p[0][r]
  
                
                
def main():
    print('Using:')
    print('\t\u2022 TensorFlow version:', tf.__version__)
    print('\t\u2022 tf.keras version:', tf.keras.__version__)
    print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')
    
    args = sys.argv[1:]
    class_names = {}
    k = 1
    path_json = ''
                
    if len(args) < 2 or len(args) > 6:
        print('Invalid number of arg :(')
        exit()
    
    img_path = args[0]
    print('Loading model......')
    model = tf.keras.models.load_model(args[1], custom_objects={'KerasLayer':hub.KerasLayer})
    print('Model loaded.')
                
    if len(args) > 2:
        n = len(args[2:])
        a = np.array(args[2:]).reshape(n//2,2)
        for r in a:
            if r[0] not in ('--category_names', '--top_k'):
                print('Unrecognize arg')
                exit()
            if r[0] == '--top_k':
                k = int(r[1])
            if r[0] == '--category_names':
                path_json = r[1]
    if len(path_json) > 4:
        with open(path_json, 'r') as f:
            class_names = json.load(f)
     
    c, p = predict(img_path, model,k)
    if len(class_names.keys()) > 0:
        c = [class_names[x] for x in c]
     
    print('K-Perdictions sorted by likelyhood:')
    print(c)
    print('Associed probabilities are:')
    print(p)
                
if __name__ == '__main__':
    main()                
