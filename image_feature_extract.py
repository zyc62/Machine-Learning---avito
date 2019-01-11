import os
from PIL import Image
from collections import defaultdict
import operator
from skimage import feature
import pandas as pd
import numpy as np
import cv2
from keras.preprocessing import image
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception_v3
import seaborn as sns


def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
  
    return light_percent, dark_percent

def perform_color_analysis(img,flag):
    img = 'avito_images/' + img+'.jpg' 
    im = Image.open(img) #.convert("RGB")
    
    # cut the images into two halves as complete average may give bias results
    size = im.size
   
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))
   
    
    try:
        light_percent1, dark_percent1 = color_analysis(im1)
      
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2    

    if flag == 'black':
        return round(dark_percent,2)
    elif flag == 'white':
        return round(light_percent,2)
    else:
        return None

def average_pixel_width(img):

    img = 'avito_images/' + img +  '.jpg' 
    im = Image.open(img)    
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return round(apw*100,2)

def get_blurrness_score(img):
    #path =  images_path + image
    img = 'avito_images/' + img + '.jpg'  
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return round(fm,2)

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

resnet_model = resnet50.ResNet50(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
#xception_model = xception.Xception(weights='imagenet')


def image_classify(model, pak, img, top_n=1):
    """Classify image and return top matches."""
    img = 'avito_images/' + img+ '.jpg' 
    im = Image.open(img)
    target_size = (224, 224)
    if im.size != target_size:
        im = im.resize(target_size)
    x = image.img_to_array(im)
    x = np.expand_dims(x, axis=0)
    x = pak.preprocess_input(x)
    preds = model.predict(x)
    resnet_preds = pak.decode_predictions(preds, top=top_n)[0]
    a, b, c = zip(*(resnet_preds))
   
    return round(c[0],2)



#image_name = [x.replace('avito_images/', '').replace('.jpg', '') for x in image_name]
#print(image_name)
#print(image_score)

#df = pd.DataFrame(image_name)
#df.to_csv('img_name.csv', index_label='id',)

#df = pd.DataFrame(image_score)
#df.to_csv('img_score.csv', index_label='id')

image_name = []
image_files = [x.path for x in os.scandir('avito_images')]
columns = ['image','dullness', 'whiteness','average_pixel_width','blurrness','resnet50_score']
df_ = pd.DataFrame(columns=columns)

df_ = df_.fillna(0)
 



for i in image_files:
    image_name.append(i)

image_name = [x.replace('avito_images', '').replace('.jpg', '') for x in image_name]

data = np.array(image_name).T
df_["image"]= data


    
df_['dullness'] = df_['image'].apply(lambda x : perform_color_analysis(x, 'black'))
df_['whiteness'] = df_['image'].apply(lambda x : perform_color_analysis(x, 'white'))
df_['average_pixel_width'] = df_['image'].apply(lambda x : average_pixel_width(x))
df_['blurrness'] = df_['image'].apply(get_blurrness_score)
df_['resnet50_score'] = df_['image'].apply(lambda x: image_classify(resnet_model,resnet50, x))

df_.to_csv('features.csv', index_label='id')
#print(df_)

