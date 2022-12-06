# Make Pokemon SV Datasets


```python
import json
import pprint
from PIL import Image, ImageFilter
from PIL import ImageDraw

import glob
import re
import os
from loguru import logger
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm
import numpy as np

import random


import cv2
import sys
import shutil

import numpy as np
import matplotlib.pyplot as plt
from xml.etree.ElementTree import Element, SubElement, ElementTree
import pandas as pd

from matplotlib.colors import rgb2hex
```


```python
!pip install pandas
```

    Requirement already satisfied: pandas in /usr/local/lib/python3.11/site-packages (1.5.2)
    Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.11/site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/site-packages (from pandas) (2022.6)
    Requirement already satisfied: numpy>=1.21.0 in /usr/local/lib/python3.11/site-packages (from pandas) (1.23.5)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/site-packages (from python-dateutil>=2.8.1->pandas) (1.16.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m

## Setting param

データセットのパスや動画のフォルダなどのパラメーターを設定します．


```python
root_path = "/home/Pokemon-SV"
datasets_root = "/home/Pokemon-SV-Datasets"

capture_dir = "capture"
capture_video_dir = "video"
capture_image_dir = "image"

capture_video_path = datasets_root  + "/" + capture_dir + "/" + capture_video_dir
capture_image_path = datasets_root  + "/" + capture_dir + "/" + capture_image_dir

diff_image_th1 = 0.50 * 1e9
diff_image_th2 = 0.85 * 1e9
save_freq     = 100
skip_freq     = 100

datasets_dir  = "datasets"
datasets_ver  = "v2.2"
datasets_path = datasets_root  + "/" + datasets_dir + "/" + datasets_ver
datasets_root_path = datasets_root  + "/" + datasets_dir
os.makedirs(datasets_path, exist_ok=True)

anotate_full = datasets_path + "/result.json"
anotate_full_repath = datasets_path + "/result_repath.json"

anotate_train_name = "pokemon_sv_train.json"
anotate_train_path = datasets_path + "/" + anotate_train_name
anotate_valid_name = "pokemon_sv_valid.json"
anotate_valid_path = datasets_path + "/" + anotate_valid_name

image_full_dir  = "images"
image_train_dir = "train2017"
image_valid_dir = "val2017"

pokemon_list_name = "pokemon_list.csv"

cm = "jet"

diff_list_total = []
```


```python
%cd $root_path
```

    /home/Pokemon-SV



```python
!ls
```

    Dockerfile	    README.md  _datasets	   docs      utils
    Make_Pokemon_SV.md  _capture   docker-compose.yml  notebook


## パラメーター探索

動画の変化量のパラメータを検索します．



```python
def param_analysis_video(video_path):
    
    video_name = video_path.split("/")[-1]
    video_single_path = capture_image_path + "/" + video_name
    logger.info("{:>20} : {}".format("video_single_path", video_single_path))
    os.makedirs(video_single_path, exist_ok=True)
       
    
    cap = cv2.VideoCapture(video_path)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #total_frame_count = 100
    
    count = 0
    image_id = 1
    
    diff_list = []
    image_pix_list = []
    
    #while True:
    for _ in tqdm(range(total_frame_count)):
        ret, frame = cap.read()

        # 読み込み可能かどうか判定
        if ret:
            #logger.info("========================")
            #logger.info("{:>20} : {}".format("count", count))
            
            # 0番目は pre frameに登録のみで処理はskip
            if(count==0):
                pre_frame = frame
            else:
                # 0番目以降は処理
                
                image_pix_list.append(np.sum(np.abs(frame)))
                # 差分を計算
                diff_image = np.sum(np.abs(pre_frame - frame))
                #logger.info("{:>20} : {}".format("diff_image", diff_image))
                diff_list.append(diff_image)
                pre_frame = frame
                
                
            count += 1
        else:
            logger.info("Video Fin ...")
            break
            
    return diff_list, image_pix_list
```


```python
def param_analysis_video_section():
    for video_path in video_list:
        logger.info("{:>20} : {}".format("video_path", video_path))
        diff_list, image_pix_list = param_analysis_video(video_path)
    
    return diff_list, image_pix_list
```


```python
#diff_list_total, image_pix_list_total = param_analysis_video_section()
```


```python
if(len(diff_list_total) > 0):
    plt.figure(figsize=[20,4.2])
    plt.hist(diff_list_total, bins=30)
```

## キャプチャー動画の分解

キャプチャーした動画を分解して画像に変換します．

変換のない静止した状態の画像はスキップした上で，`save_freq`フレームごとに画像を保存します．


### キャプチャー動画のリストを取得


```python
glob_path = capture_video_path + "/*.mp4"
video_list = glob.glob(glob_path, recursive=True)
pprint.pprint(video_list)
```

    ['/home/Pokemon-SV-Datasets/capture/video/2022-11-19 20-26-07.mp4']


### 動画の分解と保存


```python
def analysis_video(video_path):
    
    video_name = video_path.split("/")[-1]
    video_single_path = capture_image_path + "/" + video_name
    logger.info("{:>20} : {}".format("video_single_path", video_single_path))
    os.makedirs(video_single_path, exist_ok=True)
       
    
    cap = cv2.VideoCapture(video_path)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #total_frame_count = 2000
    
    count = 0
    image_id = 0
    count_save = 0
    diff_list1 = []
    diff_list2 = []
    
    
    #while True:
    for _ in tqdm(range(total_frame_count)):
        ret, frame = cap.read()

        # 読み込み可能かどうか判定
        if ret:
            #logger.info("========================")
            #logger.info("{:>20} : {}".format("count", count))
            
            # 0番目は pre frameに登録のみで処理はskip
            if(count==0):
                pre_frame = frame
            else:
                # 0番目以降は処理
                
                if(count % skip_freq == 0):
                
                    # 差分を計算
                    diff_image = np.sum(np.abs(pre_frame - frame))


                    # 閾値以上なら処理する
                    if(diff_image > diff_image_th1):

                        save_image_name = "{:09d}.jpg".format(image_id)
                        save_image_path = video_single_path + "/" + save_image_name
                        #logger.info("{:>20} : {}".format("frame[pix]", np.sum(np.abs(frame))))
                        #logger.info("{:>20} : {}".format("save_image_path", save_image_path))
                        cv2.imwrite(save_image_path, frame)

                        pre_save_frame = frame.copy()
                        image_id += 1
                        
                pre_frame = frame
                
                
            count += 1
        else:
            logger.info("Video Fin ...")
            break
        
    return diff_list1, diff_list2
        
```


```python
def video_section():
    for video_path in video_list:
        logger.info("{:>20} : {}".format("video_path", video_path))
        diff_list1, diff_list2 = analysis_video(video_path)
        return diff_list1, diff_list2
```


```python
diff_list1, diff_list2 = video_section()
```

    2022-11-25 09:50:54.682 | INFO     | __main__:video_section:3 -           video_path : /home/Pokemon-SV-Datasets/capture/video/2022-11-19 20-26-07.mp4
    2022-11-25 09:50:54.684 | INFO     | __main__:analysis_video:5 -    video_single_path : /home/Pokemon-SV-Datasets/capture/image/2022-11-19 20-26-07.mp4
    100%|█████████▉| 145757/145776 [10:02<00:00, 307.75it/s]2022-11-25 10:00:57.390 | INFO     | __main__:analysis_video:58 - Video Fin ...
    100%|█████████▉| 145775/145776 [10:02<00:00, 241.92it/s]



```python
if(len(diff_list1) > 0):
    plt.figure(figsize=[20,4.2])
    plt.hist(diff_list1, bins=30)
```


```python
if(len(diff_list2) > 0):
    plt.figure(figsize=[20,4.2])
    plt.hist(diff_list2, bins=30)
```

## Make Labeling Interface


```python
df_pokemon_list = pd.read_csv(datasets_root_path + '/' + pokemon_list_name, index_col=0)
#df_pokemon_list = df_pokemon_list.dropna(how='all')
df_pokemon_list.tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
    </tr>
    <tr>
      <th>sv_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>420</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>421</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>422</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>423</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>424</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>425</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>426</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>427</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>428</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>429</th>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

n_samples = 420

#cmap_list = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',]
#cmap_list = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
#                        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#                          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#                          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

cmap_list = ['gist_ncar']

colorcode_list = []

for cmap_name in cmap_list:
    cmap = plt.get_cmap(cmap_name)
    for k, x in enumerate(np.linspace(0.0, 1.0, n_samples)):
        colorcode = rgb2hex(cmap(x))
        colorcode_list.append(colorcode)

        
print(f"{k:02d}/{n_samples}\t{x:0.3f}\t{colorcode}")
```

    419/420	1.000	#fef8fe



```python
len(colorcode_list)
```




    420




```python
#random.seed(0) # 乱数シードを314に設定
#colorcode_list_rnd = colorcode_list.copy()
#random.shuffle(colorcode_list_rnd)
#colorcode_list_rnd
```


```python

```


```python
root = Element("View")

element1 = Element("Image")
element1.set("name", "image")
element1.set("value", "$image")
root.append(element1)

element2 = Element("RectangleLabels")
element2.set("name", "label")
element2.set("toName", "image")
root.append(element2)

for i in tqdm(range(len(df_pokemon_list))):
    pokemon_name = df_pokemon_list.loc[i, ["name"]].values[0]
    if(pokemon_name == pokemon_name):
        sub_element2 = SubElement(element2, "Label")
        sub_element2.set("value", pokemon_name)
        sub_element2.set("predicted_values", pokemon_name)
        sub_element2.set("background", colorcode_list[i])

tree = ElementTree(root)

xml_file_name = "labeling_interface.xml"
xml_file_path = datasets_path + "/" + xml_file_name
print(xml_file_path)
with open(xml_file_path, "wb") as file:
    tree.write(file, encoding='utf-8', xml_declaration=True)
```

    100%|██████████| 430/430 [00:00<00:00, 2724.40it/s]


    /home/Pokemon-SV-Datasets/datasets/v2.0/labeling_interface.xml


## 画像をアノテーション

こちらのアノテーションソフトを使ってアノテーションしていきます．

https://github.com/makiMakiTi/label-studio-1.6.0

下記のコマンドにて実行可能です．

```bash
docker-compose up --build
```

## アノテーションファイルの修正

exportされたアノテーションファイル`datasets\v0\result.json`は画像のパスが`COCO`フォーマットになっていないので修正します．



読み込みます


```python
with open(anotate_full, 'rt', encoding='UTF-8') as annotations:
    result_coco = json.load(annotations)
```

パスを修正しファイル名にします．


```python
for i in range(len(result_coco["images"])):
    file_name = result_coco["images"][i]['file_name']    
    result_coco["images"][i]['file_name'] = file_name.split("/")[-1]
```

書き出します．


```python
with open(anotate_full_repath, 'wt', encoding='UTF-8') as coco:
        json.dump(result_coco, coco, indent=2, sort_keys=True)
```


```python
anotate_full_repath
```




    '/home/Pokemon-SV-Datasets/datasets/v2.2/result_repath.json'



## データセットの split

データセットの分割します．


```python
!python utils/cocosplit.py --having-annotations --multi-class -s 0.8 $anotate_full_repath $anotate_train_path $anotate_valid_path
```

    Saved 2221 entries in /home/Pokemon-SV-Datasets/datasets/v2.2/pokemon_sv_train.json and 555 in /home/Pokemon-SV-Datasets/datasets/v2.2/pokemon_sv_valid.json



```python
def move_datasets_image_file(target_dir, anno_path):
    
    logger.info("{:>20} : {}".format("target_dir", target_dir))
    logger.info("{:>20} : {}".format("anno_path", anno_path))
    os.makedirs(target_dir, exist_ok=True)
    
    with open(anno_path, 'rt', encoding='UTF-8') as annotations:
        result_coco = json.load(annotations)

    for i in tqdm(range(len(result_coco["images"]))):
        #logger.info(">>>>>>>>>>>> {:>20} : {}".format("i", i))
        
        file_name = result_coco["images"][i]['file_name']   
        #logger.info("{:>20} : {}".format("file_name", file_name))
        
        source_path =  datasets_path + "/" + image_full_dir + "/" + file_name
        #logger.info("{:>20} : {}".format("source_path", source_path))
        
        target_path =  target_dir + "/" + file_name
        #logger.info("{:>20} : {}".format("target_path", target_path))
        
        shutil.copyfile(source_path, target_path)
        
    #pprint.pprint(result_coco)
```


```python
move_datasets_image_file(target_dir=datasets_path + "/" + image_train_dir, anno_path=anotate_train_path)
```

    2022-11-29 07:08:17.861 | INFO     | __main__:move_datasets_image_file:3 -           target_dir : /home/Pokemon-SV-Datasets/datasets/v2.2/train2017
    2022-11-29 07:08:17.867 | INFO     | __main__:move_datasets_image_file:4 -            anno_path : /home/Pokemon-SV-Datasets/datasets/v2.2/pokemon_sv_train.json
    100%|██████████| 837/837 [01:06<00:00, 12.51it/s]



```python
move_datasets_image_file(target_dir=datasets_path + "/" + image_valid_dir, anno_path=anotate_valid_path)
```

    2022-11-29 07:09:24.845 | INFO     | __main__:move_datasets_image_file:3 -           target_dir : /home/Pokemon-SV-Datasets/datasets/v2.2/val2017
    2022-11-29 07:09:24.847 | INFO     | __main__:move_datasets_image_file:4 -            anno_path : /home/Pokemon-SV-Datasets/datasets/v2.2/pokemon_sv_valid.json
    100%|██████████| 286/286 [00:23<00:00, 12.27it/s]



```python

```


```python

```


```python
c_list = []
for c in result_coco['categories']:
    print(c['name'])
    c_list.append(c['name'])
    
c_tuple = tuple(c_list)
    
```

    Amemoth
    Ametama
    Bassrao
    Buoysel
    Capsakid
    Clodsire
    Delvil
    Digda
    Dojoach
    Donmel
    Dorobanko
    Eleson
    Ennewt
    Flamigo
    Flittle
    Floragato
    Fuwante
    Ghos
    Gomazou
    Gourton
    Hanecco
    Hellgar
    Himanuts
    Hinoyakoma
    Hogator
    Hoshigarisu
    Iwanko
    Kamukame
    Kirlia
    Koduck
    Kofukimushi
    Koiking
    Koraidon
    Kuwassu
    Makunoshita
    Mankey
    Maril
    Maschiff
    Meecle
    Merriep
    Mibrim
    Mukubird
    Nacli
    Nokocchi
    Numera
    Nyahoja
    Nymble
    Pamo
    Pawmo
    Pichu
    Pinpuku
    Popocco
    Pupimocchi
    Pupurin
    Purin
    Ralts
    Riolu
    Ruriri
    Shikijika
    Shroodle
    Sleepe
    Smoliv
    Squawkabilly
    Strike
    Tadbulb
    Tamagetake
    Tandon
    Tarountula
    Tyltto
    Upah
    Usohachi
    Watacco
    Yamikarasu
    Yayakoma
    Youngoose
    player



```python
c_tuple
```




    ('Amemoth',
     'Ametama',
     'Bassrao',
     'Buoysel',
     'Capsakid',
     'Clodsire',
     'Delvil',
     'Digda',
     'Dojoach',
     'Donmel',
     'Dorobanko',
     'Eleson',
     'Ennewt',
     'Flamigo',
     'Flittle',
     'Floragato',
     'Fuwante',
     'Ghos',
     'Gomazou',
     'Gourton',
     'Hanecco',
     'Hellgar',
     'Himanuts',
     'Hinoyakoma',
     'Hogator',
     'Hoshigarisu',
     'Iwanko',
     'Kamukame',
     'Kirlia',
     'Koduck',
     'Kofukimushi',
     'Koiking',
     'Koraidon',
     'Kuwassu',
     'Makunoshita',
     'Mankey',
     'Maril',
     'Maschiff',
     'Meecle',
     'Merriep',
     'Mibrim',
     'Mukubird',
     'Nacli',
     'Nokocchi',
     'Numera',
     'Nyahoja',
     'Nymble',
     'Pamo',
     'Pawmo',
     'Pichu',
     'Pinpuku',
     'Popocco',
     'Pupimocchi',
     'Pupurin',
     'Purin',
     'Ralts',
     'Riolu',
     'Ruriri',
     'Shikijika',
     'Shroodle',
     'Sleepe',
     'Smoliv',
     'Squawkabilly',
     'Strike',
     'Tadbulb',
     'Tamagetake',
     'Tandon',
     'Tarountula',
     'Tyltto',
     'Upah',
     'Usohachi',
     'Watacco',
     'Yamikarasu',
     'Yayakoma',
     'Youngoose',
     'player')




```python

```
