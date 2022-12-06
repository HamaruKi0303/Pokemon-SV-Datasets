# Make Pokemon SV Datasets

## 必要なパッケージ

```python
import json
import pprint
import glob
import os
from loguru import logger
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm
import shutil

# image utils
import cv2
from PIL import Image, ImageFilter
from PIL import ImageDraw

# data utils
import numpy as np
import pandas as pd

# plot utils
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

# xlm utils
from xml.etree.ElementTree import Element, SubElement, ElementTree
```

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

`root_path`へ移動しておきます


```python
%cd $root_path
```

一応確認


```python
!ls
```

## パラメーター探索

画像間の変化量のパラメータを検索します．ここで探索したパラメーターを元に閾値を設定します．



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

`save_freq`フレームごとに画像間の変化量を計算し，閾値以上であればその画像を保存します．



### キャプチャー動画のリストを取得


```python
glob_path = capture_video_path + "/*.mp4"
video_list = glob.glob(glob_path, recursive=True)
pprint.pprint(video_list)
```

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

変化量をプロットします．


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

Label-Studioでラベルを追加する場合，下記のようなコードを記載する必要があります．

```xml
<?xml version='1.0' encoding='utf-8'?>
<View>
    <Image name="image" value="$image" />
    <RectangleLabels name="label" toName="image">
        <Label value="player" predicted_values="player" background="#FFA39E" />
    </RectangleLabels>
</View>
```


手作業で毎回コードを記載するのは結構骨が折れるため，図鑑のポケモンをCSVに記載するとそのデータから自動でXLMを生成するようにしました．


```python
df_pokemon_list = pd.read_csv(datasets_root_path + '/' + pokemon_list_name, index_col=0)
#df_pokemon_list = df_pokemon_list.dropna(how='all')
df_pokemon_list.head(10)
```

```bash
            name
sv_id           
0         player
1        Nyahoja
2      Floragato
3            NaN
4        Hogator
5            NaN
6            NaN
7        Kuwassu
8            NaN
9            NaN
```

### ラベルの色を設定

ラベルの色をカラーマップからRGBで抽出し，そこから16進数表記に変換します．


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


```python
len(colorcode_list)
```

### XML の生成

上記で生成した色とCSVから読み取ったポケモンの名前を使ってXMLを作成し`labeling_interface.xml`という名前で保存します．


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

保存された`labeling_interface.xml`はこんな感じです．

```xml
<?xml version='1.0' encoding='utf-8'?>
<View>
    <Image name="image" value="$image" />
    <RectangleLabels name="label" toName="image">
        <Label value="player" predicted_values="player" background="#000080" />
        <Label value="Nyahoja" predicted_values="Nyahoja" background="#000080" />
        <Label value="Floragato" predicted_values="Floragato" background="#000777" />
        <Label value="Hogator" predicted_values="Hogator" background="#000f6d" />
        <Label value="Kuwassu" predicted_values="Kuwassu" background="#001d5a" />
        <Label value="Gourton" predicted_values="Gourton" background="#002c48" />
        <Label value="Tarountula" predicted_values="Tarountula" background="#00333e" />
        <Label value="Nymble" predicted_values="Nymble" background="#003a35" />
        <Label value="Hanecco" predicted_values="Hanecco" background="#00422b" />
        <Label value="Popocco" predicted_values="Popocco" background="#004922" />
        <Label value="Watacco" predicted_values="Watacco" background="#004922" />
        <Label value="Yayakoma" predicted_values="Yayakoma" background="#005019" />
        <Label value="Hinoyakoma" predicted_values="Hinoyakoma" background="#00580f" />
        <Label value="Pamo" predicted_values="Pamo" background="#005f06" />
        <Label value="Pawmo" predicted_values="Pawmo" background="#005816" />
        <Label value="Delvil" predicted_values="Delvil" background="#005127" />
        <Label value="Hellgar" predicted_values="Hellgar" background="#005127" />
        <Label value="Youngoose" predicted_values="Youngoose" background="#004b37" />
        <Label value="Hoshigarisu" predicted_values="Hoshigarisu" background="#004448" />
        <Label value="Himanuts" predicted_values="Himanuts" background="#003d59" />
        <Label value="Kofukimushi" predicted_values="Kofukimushi" background="#00298b" />
        <Label value="Pinpuku" predicted_values="Pinpuku" background="#0007de" />
        <Label value="Ruriri" predicted_values="Ruriri" background="#000eff" />
        <Label value="Maril" predicted_values="Maril" background="#000eff" />
        <Label value="Ametama" predicted_values="Ametama" background="#001cff" />
        <Label value="Amemoth" predicted_values="Amemoth" background="#002aff" />
        <Label value="Buoysel" predicted_values="Buoysel" background="#0038ff" />
        <Label value="Upah" predicted_values="Upah" background="#0047ff" />
        <Label value="Clodsire" predicted_values="Clodsire" background="#0047ff" />
        <Label value="Koduck" predicted_values="Koduck" background="#0055ff" />
        <Label value="Kamukame" predicted_values="Kamukame" background="#0063ff" />
        <Label value="Pupurin" predicted_values="Pupurin" background="#007fff" />
        <Label value="Purin" predicted_values="Purin" background="#007fff" />
        <Label value="Ralts" predicted_values="Ralts" background="#008dff" />
        <Label value="Kirlia" predicted_values="Kirlia" background="#009bff" />
        <Label value="Sleepe" predicted_values="Sleepe" background="#00b8ff" />
        <Label value="Ghos" predicted_values="Ghos" background="#00c0ff" />
        <Label value="Pichu" predicted_values="Pichu" background="#00ceff" />
        <Label value="Pupimocchi" predicted_values="Pupimocchi" background="#00d7ff" />
        <Label value="Smoliv" predicted_values="Smoliv" background="#00edff" />
        <Label value="Usohachi" predicted_values="Usohachi" background="#00f6f8" />
        <Label value="Iwanko" predicted_values="Iwanko" background="#00fbf2" />
        <Label value="Tandon" predicted_values="Tandon" background="#00ffeb" />
        <Label value="Mukubird" predicted_values="Mukubird" background="#00fdd1" />
        <Label value="Merriep" predicted_values="Merriep" background="#00fcc4" />
        <Label value="Squawkabilly" predicted_values="Squawkabilly" background="#00fa88" />
        <Label value="Makunoshita" predicted_values="Makunoshita" background="#00fa7d" />
        <Label value="Ennewt" predicted_values="Ennewt" background="#00fc5e" />
        <Label value="Gomazou" predicted_values="Gomazou" background="#00fc54" />
        <Label value="Nacli" predicted_values="Nacli" background="#00fe2a" />
        <Label value="Koiking" predicted_values="Koiking" background="#0dff0b" />
        <Label value="Bassrao" predicted_values="Bassrao" background="#20f400" />
        <Label value="Fuwante" predicted_values="Fuwante" background="#33e800" />
        <Label value="Digda" predicted_values="Digda" background="#46dd00" />
        <Label value="Donmel" predicted_values="Donmel" background="#53d600" />
        <Label value="Mankey" predicted_values="Mankey" background="#68d500" />
        <Label value="Riolu" predicted_values="Riolu" background="#6ede00" />
        <Label value="Dojoach" predicted_values="Dojoach" background="#74e800" />
        <Label value="Tadbulb" predicted_values="Tadbulb" background="#76eb00" />
        <Label value="Numera" predicted_values="Numera" background="#7af200" />
        <Label value="Nokocchi" predicted_values="Nokocchi" background="#9bff20" />
        <Label value="Shikijika" predicted_values="Shikijika" background="#a4ff28" />
        <Label value="Maschiff" predicted_values="Maschiff" background="#b2ff34" />
        <Label value="Eleson" predicted_values="Eleson" background="#b6ff38" />
        <Label value="Shroodle" predicted_values="Shroodle" background="#c4ff34" />
        <Label value="Tamagetake" predicted_values="Tamagetake" background="#cdff2c" />
        <Label value="Tyltto" predicted_values="Tyltto" background="#f1ff0c" />
        <Label value="Meecle" predicted_values="Meecle" background="#fafa04" />
        <Label value="Yamikarasu" predicted_values="Yamikarasu" background="#ffeb00" />
        <Label value="Capsakid" predicted_values="Capsakid" background="#ffd004" />
        <Label value="Strike" predicted_values="Strike" background="#ffc10b" />
        <Label value="Flittle" predicted_values="Flittle" background="#ffbc0d" />
        <Label value="Dorobanko" predicted_values="Dorobanko" background="#ff9109" />
        <Label value="Mibrim" predicted_values="Mibrim" background="#ff6004" />
        <Label value="Flamigo" predicted_values="Flamigo" background="#c11fff" />
        <Label value="Koraidon" predicted_values="Koraidon" background="#f6c5f7" />
    </RectangleLabels>
</View>
```

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

## データセットの split

データセットの分割します．


```python
!python utils/cocosplit.py --having-annotations --multi-class -s 0.8 $anotate_full_repath $anotate_train_path $anotate_valid_path
```

分割したアノテーションファイルを元に，画像をtrainとValに仕分けします．


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


```python
move_datasets_image_file(target_dir=datasets_path + "/" + image_valid_dir, anno_path=anotate_valid_path)
```

## クラスラベルの出力

モデルの`config`ファイルに記載するクラスラベルを出力します．これをコピペして貼り付けます．


```python
c_list = []
for c in result_coco['categories']:
    print(c['name'])
    c_list.append(c['name'])
    
c_tuple = tuple(c_list)
    
```


```python
c_tuple
```

```bash
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
...
 'Watacco',
 'Yamikarasu',
 'Yayakoma',
 'Youngoose',
 'player')
```
