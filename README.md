# Deep fake Image Classification using Kaggle dataset 

## TODO

* Install requirements
```bash
$ pip install -r requirements.txt
```
* Download dataset 

> Data can be downloaded [here](https://www.kaggle.com/c/ads5035-01/data) . Click on **Download All** and download ads5035-01.zip .

* Extract zip file and move to _data_ directory 

```bash
$ mkdir data
$ unzip ads5035-01.zip -d data/
```

## Training

* EfficientNet training
```bash
$ python effnetb0.py
```

* PyramidNet training
```bash
$ python pyramid.py
```

* XceptionNet training
```bash
$ python xception.py
```

## Predict from test data and make submission files

* EfficientNet 
```bash
$ python evaluateAndSubmit_efnet.py
```

* PyramidNet 
```bash
$ python evaluateAndSubmit_PyramidNet.py
```

* XceptionNet 
```bash
$ python evaluateAndSubmit_XcepNet.py
```

## GradCam Visualization 

* EfficientNet 
```bash
$ python GradCamEfnet.py --choose_file test --Num_img 5
```

* PyramidNet 
```bash
$ python GradCamPyramidNet.py --choose_file test --Num_img 5
```

* XceptionNet 
```bash
$ python GradCamXception.py --choose_file test --Num_img 5
```
* Images will be saved in Gradcam_Image directory

## T-SNE Visualization 

* EfficientNet 
```bash
$ bash tsneEf.sh
```

* PyramidNet 
```bash
$ bash tsnePyramid.sh
```

* XceptionNet 
```bash
$ bash tsneXcep.sh
```
* Images will saved in Tsne_Image directory

* Trained model can be downloaded from [here](https://drive.google.com/file/d/1StmCTnZkU52CZJ7ZLsKhx3aRBWvesg8H/view?usp=sharing)