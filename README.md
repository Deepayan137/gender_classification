# gender_classification
This repository contains the codes for SMAI course project for the year 2017

## To run the code 

``` bash
python test.py -c configs/faces.json -o results/ -d old 

```
## Parameters to be set
1. -c path of configuration file
2. -o path of output directory
3. -d data set to be used

## Methodology
We compute eigen face for each face image and use SVM classifier with an rbf kernel to do a binary 
classifucation between fale and female face images.

The below image shows the genarated eigen faces for the Training dataset

![Eigenfaces](Images/eigen.png)

## Results

![Results](Images/facescrub_stats.png)

![Acuracy, F1 vs. n_comp](Images/stats.png)

# Real time gender classification

## To run the gender classification code on real time data

```bash
python videocap.py -c configs/faces.json -o results/ -d facescrup 
```
![Results](Images/AS.png)
## To run the BoW code 

``` bash
python test_bow.py configs/faces.json
```
 ## To run logistic regression

 ```bash
 python test_lr.py configs/faces.json
 ```

 
