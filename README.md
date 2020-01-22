# cats_vs_dogs

Simple CNN based model to classify cats vs dogs

first make sure all requirements installed
```
pip3 install -r requirements.txt
```
and then run
```
python3 main.py --model CNN2Layers 
```
to train and test a two layered CNN or
```
python3 main.py --model CNN4Layers 
```
to train and test a 4 layered CNN .


Note for this to work you need the dataset [cats_vs_dogs](https://www.kaggle.com/c/dogs-vs-cats) (extracted) in your current directory 

### Model Graph

![Model's Graph](https://github.com/AbubakrHassan/cats_vs_dogs/blob/master/images/model_screenshot.png)

### Training loss

![Training loss](https://github.com/AbubakrHassan/cats_vs_dogs/blob/master/images/loss.png)

### Predictions

![Training loss](https://github.com/AbubakrHassan/cats_vs_dogs/blob/master/images/predictions.png)
