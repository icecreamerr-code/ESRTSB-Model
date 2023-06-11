# Model source code the following paper is released here.

## Dependencies
- [Tensorflow](https://www.tensorflow.org) >= 2.4.0 with cuda 11
- [Python](https://www.python.org) >= 3.8
- [numpy](https://numpy.org)
- [sklearn](https://scikit-learn.org)

## Data Preparation & Preprocessing
- We give a sample raw data in the `ESRTSB-data` folder.
- Feature Engineering:
```
python feateng_php.py # for PHP

```


```
configure `data_set and model_type` from code/configure/configure.py 
    data_set='PHP'   
    model_type='PP'
```

- Pouring data into pkl:
```

python graph_storage.py 
python graph_generator.py
```

- Generate target and history sequence data
```
python gen_target.py  # generate target user and item
python sampling.py # sampling
```

## Train the Models
- To run model
```
cd ESRTSB/
python train.py
```
