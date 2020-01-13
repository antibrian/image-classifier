# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


## train.py

```
Image Classifier Trainer

positional arguments:
  data_dir              directory containing testing, validation, and training
                        images

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        load pre-existing checkpoint for further training
  --arch NETWORK        pre-trained feature set model name (default: resnet50)
  --save_dir SAVE_DIR   directory in which to save checkpoint
  --learning_rate LEARNING_RATE
                        optimizer learning rate (default: .001)
  --hidden_units HIDDEN_UNITS
                        number of hidden units (default: 2)
  --epochs EPOCHS       number of training passes. 0 for no training
  --gpu                 use GPU for computation, if available
  --test                run tests against testing images
  --no-save             do not save a checkpoint
```

Example Usage:  
```
python train.py flowers --epochs 15 --gpu
```  
Will train a model on the `./flowers` directory for `15` epochs using the GPU and save a checkpoint to `./checkpoint.pth`.

## predict.py
```
Image Classifier Predictor

positional arguments:
  image_path            path to test image

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT_PATH
                        path to checkpoint
  --category_names JSON_PATH
                        path to json containing category names
  --top_k TOP_K         number of predictions to display
  --gpu                 use GPU for computation, if available
```

Example Usage:  
```
python predict.py flowers/test/1/image_06743.jpg --category_names cat_to_name.json --gpu
```  
Will load the classifier stored in `./checkpoint.pth` and provide 5 class predictions for the supplied image and categories using the GPU