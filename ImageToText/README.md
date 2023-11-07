# Image to text

## Work with repo

### Preparing the environment for work
    
    poetry shell
    poetry install
    
   
### Train model
    python image_tegging/train_runner.py
    

### Monitoring in tensorbord

1. execute in terminal:
    ```bash
    tensorboard --logdir='logs'
    ```
2. go to the url in your browser. Tensorboard doesn't work in safari 
    ```
    http://localhost:6006/
    ```
## Dataset
1. images 
https://cocodataset.org/#download

2. labels
https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

## Result  
<img src="images/COCO_train2014_000000223373.jpg" width="50%" />

**predict label by trained model:** a man holding a baseball bat while standing next to home plate.

## Conclusion 

The model predicts generally well, despite the fact that there are words that do not fit into the context.


