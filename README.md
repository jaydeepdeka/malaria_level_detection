## Malaria Level Detection classifier

* This notebook implements a classifier using PyTorch to detect different stages of the malaria.
* The dataset used for this project has been downloaded from [kaggle](https://www.kaggle.com/kmader/malaria-bounding-boxes). The dataset contains total 1364 images with (~80000) cells annotated by human researchers in different categories.
* In each of the images, tens of blood smears are present. There are two JSON files in the dataset, which contains details about:
 * Image **path**
 * **shape** containing size of the image and number of channels
 * **objects** containing `lower left co-ordinates` and `upper right co-ordinates` of the the blood smears and `category` of the smear.
* We have used Python to crop out each cell using the co-ordinates of the images and save it to the respective folders created for each category. The script `crop_utils.py` uses opencv, pandas and other libraries.
 * Raw Image:
 ![Raw Image](contents/sample_image.png?raw=true "RAW IMAGE")
 * Cropped Image:
 ![Raw Image](contents/cropped_samples.jpg?raw=true "CROPPED IMAGE")
* Exploratory Data Analysis and data preprocessing is done as the dataset is highly imbalanced. We have used up-sampling and down-sampling to bring the data disctribution in a desired ratio. The details and implementation is in `EDA_DataPreProcessing.ipynb`.
* The processed dataset is divided into three different subsets, `train`, `valid` and `test`.

### Classifier implementation
* As the dataset size is relatively small we have used [transfer learning](https://towardsdatascience.com/what-is-transfer-learning-8b1a0fa42b4), where a pre-trained model is used and we have customized the classifier part of the model.
* As the pre-trained model, for better feature extraction we have used the model saved from `Pretrained_model.ipynb`.
* In building the model we have done:
 * Data Transformation: [torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html) module has been used for augmenting data while training to `flip`, `rotate`, `jitteruing`, ` cropping` and `normalizing`. The transformations are divided for `train` and `test and valid` separately as `test and validation` doesn't need same set of transformation.
 * We are feeding the network the dataset each epoch in batches of 16 for faster convergence.
 * We have dynamically allocatted the `device` based on availability of CUDA.
 * The `feature` network parametrs are frozen with pre-trained values and gradient calculation is set to False.
 * The customized fully connected `classifier` network uses:
  * a layer 1024 neurons, which takes input from the `feature` CNN network.
  * We have used [`ReLU`](https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU) as our activation function.
  * And a [`dropout`](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout) of 0.2 is used to turn off 20% of the neurons randomly while training reduce overfitting and make the model more robust for generalisation.
 * As a loss function [`CrossEntropyLoss`](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss) has been used as we have multiple categories.
 * Stochastic Gradient Descent([SGD](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD)) is used as the optimizer of for the network to update the parameters per batches per epoch.
 * We are decaying the learning rate at a rate of 0.2 for each 5 epoch for smooth convergence to the optima.


### Note: Error in loading the notebooks: (https://github.com/jupyter/notebook/issues/3035)
* Kindly use https://nbviewer.jupyter.org/ and paste the URLs of the notebook and hit Go!. 
