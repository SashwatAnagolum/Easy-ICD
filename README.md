# Easy-ICD: Automatic scraping and denoising of image classification datasets

#### CMPSC 445 Final Project
	
## Installation
1. Make sure you have the latest versions of Git and Python downloaded beforehand.
2. Open the command line interface in the directory you want to store Easy-ICD.
3. Enter and run the following command lines below:

```
git clone https://github.com/SashwatAnagolum/Easy-ICD.git
cd Easy-ICD/easy_icd
pip install --editable .
```
4. Additional `pip install`s such as Jupyter Notebook may be required if not already installed.

## Using Easy-ICD

See the `examples` folder for Jupyter Notebooks illustrating how to use Easy-ICD for scraping images, training an outlier detector, and using a trained model to identify and mark images for removal from a scraped dataset.

To reproduce all the figures for CIFAR-10-Alt, Cars & Deer, and Cats & Dogs datasets from scratch, use the Jupyter Notebooks in the `pretrained_models` folder. To save time by using pretrained models, load the `state_dict`s of the models saved in the `pretrained_models/models` folder intead of training from scratch. Instructions: https://pytorch.org/tutorials/beginner/saving_loading_models.html
