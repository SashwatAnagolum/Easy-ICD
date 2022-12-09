# Easy-ICD: Automatic scraping and denoising of image classification datasets

#### CMPSC 445 Final Project
	
## Installation

```
git clone https://github.com/SashwatAnagolum/Easy-ICD.git
cd Easy-ICD/easy_icd
pip install --editable .
```

## Using Easy-ICD

See the `examples` folder for notebooks illustrating how to use Easy-ICD to scrape images, train an outlier detector, and use a trained model to identify and mark images for removal from a scraped dataset.

To reproduce all the figures for CIFAR-10-Alt, Cars & Deer, and Cats & Dogs from scratch, use the notebooks in the `pretrained_models` folder. To save time by using pretrained models, load the `state_dict`s of the models saved in the `pretrained_models/models` folder intead of training from scratch.