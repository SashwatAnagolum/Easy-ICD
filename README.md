# Easy-ICD: Automatic scraping and denoising of image classification datasets

## CMPSC 445 Final Project

### Framework overview

* Get class names / keywords, # images per class, noise tolerance
* Retrieve images from Flickr
* Process / resize images
* Identify unrelated images using constrastive learning
* Eliminate unrelated images
* Split data into train and test sets
* Organize data, return newly minted dataset to user

### TODO

* Investigate performance of supervised / self-supervised / sliding contrastive learning approaches for different levels of noise in scraped datasets

* Investigate required network scaling for outlier detection with dataset size, image size, and intrinsic class differentiation difficulty

* Quantify class differentiation difficulty

* Create __all__ variables in all __init__.py files


### References

* [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2004.11362)
* [Supervised Contrastive Learning](https://arxiv.org/abs/2002.05709)