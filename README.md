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

* Build custom PyTorch SupCon and SimCLR losses

* Investigate required network scaling for outlier detection with dataset size, image size, and intrinsic class differentiation difficulty

* Quantify class differentiation difficulty

* Build random image augmentation stack for model training
	* Blurring
	* Random crop + upsample to original size
	* Color distortion
	* CutOut
	* Rotation
	* Reflection

### Ideas:

* Create a (hopefully) training procedure that is robust to dataset noise by initially using the SimCLR loss, and then slowly switching to the SupCon loss as training continues. 

* Can we frame the labelling problem as an adversarial game of some kind? Or some kind of two-player game where we can use a self-supervised representation learner, and a linear classifer that uses transferred representationg (modulo some finetuning)?
	* 

### References

* [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2004.11362)
* [Supervised Contrastive Learning](https://arxiv.org/abs/2002.05709)