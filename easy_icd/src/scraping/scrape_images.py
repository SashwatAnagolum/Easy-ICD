"""
Author: Sashwat Anagolum
"""

import flickrapi
import requests
import io
import os
import time
import itertools
import numpy as np
import json
import traceback

from PIL import Image, ImageOps
from typing import List, Union, Dict, Optional, Tuple, Any
from datetime import datetime

def scrape_images_with_search_term(flickr_handle: Any, search_term: str, 
	                               num_desired_images: int, image_dir: str,
	                               image_size: Optional[Tuple[int, int]] = None,
	                               crop_to_fit: Optional[bool] = False,
	                               images_size_ext: Optional[str] = '') -> Dict:
	"""
	Scrape images from flickr by searching for a specified search term.

	Args:
		flickr_handle (Any): handle to the flickr API.
		search_term (str): the search term to scrape images based on.
		num_desired_images (int): the maximum number of images to scrape.
		image_dir (str): the folder to store the scraped images.
		image_size (tuple): the size of the images to be stored. Defaults to None,
			in which case the images are stored without resizing.
		crop_to_fit (bool, optional): bool indicating whether the images should be
			cropped to fit or not. If False, the images will be resized and then
			padded to fit the desired image size before saving.
		images_size_ext (str, optional): string extension for the image urls
			that can be used to fetch higher resolution images if the user wants to 
			save large image sizes.

	Returns:
		(dict): summarized scraping results.
	"""
	url_string = 'https://live.staticflickr.com/{}/{}_{}.jpg'
	image_file_name_string = '{}.jpg'

	if images_size_ext != '':
		images_size_ext = '_' + images_size_ext

	info_dict = dict()
	
	existing_image_names = [i[:-4] for i in os.listdir(image_dir) if '.jpg' in i]
	existing_image_names_dict = {path: True for path in existing_image_names}
	num_images_scraped_already = len(existing_image_names)
	
	photo_stream = flickr_handle.walk(text=search_term, media='photos',
		sort='relevance', per_page=500)

	num_saved_images = 0
	scraping_time_start = time.time()
	
	while (num_saved_images < num_desired_images):
		try:
			photo = next(iter(photo_stream))
		except StopIteration:
			break

		photo_secret = photo.get('secret') + images_size_ext
		photo_server_id = photo.get('server')
		photo_id = photo.get('id')

		photo_url = url_string.format(photo_server_id, photo_id, photo_secret)
		photo_unique_id = photo_server_id + '_' + photo_id + '_' + photo_secret

		try:
			if existing_image_names_dict[photo_unique_id]:
				pass
		except KeyError:
			response = requests.get(photo_url)

			if response.status_code == 200:
				image_file = io.BytesIO(response.content)
				image = Image.open(image_file).convert('RGB')

				if image_size:
					if crop_to_fit:
						image = ImageOps.fit(image, image_size)
					else:
						image.thumbnail(image_size)    
						image = ImageOps.pad(image, image_size)
							

				image_file_name = image_file_name_string.format(
					num_images_scraped_already + num_saved_images)

				image_file_path = os.path.join(image_dir, image_file_name)

				with open(image_file_path, 'wb') as f:
					image.save(f, 'JPEG', quality=85)

				num_saved_images += 1
				existing_image_names_dict[photo_unique_id] = True
	
	scraping_time = time.time() - scraping_time_start
		
	info_dict['successful'] = bool(num_saved_images == num_desired_images)
	info_dict['num_desired_images'] = int(num_desired_images)
	info_dict['num_saved_images'] = int(num_saved_images)
	info_dict['time_taken_in_seconds'] = float(scraping_time)

	return info_dict
	
def scrape_images(class_names: List[str], image_dir: Optional[str] = None,
	              class_keywords: Optional[List[List[str]]] = None,
	              images_per_class: Union[int, List[int], List[List[int]]] = None,
	              image_size: Optional[Tuple[int, int]] = None,
	              crop_to_fit: Optional[bool] = False,
	              images_size_ext: Optional[str] = None) -> Dict:
	"""
	Scrape images using the flickr API.

	Args:
		class_names (list): class names to scrape images for.
		image_dir (str): the folder to store images in.
		class_keywords (list): keywords to search for for every class.
		images_per_class (list): how many images to scrape for each
			keyword in each class. Can also specificy a single int per class, in which 
			case the images scrape per keyword are 0evenly divided across all keywords
			for a class. Can also specify a single int, in which case the same number
			of images are scraped for each class.
		image_size (tuple, optional): desired image side lengths in pixels.
			Defaults to None, in which case images are not resized before saving.
		crop_to_fit (bool, optional): bool indicating whether the images should be
			cropped to fit or not. If False, the images will be resized and then padded
			to fit the desired image size before saving.
		images_size_ext (str, optional): string extension for the image urls
			that can be used to fetch higher resolution images if the user wants to 
			save large image sizes.

	Returns:
		(dict): summarized scraping results.
	"""
	key = "91c796378356002c5ba8be27758cada5"
	secret = "fd4f5ab352fa08e8"
	flickr = flickrapi.FlickrAPI(key, secret)
	
	info_dict = dict()
	num_classes = len(class_names)
	
	if image_dir is None:
		curr_datetime = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
		image_dir = f'./images_{curr_datetime}'
		
	if image_dir[-1] == '/':
		image_dir = image_dir[:-1]
	
	if class_keywords is None:
		class_keywords = [[class_name] for class_name in class_names]
		
	if images_per_class is None:
		images_per_class = 1000
		
	if isinstance(images_per_class, int):
		images_per_class = [images_per_class for i in range(num_classes)]
		
	if isinstance(images_per_class[0], int):
		images_per_class = [
			[np.ceil(images_per_class[i] / len(class_keywords[i])).astype(np.int32)
			for j in range(len(class_keywords[i]))] for i in range(num_classes)]
	
	if not os.path.exists(image_dir):
		os.mkdir(image_dir)
	
	total_scraping_time_start = time.time()
	
	for class_number in range(num_classes):
		class_name = class_names[class_number]
		class_dir = os.path.join(image_dir, class_name.replace(' ', '_'))
		
		num_desired_class_images = sum(images_per_class[class_number])
		num_saved_class_images = 0

		if not os.path.exists(class_dir):
			os.mkdir(class_dir)
		
		info_dict[class_name] = dict() 
		class_scraping_time_start = time.time()
		
		for search_term_number in range(len(class_keywords[class_number])):
			search_term = class_keywords[class_number][search_term_number]
			num_desired_images = images_per_class[class_number][search_term_number]
			
			scraped_images_info = scrape_images_with_search_term(flickr, search_term,
				num_desired_images, class_dir, image_size, crop_to_fit,
				images_size_ext)
			
			num_saved_class_images += scraped_images_info['num_saved_images']
			info_dict[class_name][search_term] = scraped_images_info
			
		class_scraping_time = time.time() - class_scraping_time_start
		scrape_successful = bool(num_saved_class_images == num_desired_class_images)

		info_dict[class_name]['time_taken_in_seconds'] = float(class_scraping_time)
		info_dict[class_name]['num_desired_images'] = int(num_desired_class_images)
		info_dict[class_name]['num_saved_images'] = int(num_saved_class_images)  
		info_dict[class_name]['successful'] = bool(scrape_successful)

		json_file_path = os.path.join(class_dir, 'class_scraping_info.json')
		
		with open(json_file_path, 'w') as f:
			json.dump(info_dict[class_name], f)

	total_scraping_time = time.time() - total_scraping_time_start
	info_dict['time_taken_in_seconds'] = float(total_scraping_time)

	json_file_path = os.path.join(image_dir, 'scraping_info.json')
	
	with open(json_file_path, 'w') as f:
		json.dump(info_dict, f)
	
	return info_dict