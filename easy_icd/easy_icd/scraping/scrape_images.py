import flickrapi
import requests
import io
import os
import time
import itertools
import numpy as np
import json

from PIL import Image, ImageOps
from typing import List, Union, Dict, Optional, Tuple, Any
from datetime import datetime

def scrape_images_with_search_term(flickr_handle: Any, search_term: str, 
	                               num_desired_images: int, image_dir: str,
	                               image_size: Optional[Tuple[int, int]] = None,
	                               crop_to_fit: Optional[bool] = False) -> Dict:
	"""
	Scrape images with search term:
	Scrape images from the results of a Flicker search using the passed in search
	term until num_desired_images have been saved or there are no more search 
	results. Users can optionally specify an image_size that all scraped images
	are resized to before saving, and specify whether the image should be resized
	and cropped to fit the desired size, or resized and padded.
	
	arguments:
	    flickr_handle: 
      		(Any) - specific flickr username
            search_term:
		(str) - term to enter into flickr search
            num_desired_images:
                (int) - number of images you want to save
            image_dir: 
            	(str) - path for directory to save scraped images
            image_size:
            	OPTIONAL Tuple[int, int] - desired size of saved images
            crop_to_fit:
            	OPTIONAL [bool] - true to crop, default = false
    
    	Returns:
            dictionary containing the following for each class name:
            	successful (bool),
            	num_desired_images (int), 
            	num_saved_images (int), 
            	time_taken_in_seconds (float)
	    
	"""
	url_string = 'https://live.staticflickr.com/{}/{}_{}.jpg'
	image_file_name_string = '{}.jpg'

	info_dict = dict()
	
	existing_image_names = [i[:-4] for i in os.listdir(image_dir) if '.jpg' in i]
	existing_image_names_dict = {path: True for path in existing_image_names}
	num_images_scraped_already = len(existing_image_names)
	
	photo_stream = flickr_handle.walk(text=search_term, media='photos',
		sort='relevance', per_page=500)

	num_saved_images = 0
	scraping_time_start = time.time()
	
	try:
		while (num_saved_images < num_desired_images):
			photo = next(iter(photo_stream))

			photo_secret = photo.get('secret')
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
	except StopIteration:
		pass
	
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
	              crop_to_fit: Optional[bool] = False) -> Dict:
	"""
	Scrape images:
	Scrape images using the Flickr API based on the results of Flickr searches
	using the search terms passed in. Images will be saved in directories given by
	the class_names. Users can specify mutliple search terms per class in order to 
	ex. force image diversity or enable cleaner image results, as well as the
	proportion of images that should come from the  results of each search term for
	a class.
	
	arguments:
            class_names: 
            	List[str] - basic search terms
            image_dir: 
            	OPTIONAL [str] - path for directory to save scraped images
            class_keywords: 
            	OPTIONAL List[List[str]] - advance search terms
            images_per_class: 
            	Union[int, List[int], List[List[int]]] - images to scrape per class
            	name.  Pass List[int] for different amounts of each 'class_names'
            	and List[List[int]] for different amounts of each 'class_keyword'
            image_size:
            	OPTIONAL Tuple[int, int] - desired size of saved images
            crop_to_fit:
            	OPTIONAL [bool] - true to crop, default = false
    
    	Returns:
            dictionary containing the following for each class name:
            	time_taken_in_seconds (float), 
            	num_desired_images (int), 
            	num_saved_images (int), 
            	successful (bool)
	    
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
				num_desired_images, class_dir, image_size, crop_to_fit)
			
			num_saved_class_images += scraped_images_info['num_saved_images']
			info_dict[class_name][search_term] = scraped_images_info
			
		class_scraping_time = time.time() - class_scraping_time_start
		scrape_successful = bool(num_saved_class_images == num_desired_class_images)

		info_dict[class_name]['time_taken_in_seconds'] = float(class_scraping_time)
		info_dict[class_name]['num_desired_images'] = int(num_desired_class_images)
		info_dict[class_name]['num_saved_images'] = int(num_saved_class_images)  
		info_dict[class_name]['successful'] = bool(scrape_successful)

	total_scraping_time = time.time() - total_scraping_time_start
	info_dict['time_taken_in_seconds'] = float(total_scraping_time)

	json_file_path = os.path.join(image_dir, 'scraping_info.json')
	
	with open(json_file_path, 'w') as f:
		json.dump(info_dict, f)
	
	return info_dict
