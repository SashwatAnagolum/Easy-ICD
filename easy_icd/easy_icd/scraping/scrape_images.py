import flickrapi
import requests
import io
import os
import time
import itertools
import numpy as np
import json

from PIL import Image, ImageOps
from typing import List, Union, Dict, Optional, Tuple
from datetime import datetime

def scrape_images_with_search_term(flickr_handle, search_term: str, 
	num_desired_images: int, image_dir: str,
    image_size: Optional[Tuple[int, int]] = None) -> Dict:
    """
    Scrape images with search term:
    Scrape images from the results of a Flicker search using the passed in search
    term until num_desired_images have been saved or there are no more search 
    results. Users can optionally specify an image_size that all scraped images
    are resized to before saving.
    """
    url_string = "https://live.staticflickr.com/{}/{}_{}.jpg"
    info_dict = dict()
    
    existing_image_names = [i for i in os.listdir(image_dir) if '.jpg' in i]
    existing_image_names_dict = {path: True for path in existing_image_names}
    
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
        
            response = requests.get(photo_url)

            if response.status_code == 200:
                image_file = io.BytesIO(response.content)
                image = Image.open(image_file).convert('RGB')

                if image_size:
                    image.thumbnail(image_size)    
                    image = ImageOps.pad(image, image_size)

                image_file_name = photo_unique_id + '.jpg'
                image_file_path = os.path.join(image_dir, image_file_name)

                try:
                    if existing_image_names_dict[image_file_name]:
                        pass
                except KeyError:
                    existing_image_names_dict[image_file_name] = True

                    with open(image_file_path, 'wb') as f:
                        image.save(f, "JPEG", quality=85)

                    num_saved_images += 1
    except StopIteration:
        pass
    
    scraping_time = time.time() - scraping_time_start
        
    info_dict['successful'] = num_saved_images == num_desired_images
    info_dict['num_desired_images'] = num_desired_images
    info_dict['num_saved_images'] = num_saved_images
    info_dict['time_taken_in_seconds'] = scraping_time
    
    return info_dict
    
def scrape_images(class_names: List[str], image_dir: Optional[str] = None,
	class_keywords: Optional[List[List[str]]] = None,
    images_per_class: Union[int, List[int], List[List[int]]] = None) -> Dict:
    """
    Scrape images:
    Scrape images using the Flickr API based on the results of Flickr searches
    using the search terms passed in. Images will be saved in directories given by
    the class_names. Users can specify mutliple search terms per class in order to 
    ex. force image diversity or enable cleaner image results, as well as the
    proportion of images that should come from the  results of each search term for
    a class.
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
            	num_desired_images, class_dir)
            
            num_saved_class_images += scraped_images_info['num_saved_images']
            info_dict[class_name][search_term] = scraped_images_info
            
        class_scraping_time = time.time() - class_scraping_time_start
        scrape_successful = num_saved_class_images == num_desired_class_images

        info_dict[class_name]['time_taken_in_seconds'] = class_scraping_time
        info_dict[class_name]['num_desired_images'] = num_desired_class_images
        info_dict[class_name]['num_saved_images'] = num_saved_class_images  
        info_dict[class_name]['successful'] = scrape_successful

    total_scraping_time = time.time() - total_scraping_time_start
    info_dict['time_taken_in_seconds'] = total_scraping_time

    json_file_path = os.path.join(image_dir, 'execution_info.json')
    
    with open(json_file_path, 'w') as f:
        json.dump(info_dict, f)
    
    return info_dict