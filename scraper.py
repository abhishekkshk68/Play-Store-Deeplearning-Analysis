# -*- coding: utf-8 -*-
# This code belongs to Daniel Liu. The only changes done for the purpose of this study, is done in the Reviews function.
# A list of App Ids is passed to this function to fetch reviews for mentioned app ids. 
# The call to this function is given in MultipleAppsReviews.py file listed in this branch.

import logging
import json
import pandas as pd
try:
    from urllib import quote_plus
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urljoin, quote_plus
try:
    basestring
except NameError:
    basestring = str

import requests
from bs4 import BeautifulSoup, SoupStrainer
import cssutils

from play_scraper import settings as s
from play_scraper.constants import HL_LANGUAGE_CODES, GL_COUNTRY_CODES
from play_scraper.lists import AGE_RANGE, CATEGORIES, COLLECTIONS
from play_scraper.utils import (
    build_collection_url,
    build_url,
    generate_post_data,
    multi_futures_app_request,
    parse_app_details,
    parse_card_info,
    send_request,
)


class PlayScraper(object):
    
	print("In Scraper - Entering PlayScraper  \n")
	#print(object)
	#print("In Scraper - After printing Object PlayScraper  \n")
	def __init__(self, hl='en', gl='us'):
	    #print("In Scraper - in init Now print h1 value\n")
	    #print(hl)
            self.language = hl
            if self.language not in HL_LANGUAGE_CODES:
				raise ValueError('{hl} is not a valid language interface code.'.format(
				hl=self.language))
            self.geolocation = gl
            if self.geolocation not in GL_COUNTRY_CODES:
				raise ValueError('{gl} is not a valid geolocation country code.'.format(
				gl=self.geolocation))
            self.params = {'hl': self.language,'gl': self.geolocation}

            self._base_url = s.BASE_URL
            self._suggestion_url = s.SUGGESTION_URL
            self._search_url = s.SEARCH_URL
            self._pagtok = s.PAGE_TOKENS
            self._log = logging.getLogger(__name__)

        def _parse_multiple_apps(self, list_response):

			list_strainer = SoupStrainer('span', {'class': 'preview-overlay-container'})
			soup = BeautifulSoup(list_response.content,
								 'lxml',
								 from_encoding='utf8',
								 parse_only=list_strainer)
			app_ids = [x.attrs['data-docid'] for x in soup.select('span.preview-overlay-container')]
			return multi_futures_app_request(app_ids, params=self.params)

		
        def details(self, app_id):
			url = build_url('details', app_id)

			try:
				response = send_request('GET', url, params=self.params)
				soup = BeautifulSoup(response.content, 'lxml', from_encoding='utf8')
			except requests.exceptions.HTTPError as e:
				raise ValueError('Invalid application ID: {app}. {error}'.format(app=app_id, error=e))

			app_json = parse_app_details(soup)
			app_json.update({'app_id': app_id,'url': url,})
			return app_json

        def collection(self, collection_id, category_id=None,
		 results=None, page=None, age=None, detailed=False):

			if (collection_id not in COLLECTIONS and
					not collection_id.startswith('promotion')):
					raise ValueError('Invalid collection_id \'{collection}\'.'.format(collection=collection_id))
			collection_name = COLLECTIONS.get(collection_id) or collection_id

			category = '' if category_id is None else CATEGORIES.get(category_id)
			if category is None:
				raise ValueError('Invalid category_id \'{category}\'.'.format(
					category=category_id))

			results = s.NUM_RESULTS if results is None else results
			if results > 120:
				raise ValueError('Number of results cannot be more than 120.')

			page = 0 if page is None else page
			if page * results > 500:
				raise ValueError('Start (page * results) cannot be greater than 500.')

			if category.startswith('FAMILY') and age is not None:
				self.params['age'] = AGE_RANGE[age]

			url = build_collection_url(category, collection_name)
			data = generate_post_data(results, page)
			response = send_request('POST', url, data, self.params)

			if detailed:
				apps = self._parse_multiple_apps(response)
			else:
				soup = BeautifulSoup(response.content, 'lxml', from_encoding='utf8')
				apps = [parse_card_info(app_card)
						for app_card in soup.select('div[data-uitype="500"]')]

			return apps

        def developer(self, developer, results=None, page=None, detailed=False):
        
			if not isinstance(developer, basestring) or developer.isdigit():
					raise ValueError('Parameter \'developer\' must be the developer name, not the developer id.')

			results = s.DEV_RESULTS if results is None else results
			page = 0 if page is None else page
			page_num = (results // 20) * page
			if not 0 <= page_num <= 12:
				raise ValueError('Page out of range. (results // 20) * page must be between 0 - 12')
			pagtok = self._pagtok[page_num]

			url = build_url('developer', developer)
			data = generate_post_data(results, 0, pagtok)
			response = send_request('POST', url, data, self.params)

			if detailed:
				apps = self._parse_multiple_apps(response)
			else:
				soup = BeautifulSoup(response.content, 'lxml', from_encoding='utf8')
				apps = [parse_card_info(app)
						for app in soup.select('div[data-uitype="500"]')]

			return apps

        def suggestions(self, query):
			if not query:
				raise ValueError("Cannot get suggestions for an empty query.")				
			self.params.update({
				'json': 1,
				'c': 0,
				'query': query,
			})

			response = send_request('GET',
									self._suggestion_url,
									params=self.params)
			suggestions = [q['s'] for q in response.json()]
			return suggestions

        def search(self, query, page=None, detailed=False):
			page = 0 if page is None else int(page)
			
			if page > len(self._pagtok) - 1:
				raise ValueError('Parameter \'page\' ({page}) must be between 0 and 12.'.format(page=page))

			pagtok = self._pagtok[page]
			data = generate_post_data(0, 0, pagtok)

			self.params.update({'q': quote_plus(query),'c': 'apps',})

			response = send_request('POST', self._search_url, data, self.params)
			soup = BeautifulSoup(response.content, 'lxml', from_encoding='utf8')

			if detailed:
				apps = self._parse_multiple_apps(response)
			else:
				apps = [parse_card_info(app)
					    for app in soup.select('div[data-uitype="500"]')]

			return apps
		
        def similar(self, app_id, detailed=False, **kwargs):
		
			url = build_url('similar', app_id)
			response = send_request('GET',url,params=self.params,allow_redirects=True)
			soup = BeautifulSoup(response.content, 'lxml', from_encoding='utf8')

			if detailed:
				apps = self._parse_multiple_apps(response)
			else:
				apps = [parse_card_info(app)
						for app in soup.select('div[data-uitype="500"]')]

			return apps

        def categories(self, ignore_promotions=True):
        
			categories = {}
			strainer = SoupStrainer('ul', {'class': 'submenu-item-wrapper'})

			response = send_request('GET', s.BASE_URL, params=self.params)
			soup = BeautifulSoup(response.content,
								 'lxml',
								 from_encoding='utf8',
								 parse_only=strainer)
			category_links = soup.select('a.child-submenu-link')
			category_links += soup.select('a.parent-submenu-link')
			age_query = '?age='

			for cat in category_links:
				url = urljoin(s.BASE_URL, cat.attrs['href'])
				category_id = url.split('/')[-1]
				name = cat.string.strip()

				if age_query in category_id:
					category_id = 'FAMILY'
					url = url.split('?')[0]
					name = 'Family'

				if category_id not in categories:
					if ignore_promotions and '/store/apps/category/' not in url:
						continue

					categories[category_id] = {
						'name': name,
						'url': url,
						'category_id': category_id}

			return categories

	def reviews(self, app_id_list, page=0):
			#print("In Scraper - reviews def   \n")
			reviews_adder = []
			for n in range(len(app_id_list)):
				#app_id=app_id_list[n]
				#print(app_id)
				data = {
					'reviewType': 0,
					'pageNum': page,
					'id': app_id_list[n],
					'reviewSortOrder': 4,
					'xhr': 1,
					'hl': self.language
				}
				self.params['authuser'] = '0'
				#print('before send request')	
				#print(app_id_list[n])
				response = send_request('POST', s.REVIEW_URL, data, self.params)				
				content = response.text
				content = content[content.find('[["ecr"'):].strip()
				data = json.loads(content)
				#print(data)
				html = data[0][2]
				soup = BeautifulSoup(html, 'lxml', from_encoding='utf8')
				#print(soup)
				reviews = []
				for element in soup.select('.single-review'):
					#print('Inside single review')	
					review = {}
					#print("In Scraper - reviews def- rev_app_id:: \n")
					#print(app_id)
					review['rev_app_id'] = app_id_list[n]
					avatar_style = element.select_one('.author-image').get('style')
					#print(avatar_style)
					if avatar_style:
						sheet = cssutils.css.CSSStyleSheet()
						sheet.add('tmp { %s }' % avatar_style)
						review['author_image'] = list(cssutils.getUrls(sheet))[0]

					review_header = element.select_one('.review-header')
					review['review_id'] = review_header.get('data-reviewid', '')
					review['review_permalink'] = review_header.select_one('.reviews-permalink').get('href')

					review['author_name'] = review_header.select_one('.author-name').text
					review['review_date'] = review_header.select_one('.review-date').text

					curr_rating = review_header.select_one('.current-rating').get('style')
					review['current_rating'] = int(int(str(cssutils.parseStyle(curr_rating).width).replace('%', '')) / 20)

					body_elem = element.select_one('.review-body')
					review_title = body_elem.select_one('.review-title').extract()
					body_elem.select_one('.review-link').decompose()
					review['review_title'] = review_title.text
					review['review_body'] = body_elem.text

					reviews.append(review)
					reviews_adder.append(review)
					data=''
			return reviews_adder
