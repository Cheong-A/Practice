import requests
from bs4 import BeautifulSoup

#-------------------------------------------------------#

url = 'https://news.v.daum.net/v/20190728165812603'
resp = requests.get(url)

resp.text

url = 'https://news.v.daum.net/v/20190728165812603'
resp = requests.get(url)

soup = BeautifulSoup(resp.text)
title = soup.find('h3', class_='tit_view')
title.get_text()