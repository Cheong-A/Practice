
##어떤걸 하기위해서 데이터를 가져올 것인지!
##무슨 데이터를 가지고 올지
##웹페이지 구조가 바뀌면 다시 짜야함
##크롤링을 할때는 Chrome

# 위의 일별시세를 가져오기 위해 페이지에 접속합니다.
index_cd = 'KPI200'
page_n = 1
naver_index = f'https://finance.naver.com/sise/sise_index_day.nhn?code=KPI200&page={page_n}'
# 페이지 주소를 규칙화 합니다. 첫번째 가장 중요한 포인트입니다.
naver_index

# 해당 페이지의 page source를 직접 가져옵니다. 웹페이지에서도 우클릭 "페이지 소스 보기"로 같은 HTML 소스를 볼 수 있습니다.
from urllib.request import urlopen
# urlopen 함수를 이용해서 source를 봅니다.
source = urlopen(naver_index).read()
source

# beautifulsoup4를 불러옵니다.
#from bs4 import BeautifulSoup
import bs4
# BeautifulSoup 함수를 사용해서 불러온 html source를 "lxml" parser로 parsing 합니다.
source = bs4.BeautifulSoup(source, 'lxml')
source

# bs4의 prettify() 함수는 HTML source를 tab을 기준으로 "이쁘게" 보여줍니다.
print(source.prettify())

# find_all()는 HTML source에서 조건을 만족하는 모든 tag을 가져오는 함수입니다.
td = source.find_all("td") #td tag를 다가져 오세요
len(td)

# date에 대한 xpath
#/html/body/div/table[1]/tbody/tr[3]/td[1]
source.find_all('table')[0].find_all('tr')[2].find_all('td')[0]
date = source.find_all('td', class_='date')[0].text

import datetime as dt

# 가져온 datetime을 YYYY.MM.DD 형태로 변환합니다.
temp = date.split('.')
yyyy = int(temp[0])
mm = int(temp[1])
dd = int(temp[2])

#yyyy, mm, dd = [int(x) for x in date.split(".")]

this_date= dt.date(yyyy, mm, dd)
this_date

# 날짜 변경 하는 기능은 계속해서 사용할 것이기 때문에, 함수로 정의합니다. 나중에 어떻게 코드상에서 사용되는지 눈여겨 보시면 좋습니다.

def date_format(date):
    yyyy, mm, dd = [int(x) for x in date.split(".")]
    return dt.date(yyyy, mm, dd)

p = source.find_all('td', class_='number_1')[0].text
p

dates = source.find_all('td', class_='date')
prices = source.find_all('td', class_='number_1')
len(dates)
len(prices)

# dates 데이터를 홣용하여 number_1에서 종가를 추출해봅니다.
for i in range(len(dates)):
    this_date = dates[i].text
    this_date = date_format(this_date)  # dt.date로 타입변환

    this_close = prices[i * 4].text  # 0, 4, 8, 12,
    this_close = float(this_close)

    print(this_date, this_close)
#----------------------------------------------------------------------------#

# 100 페이지에 접속하는 예시
page_n = 100

# 위에서 작성했던 모든 코드를 종합합니다.
naver_index = f'http://finance.naver.com/sise/sise_index_day.nhn?code=KPI200&page={page_n}'
source = urlopen(naver_index)
source = bs4.BeautifulSoup(source, 'lxml')
dates = source.find_all('td', class_='date')
prices = source.find_all('td', class_='number_1')

# dates 데이터를 홣용하여 number_1에서 종가를 추출해봅니다.
for i in range(len(dates)):
    this_date = dates[i].text
    this_date = date_format(this_date)  # dt.date로 타입변환

    this_close = prices[i * 4].text  # 0, 4, 8, 12,
    this_close = float(this_close)

    print(this_date, this_close)


#******************************************************************#
# 위에서 구현한 모든 내용을 하나의 함수로 구현합니다.
import pandas as pd


# 다시 한번 함수를 어떻게 구성해야 하는지 생각해봅시다.
def crawl_naver_index(end_page):
    date_list = []
    prices_list = []

    for page_n in range(1, end_page + 1):

        naver_index = f'http://finance.naver.com/sise/sise_index_day.nhn?code=KPI200&page={page_n}'
        source = urlopen(naver_index)
        source = bs4.BeautifulSoup(source, 'lxml')

        dates = source.find_all('td', class_='date')
        prices = source.find_all('td', class_='number_1')

        for i in range(len(dates)):
            this_date = dates[i].text
            this_date = date_format(this_date)  # dt.date로 타입변환

            this_close = prices[i * 4].text  # 0, 4, 8, 12,
            this_close = float(this_close)

            date_list.append(this_date)
            prices_list.append(this_close)

    df = pd.DataFrame({'날짜': date_list, '체결가': prices_list})
    return df


df = crawl_naver_index(20)
df

