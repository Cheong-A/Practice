#네이버 영화 랭킹 크롤링
import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'}


def crawler(sort1, day):
    #정렬기준 숫자로 받아 문자로 변환해주기
    if sort1 in '1':
        sort1 = 'cur'
    elif sort1 in '2':
        sort1 = 'pnt'
    elif sort1 in '0':
        sort1 = 'cnt'

    url = 'https://movie.naver.com/movie/sdb/rank/rmovie.naver?sel=' + sort1 + '&date=' + day
    print(url)

    data = requests.get(url, headers=headers)
    soup = BeautifulSoup(data.text, 'html.parser')
    movies = soup.select("#old_content > table > tbody > tr")

    #영화 제목, 평점 크롤링, None값일때 제외 조건 추가(오류방지)
    for movie in movies:
        if sort1=='cur' or sort1=='pnt':
            movie_name = movie.select_one("td.title > div > a")
            movie_point = movie.select_one("td.point")
            if movie_name is not None:
                ranking = movie.select_one("td:nth-child(1) > img")["alt"]
                print(ranking, movie_name.text, movie_point.text)
        elif sort1=='cnt':
            movie_name = movie.select_one("td.title > div > a")
            try : #try
                if movie_name is not None:
                    ranking = movie.select_one("td:nth-child(1) > img")["alt"] #에러난곳
                    print(ranking, movie_name.text)
            except TypeError:
                pass
# 메인함수
def main():
    info_main = input("=" * 50 + "\n" + "입력 형식에 맞게 입력해주세요." + "\n" + " 시작하시려면 Enter를 눌러주세요." + "\n" + "=" * 50)
    sort1 = input("영화 랭킹 기준(조회순=0 현재상영작(평점)=1  모든영화(평점=2): ")
    day = input("기준날짜 입력(20190104):")

    crawler(sort1, day)


# 메인함수 수행
main()
