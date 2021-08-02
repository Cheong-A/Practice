#네이버 영화 랭킹 크롤링
import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'}


def crawler(sort1, day):
        if sort1 in '1':
            sort1='cur'
        else:
            sort1 ='pnt'
        url = 'https://movie.naver.com/movie/sdb/rank/rmovie.naver?sel=' + sort1 + '&date=' + day
        print(url)

        data = requests.get(url, headers=headers)
        soup = BeautifulSoup(data.text, 'html.parser')
        movies = soup.select("#old_content > table > tbody > tr")

        for movie in movies:
            movie_name = movie.select_one("td.title > div > a")
            movie_point = movie.select_one("td.point")
            if movie_name is not None:
                ranking = movie.select_one("td:nth-child(1) > img")["alt"]
                print(ranking, movie_name.text, movie_point.text)


# 메인함수
def main():
    info_main = input("=" * 50 + "\n" + "입력 형식에 맞게 입력해주세요." + "\n" + " 시작하시려면 Enter를 눌러주세요." + "\n" + "=" * 50)
    sort1 = input("영화 랭킹 평점순 기준 입력(현재상영작=1  모든영화=2): ")
    day = input("기준날짜 입력(20190104):")

    crawler(sort1, day)


# 메인함수 수행
main()
