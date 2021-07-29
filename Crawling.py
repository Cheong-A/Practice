#requests 모듈 사용하여 http request/response 확인

import requests

#get 요청
url = 'https://news.v.daum.net/v/20190728165812603'
resp = requests.get(url)
print(resp) #응답이 200이면 문제 없음

#print(resp.text)

#post 요청
#post 요청
url1='https://www.saramin.co.kr/zf_user/auth/login'
resp1 = requests.get(url1)
#print(resp1)

data = {
    'id' : '자기아이디',
    'password': '자기비번'
}
requests.post(url1, data=data)
#print(resp1.text)

#Header 사용하기
headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'
}


resp = requests.get(url, headers=headers)
#print(resp.text)

#HTTP response 처리
resp = requests.get(url)
if resp.status_code == 200:
    resp.headers
else:
    print('error')