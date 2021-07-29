
#공공데이터 포털에서 데이터 가져오기

#내인증키
serviceKey = 'zBh%2BPi6KDaUex%2F8cL7S4iMpU8Xdktf%2FuZzI4BOrR0DgphDwStGoQ4nW4Oep%2BBvD2V6%2Fn6Y3z43EIneLuuFi1nQ%3D%3D'

#Endpoint 확인
endpoint = 'http://api.visitkorea.or.kr/openapi/service/rest/EngService/areaCode?serviceKey={}&numOfRows=10&pageSize=10&pageNo=1&MobileOS=ETC&MobileApp=AppTest&_type=json'.format(serviceKey)
print(endpoint)

import requests

#Parameter 확인
endpoint = 'http://api.visitkorea.or.kr/openapi/service/rest/EngService/areaCode?serviceKey={}&numOfRows=10&pageSize=10&pageNo={}&MobileOS=ETC&MobileApp=AppTest&_type=json'.format(serviceKey, 1)
resp = requests.get(endpoint)

print(resp.status_code)
print(resp.text)
# requests 모듈을 활용하여 API 호출
# response 확인하여 원하는 정보 추출
data = resp.json()
print(data['response']['body']['items']['item'][0])