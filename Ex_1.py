
#전설의 구구단 만들기
#21.07.21_20:00

#2단을 while문으로
number = 1
while number<10:
    print("2 x %d = %d" % (number,2*number))
    number+=1
print('------------------------------')
#for 문으로 구구단 !
for a in range(1,9):
    a+=1
    print('------------------------------') #보기 좋게 추가
    for i in range(9):
        i += 1
        print(a,"x",i,"=",a*i)

# List 를 2개 사용할 떄, 아래 3개의 결과값은 동일
coffees = ['아메리카노', '카페라떼', '카페모카', '바닐라라떼', '핸드드립', '콜드브루']
prices = [4100, 4600, 4600, 5100, 6000, 5000]

for i in range(len(prices)): #0,1,2,3,4,5
    if prices[i] <= 5000:
        print(coffees[i])
print("-------------------")
# 1. enumerate
for i, price in enumerate(prices):
    if price <= 5000:
        print(coffees[i])

print("-------------------")
# 2. zip
for coffee, price in zip(coffees, prices):
    if price <= 5000:
        print(coffee)