# List 를 2개 사용할 떄, 아래 3개의 결과값은 동일
coffees = ['아메리카노', '카페라떼', '카페모카', '바닐라라떼', '핸드드립', '콜드브루']
prices = [4100, 4600, 4600, 5100, 6000, 5000]

for i in range(len(prices)): #0,1,2,3,4,5
    if prices[i] <= 5000:
        print(coffees[i])
print("------------------------------")
# 1. enumerate
for i, price in enumerate(prices):
    if price <= 5000:
        print(coffees[i])

print("------------------------------")
# 2. zip
for coffee, price in zip(coffees, prices):
    if price <= 5000:
        print(coffee)