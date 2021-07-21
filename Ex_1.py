
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

#-----------------------------------------------
for a in range(2,10):
    print('------------------------------') #보기 좋게 추가
    for i in range(1,10):
        print(a,"x",i,"=",a*i)