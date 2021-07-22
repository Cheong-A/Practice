#21.07.22
#함수를 창조해 보자 !!

#한글 함수명도 가능
def 더하기 (a,b):
    return print(a+b)

a = int(input("정수입력:"))
b = int(input("정수입력:"))
더하기(a,b)

#lambda로 바꾸면?
add= lambda a, b: a+b

print(add(5,6))


#길이로 순서 정리!
strings = ['yoon', 'kim', 'jessica', 'jeong']
strings.sort(key=lambda s:len(s))
print(strings)