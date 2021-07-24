
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

#강사님이 알려준 방법
for a in range(2,10):
    print('------------------------------') #보기 좋게 추가
    for i in range(1,10):
        print(a,"x",i,"=",a*i)


#별을 찍어 봅시다!!
num = int(input("하늘에 별이 얼마나 ?"))

for i in range(num):
#    for j in range(num - i):
#        print("*",end='')
#    print()
    print("*" * (num - i))


#단어 갯수 구하기
#아래 5개의 변수들은 각각 하나의 문서를 의미합니다.
news1 = "earn	champion products ch approves stock split champion products inc said its board of directors approved a two for one stock split of its common shares for shareholders of record as of april the company also said its board voted to recommend to shareholders at the annual meeting april an increase in the authorized capital stock from five mln to mln shares reuter"
news2 = "acq	computer terminal systems cpml completes sale computer terminal systems inc said it has completed the sale of shares of its common stock and warrants to acquire an additional one mln shares to sedio n v of lugano switzerland for dlrs the company said the warrants are exercisable for five years at a purchase price of dlrs per share computer terminal said sedio also has the right to buy additional shares and increase its total holdings up to pct of the computer terminal s outstanding common stock under certain circumstances involving change of control at the company the company said if the conditions occur the warrants would be exercisable at a price equal to pct of its common stock s market price at the time not to exceed dlrs per share computer terminal also said it sold the technolgy rights to its dot matrix impact technology including any future improvements to woodco inc of houston tex for dlrs but it said it would continue to be the exclusive worldwide licensee of the technology for woodco the company said the moves were part of its reorganization plan and would help pay current operation costs and ensure product delivery computer terminal makes computer generated labels forms tags and ticket printers and terminals reuter"
news3 = "earn	cobanco inc cbco year net shr cts vs dlrs net vs assets mln vs mln deposits mln vs mln loans mln vs mln note th qtr not available year includes extraordinary gain from tax carry forward of dlrs or five cts per shr reuter"
news4 = "earn	am international inc am nd qtr jan oper shr loss two cts vs profit seven cts oper shr profit vs profit revs mln vs mln avg shrs mln vs mln six mths oper shr profit nil vs profit cts oper net profit vs profit revs mln vs mln avg shrs mln vs mln note per shr calculated after payment of preferred dividends results exclude credits of or four cts and or nine cts for qtr and six mths vs or six cts and or cts for prior periods from operating loss carryforwards reuter"
news5 = "earn	brown forman inc bfd th qtr net shr one dlr vs cts net mln vs mln revs mln vs mln nine mths shr dlrs vs dlrs net mln vs mln revs billion vs mln reuter"

#news_list는 뉴스 5개가 원소인 리스트입니다.
#word_Dict는 문서 하나당 단어와 단어 갯수가 저장되는 사전입니다. 하나만 사용하셔도 되고, 여러개를 사용하셔도 됩니다.

D_news1={}
for word in news1.split():
    D_news1[word] = D_news1.get(word,0) +1 #(넣은 단어가 있으면 출력, 없으면 0)+1
print(D_news1)
print('-------------------------------------------')

D_news2={}
for word in news2.split():
    D_news2[word] = D_news2.get(word,0) +1 #(넣은 단어가 있으면 출력, 없으면 0)+1
print(D_news2)

print('-------------------------------------------')
#100 이하의 자연수 중에서 5의 배수를 모두 찾아서 출력
result=[]
for i in range(1,101):
    if i % 5 == 0:
        result.append(i)
print(result)