#Introduction (all  total: 7 - max points: 75)
# Solve Me First
def solveMeFirst(a,b):
	# Hint: Type return a+b below
 return a+b

num1 = int(input())
num2 = int(input())
res = solveMeFirst(num1,num2)
print(res)

#Say "Hello, World!" With Python
Str="Hello, World!"
print(Str)

#Python If-Else

import math
import os
import random
import re
import sys

n = int(input().strip())
res= n % 2

if res==1:
    print('Weird')
elif res==0:
    if 6 <= n <= 20:
     print('Weird')
    else:
        print('Not Weird')

#Arithmetic Operators

a = int(input())
b = int(input())

l1 = a+b
l2 = a-b
l3 = a*b

print(l1)
print(l2)
print(l3)

#Python: Division

a = int(input())
b = int(input())
d1=a//b
d2=a/b
print(d1)
print(d2)

#Loops

n = int(input())
for x in range(0 , n) :
    print(x*x)

#Write a function

def is_leap(year):
    leap = False

    r4 = year % 4
    r100 = year % 100
    r400 = year % 400

    if r4 == 0 and not (r100 == 0):
        leap = True
    elif r400 == 0:
        leap = True

    return leap

#Print Function

n = int(input())
for x in range(1, n+1):
 print (x, end="")

#Data types (all total: 6 - max points: 60)
#List Comprehensions

x = int(input())
y = int(input())
z = int(input())
n = int(input())

Initial=[]
for i in range(x+1):
    for j in range(y+1):
        for k in range(z+1):
            Initial.append([i,j,k])


print( [d for d in Initial if not(d[0]+d[1]+d[2]== n)] )

#Find the Runner-Up Score!


n = int(input())
arr = list(map(int, input().split()))

arr.sort(reverse=True)
for x in arr:
    if x<max(arr):
        print(x)
        break

#Nested Lists

import itertools

records = []
for _ in range(int(input())):
    name = input()
    score = float(input())
    records.append([name, score])

everysort = sorted(records, key=lambda x: x[1])

for u in everysort:
    if not (u[1] == min(everysort, key=lambda x: x[1])[1]):
        slg = u[1]
        break

c = []
for z in everysort:
    if z[1] == slg:
        c.append(z[0])
c.sort()
for d in c:
    print(d)

#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

    x=student_marks[query_name]
    avg= sum(x)/len(x)
    avg1="{:.2f}".format(avg)
    print(avg1)

#Lists

if __name__ == '__main__':
    N = int(input())
    orders=[]
    final=[]
    for x in range(N):
        orders.append(input().split())

    for O in orders:

        if O[0]=='insert':
            final.insert(int(O[1]),int(O[2]))
        elif O[0]=='print':
            print(final)
        elif O[0]=='remove':
            final.remove(int(O[1]))
        elif O[0]=='append':
            final.append(int(O[1]))
        elif O[0]=='sort':
            final.sort()
        elif O[0]=='pop':
            final.pop()
        elif O[0]=='reverse':
            final.reverse()

#Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    ss=tuple(integer_list)
    print(hash(ss))

#Strings (all  total: 14 - max points: 220)
#sWAP cASE

def swap_case(s):
    return s.swapcase()

#String Split and Join

def split_and_join(line):
    line=line.split(" ")
    line="-".join(line)
    return line

#What's Your Name?

def print_full_name(a, b):
    print("Hello",a,b+"!","You just delved into python.")

#Mutations

def mutate_string(string, position, character):
    l=list(string)
    l[position]=character
    string= ''.join(l)
    return string

#Find a string

def count_substring(string, sub_string):
    count = 0

    for i in range(0, len(string)):
        if sub_string == string[i:i + len(sub_string)]:
            count = count + 1
    return count

#String Validators

if __name__ == '__main__':
    s = input()
print(any(o.isalnum() for o in s))
print(any(o.isalpha() for o in s))
print(any(o.isdigit() for o in s))
print(any(o.islower() for o in s))
print(any(o.isupper() for o in s))


#Text Alignment

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#Text Wrap

import textwrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

#Designer Door Mat

# Enter your code here. Read input from STDIN. Print output to STDOUT
n, m = map(int,input().split())
sym='.|.'

texture_top = [(sym*(2*i + 1)).center(m, '-') for i in range(n//2)]
# creating the basic texture
texture_bottom = texture_top[::-1]
# creating the reverse texure by  reverse indexing
print('\n'.join(texture_top + ['WELCOME'.center(m, '-')] + texture_bottom))
#adding to the next line,the welcome line to the top texture and bottom texture

#String Formatting

def print_formatted(number):
    # your code goes here
    s = []

    for i in range(1, number + 1):
        o = oct(i).lstrip("0o").rstrip("L")
        b = bin(i).lstrip("0b").rstrip("L")
        h = hex(i).lstrip("0x").rstrip("L").upper()
        s.append([str(i), o, h, b])
    for i in s:
        g = len("{0:b}".format(number))
        print(i[0].rjust(g, ' '), i[1].rjust(g, ' '), i[2].rjust(g, ' '), i[3].rjust(g, ' '))

#Alphabet Rangoli

def print_rangoli(n):
    # your code goes here
    l1=list(map(chr,range(97,123)))
    x=l1[n-1::-1]+l1[1:n]
    m=len('-'.join(x))
    for i in range(1,n):
        print('-'.join(l1[n-1:n-i:-1]+l1[n-i:n]).center(m,'-'))
    for i in range(n,0,-1):
        print('-'.join(l1[n-1:n-i:-1]+l1[n-i:n]).center(m,'-'))

#Capitalize

import math
import os
import random
import re
import sys
import string

def solve(s):

    return string.capwords(s,' ')

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()

#The Minion Game

def minion_game(string):
    # your code goes here
    points_s=0
    points_k=0
    vowels='AEIOU'
    for i in range(len(string)):
        if string[i] not in vowels:
            points_s=points_s+(len(string)-i)
        else:
            points_k=points_k+(len(string)-i)
    if points_s>points_k:
        print("Stuart",points_s)
    elif points_k>points_s:
        print("Kevin",points_k)
    else:
        print("Draw")
if __name__ == '__main__':
    s = input()
    minion_game(s)

#Merge the Tools!

def merge_the_tools(string, k):
    # your code goes here
    l=[]
    m=0
    for i in range(len(string)//k):
        l.append(string[m:m+k])
        m=m+k
    for v in l:
        print(''.join(list(dict.fromkeys(list(v)).keys())))
if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

#Sets (all total: 13 - max points: 170)
#Introduction to Sets

def average(array):
    # your code goes here
 dis=list(set(array))
 return sum(dis)/len(dis)
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

#No Idea

[m,n]=list(map(int,input().split(' ')))
arr=list(map(int,input().split(' ')))
a=set(map(int,input().split(' ')))
b=set(map(int,input().split(' ')))

happy=0

for i in arr:
    if i in a:
        happy=happy+1
    elif i in b:
        happy=happy-1

print(happy)

#Symmetric Difference

# Enter your code here. Read input from STDIN. Print output to STDOUT
m=int(input())
a=set(list(map(int, input().split())))
n=int(input())
b=set(list(map(int, input().split())))
final=a.union(b).difference(a.intersection(b))
f=list(final)
f.sort()
for i in f:
 print(i)

#Set .add()

# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
cntry=set()
for i in range(0,n):
   cntry.add(input())

print(len(cntry))

#Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
N=int(input())

for i in range(0,N):
    cmmnd=input().split()
    if cmmnd[0]=='remove':
        s.remove(int(cmmnd[1]))
    elif cmmnd[0]=='discard':
        s.discard(int(cmmnd[1]))
    elif cmmnd[0]=='pop':
        s.pop()

print(sum(list(s)))

#Set .union() Operation

n=int(input())
eng=set(map(int,input().split()))
b=int(input())
frn=set(map(int,input().split()))
print(len(eng|frn))

#Set .intersection() Operation

n=int(input())
eng=set(map(int,input().split()))
b=int(input())
frn=set(map(int,input().split()))
print(len(eng&frn))

#Set .difference() Operation

n=int(input())
eng=set(map(int,input().split()))
b=int(input())
frn=set(map(int,input().split()))
print(len(eng-frn))

#Set .symmetric_difference() Operation

n=int(input())
eng=set(map(int,input().split()))
b=int(input())
frn=set(map(int,input().split()))
print(len(eng^frn))

#Set Mutations

n = input()
a = set(map(int, input().split()))
N = int(input())

for i in range(0, N):
    cmnd = list(input().split())
    b = set(map(int, input().split()))

    if cmnd[0] == 'update':
        a.update(b)

    elif cmnd[0] == 'intersection_update':
        a.intersection_update(b)

    elif cmnd[0] == 'difference_update':
        a.difference_update(b)

    elif cmnd[0] == 'symmetric_difference_update':
        a.symmetric_difference_update(b)

print(sum(a))

#The Captain's Room

k=int(input())
arr=list(map(int,input().split()))
arr.sort()
cap=list(set(arr[0::2])^set(arr[1::2]))
print(cap[0])

#Check Strict Superset

# Enter your code here. Read input from STDIN. Print output to STDOUT
a = set(map(int, input().split()))
n = int(input())
c = 0

for i in range(0, n):
    n1 = set(map(int, input().split()))

    if (n1 & a == n1) and (len(a - n1) >= 1):
        c = c + 1
    else:
        print('False')
        break

if c == n:
    print('True')

#Collections (all total: 8 - max points: 220)
#collections.Counter()

from collections import Counter

x = int(input())
size = Counter(list(map(int, input().split())))
N = int(input())
cash = 0

for i in range(N):

    [s, p] = list(map(int, input().split()))
    if s in size.keys():

        cash = cash + p
        size[s] = size[s] - 1

        if size[s] == 0:
            del size[s]

print(cash)

#DefaultDict Tutorial

from collections import defaultdict
A = defaultdict(list)
B=[]
[n,m]=list(map(int,input().split()))

for i in range(1,n+1):
    A[input()].append(i)

for i in range(m):
    B.append(input())



for i in B:
    if i in A:
        print(" ".join(map(str,(A[i]))))
    else :
        print (-1)

#Collections.namedtuple()

from collections import namedtuple
N=int(input())
Marks=0
column=input().split()

for i in range(N):
    C=namedtuple('C',column)
    c1, c2, c3,c4 = input().split()
    C = C(c1,c2,c3,c4)
    Marks +=int(C.MARKS)
print('{:.2f}'.format(Marks/N))

#Collections.OrderedDict()

from collections import OrderedDict

N = int(input())
item = OrderedDict()
for i in range(N):
    name, space, price = input().rpartition(' ')
    item[name] = item.get(name, 0) + int(price)
for name, price in item.items():
    print(name, price)

#Word Order

from collections import Counter
n=int(input())
l1=list()
for i in range(n):
    l1.append(input())
x=Counter(l1)
print(len(x))
print(*x.values())

#Collections.deque()

from collections import deque
d=deque()
N=int(input())
for i in range(N):
  order=list(input().split())
  if order[0]=='append':
    d.append(int(order[1]))
  elif order[0]=='appendleft':
    d.appendleft(int(order[1]))
  elif order[0]=='pop':
    d.pop()
  elif order[0]=='popleft':
    d.popleft()
print(*d)

#Company Logo

from collections import Counter
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    s = input()
    s_so=Counter(sorted(s)).most_common(3)
    for i,j in s_so:
        print(i,j)

#Piling Up!

from collections import deque
t= int(input())
while t>0:
    n=int(input())
    cube=list(map(int,input().split()))
    lst=deque(cube)
    r=lst.pop()
    l=lst.popleft()

    if l>r:
        csv= l
    else:
        csv=r

    flag=False
    while(len(lst)>0):
        if(l>=r and l<=csv):
            csv=l
            l=lst.popleft()
            latest=l
        elif(r>l and r<=csv):
            csv=r
            r=lst.pop()
            latest=r
        else:
            flag= True
            break
    if flag or latest > csv:
        print("No")
    else:
        print("Yes")
    t-=1

#Date and Time (all total: 2 - max points: 40)
#Calendar Module

import calendar
[m,d,y]=list(map(int,input().split()))
print(list(calendar.day_name)[calendar.weekday(y, m, d)].upper())

#Time Delta

import math
import os
import random
import re
import sys
import datetime as dt
# Complete the time_delta function below.
def time_delta(t1, t2):
 a=dt.datetime.strptime(t1, "%a %d %b %Y %H:%M:%S %z")
 b=dt.datetime.strptime(t2, "%a %d %b %Y %H:%M:%S %z")
 return (str(int(abs(a-b).total_seconds())))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

#Exceptions (only 1 - max points: 10)
#Exceptions

T=int(input())

for _ in range(T):
    try:
        [a,b]=list(map(int,input().split()))
        print(a//b)
    except ZeroDivisionError as e:
        print("Error Code:",e)
    except ValueError as c:
        print("Error Code:",c)

#Built-ins (only 3 - max points: 80)
#Zipped!

# Enter your code here. Read input from STDIN. Print output to STDOUT
[n,x]=list(map(int,input().split()))
table=[]
for i in range(x):
    table.append(list(map(float,input().split())))
for i in zip(*table):
    print(sum(i)/len(i))

#Athlete Sort

from operator import itemgetter
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    final = sorted(arr, key=itemgetter(k))
    for i in final:
        print(*i)

#ginortS

# Enter your code here. Read input from STDIN. Print output to STDOUT
s=sorted(input())
u=""
l=""
o=""
e=""
for i in s:
    if i.isupper():
        u +=i
    elif i.islower():
        l +=i
    elif int(i)%2!=0:
        o +=i
    elif int(i)%2==0:
        e +=i
print(l +u +o +e)

#Python Functionals (only 1 - max points: 20)
#Map and Lambda Function

cube = lambda x: x^3

def fibonacci(n):
    # return a list of fibonacci numbers
 fib=[0,1]
 for i in range(n-2):
     fib.append(fib[i]+fib[i+1])
 return fib
if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

#Regex and Parsing challenges (all total: 17 - max points: 560)

#Detect Floating Point Number

import re

T = int(input())
for i in range(T):
    N = str(input())
    print(bool(re.match(r'^[-+.]?[0-9]*\.[0-9]+$', N)))

# ^ search in the start of string
# [-+.]? it can be each of +-. in the start
# [0-9] possible character (numbers:0 to 9) following
# * following character may be repeated as many as it like
# \. is placeholder for any character("." in the middle)
# [0-9] possible character (numbers:0 to 9) following
# + following character may be repeated as many as it like (one time is required)
# $ Matches the end of the string

#Re.split()

regex_pattern = r"[^0-9]+"
#[^0-9] it will match anything other than character marked as digits in the Unicode character properties database.
#match 1 or more repetitions

import re
print("\n".join(re.split(regex_pattern, input())))

#Group(), Groups() & Groupdict()

import re
m = re.search(r'([a-zA-Z0-9])\1', input().strip())
print(m.group(1) if m else -1)
#[a-zA-Z0-9] alphanumeric characters search
# \1 Matches the contents of the group of the same number

#Re.findall() & Re.finditer()

import re
s=input()
vowels=['aeiou']
consonant=['qwrtypsdfghjklzxcvbnm']
f=re.findall(r'{consonant}({vowels}{{2,}})(?={consonant})'.format(vowels=vowels,consonant=consonant),s,re.IGNORECASE)
if f:
    print(*f, sep='\n')
else:
    print (-1)

#{{2,}} Causes the resulting RE to match from 2 to or more  repetitions of the {vowels}
#(?={consonant}) Matches if {consonant} matches next
#re.IGNORECASE provides case-insensitive matching

#Re.start() & Re.end()

import re
s=input()
k=input()
matches = list(re.finditer(r'(?={})'.format(k), s))
#creats matches of substring k in s
if matches:
    print('\n'.join(str((match.start(),
          match.start() + len(k) - 1)) for match in matches))
#indexing end without using re.end() and just by ading length of k to start index
else:
    print('(-1, -1)')

#Regex Substitution

import re

n = int(input())
for i in range(n):
    s = input()
    print(re.sub(r"(?<= )(&&|\|\|)(?= )",
                 lambda x: 'and' if x.group() == '&&' else 'or', s))
# Matches if the current position in the string is preceded by a match for " " that ends at the current position(space character)
# Matches if " "  matches next(space character)

#Validating Roman Numerals

thousands = 'M{0,3}'  # matches anything like M, MM, MMM
hundreds = '(C[MD]|D?C{0,3})'  # matches anything like C,CC, CCC, CD,D, DC and CM(900)
tens = '(XC|XL|L?X{0,3})'  # matches anything like X,XX,XXX,XL,... up to XC(90)
digits = '(IX|IV|V?I{0,3})'  # the roman numeral digits

regex_pattern = r"%s%s%s%s$" % (thousands, hundreds, tens, digits)

import re

print(str(bool(re.match(regex_pattern, input()))))

#Validating phone numbers

import re
n=int(input())
starter="^[789]"
for i in range(n):
    num=input()
    if len(num)==10 and num.isdecimal():
        final=re.findall(starter,num)
        if bool(final):
            print("YES")
        else:
            print("NO")
    else:
            print("NO")

#Validating and Parsing Email Addresses

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n=int(input())
validity="<[a-z][a-zA-Z0-9\-\_\.]+@[a-zA-Z]+\.[a-zA-Z]{1,3}>"
for i in range(n):
    name,mail=input().split(" ")
    v=re.match(validity,mail)
    if bool(v):
     print(name,mail)
# {1,3} check th 1 to 3 extention strick
# others is just obvious AF

#Hex Color Code

import re
N=int(input())
started=False
codecolor="#[0-9a-fA-f]{3,6}"

for i in range(N):
    l=input()
    if '{' in l:
        started= True
    elif '}' in l:
        started= False
    elif bool(started):
     final=re.findall(codecolor,l)
     for i in final:
        print(i)

#HTML Parser - Part 1

# solution of this exersice is based on the answer on GitHub
from html.parser import HTMLParser


class Html(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print('Start :', tag)
        for attr in attrs:
            print('->', ' > '.join(map(str, attr)))

    def handle_endtag(self, tag):
        print('End   :', tag)

    def handle_startendtag(self, tag, attrs):
        print('Empty :', tag)
        for attr in attrs:
            print("->", " > ".join(map(str, attr)))


line = ""
N = int(input())
for i in range(N):
    line += input()

parser = Html()
parser.feed(line)
parser.close()

#HTML Parser - Part 2

# hints to the solution by tutorial
from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if data != "\n":
            if "\n" not in data:
                print(">>> Single-line Comment")
                print(data)
            else:
                print(">>> Multi-line Comment")
                print(data)

    def handle_data(self, data):
        if data != "\n":
            print(">>> Data")
            print(data)


html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'

parser = MyHTMLParser()
parser.feed(html)
parser.close()

#Detect HTML Tags, Attributes and Attribute Values

# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser

class H(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for att in attrs:
            print("-> {} > {}".format(att[0],att[1]))
html=""
N=int(input())
for i in range(N):
    html +=input()
    html +="\n"
parser=H()
parser.feed(html)
parser.close()

#Validating UID

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
T=int(input())

for i in range(T):
    uid=input().strip()
    ucheck=re.search(r'(.*[A-Z]){2,}',uid)
    dcheck=re.search(r'(.*[0-9]){3,}',uid)

    if uid.isalnum() and len(uid)==10:
        if bool(ucheck)and bool(dcheck):
            if re.search(r'.*(.).*\1+.*',uid):
                print ("Invalid")
            else:
                print ("Valid")

        else:
           print("Invalid")
    else:
        print("Invalid")

#XML (all total: 2 - max points: 40)
#XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
 s=len(node.attrib) + sum(get_attr_number(c) for c in node)
 return (s)


if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

#XML2 - Find the Maximum Depth

import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    if level >= maxdepth:
        maxdepth = level
    for child in elem:
        depth(child, level)
if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)

#Closures and Decorations (all total: 2 - max points: 60)
#Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        f(["+91 "+c[-10:-5]+" "+c[-5:] for c in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)

#Decorators 2 - Name Directory

import operator

def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')


#Numpy (all total: 15 - max points: 300)
#Arrays

import numpy

def arrays(arr):

 f=numpy.array(list(arr),float)
 return f[::-1]

arr = input().strip().split(' ')

#Shape and Reshape

import numpy
f=numpy.array(input().split(),int)
print(numpy.reshape(f,(3,3)))

#Transpose and Flatten

import numpy
[n,m]=list(map(int, input().split()))
f=numpy.array([input().split() for i in range(n)],int)
print(numpy.transpose(f))
print(f.flatten())

#Concatenate

import numpy
[n,m,p]=list(map(int,input().split()))
f=numpy.array([input().split() for i in range(n)],int)
s=numpy.array([input().split() for i in range(m)],int)
print (numpy.concatenate((f, s), axis = 0))

#Zeros and Ones

import numpy
nums = tuple(map(int, input().split()))
print (numpy.zeros(nums, dtype = numpy.int))
print (numpy.ones(nums, dtype = numpy.int))

#Eye and Identity

import numpy
[n,m]=list(map(int, input().split()))
print (numpy.eye(n, m).replace('1',' 1').replace('0',' 0'))

#Array Mathematics

import numpy as np
n, m = map(int, input().split())
a= (np.array([input().split() for _ in range(n)], dtype=int))
b= (np.array([input().split() for _ in range(n)], dtype=int))
print(a+b, a-b, a*b, a//b, a%b, a**b, sep='\n')

#Floor, Ceil and Rint

import numpy as np
arr= np.array(input().split(), float)
np.set_printoptions(sign=' ')
print(np.floor(arr))
print(np.ceil(arr))
print(np.rint(arr))

#Sum and Prod

import numpy as np
[n,m]=list(map(int,input().split()))
A=np.array([input().split() for _ in range(n)],int)
print(np.prod(np.sum(A, axis=0), axis=0))

#Min and Max

import numpy
import numpy as np
[n,m]=list(map(int,input().split()))
A=np.array([input().split() for _ in range(n)],int)
print(np.max(np.min(A, axis=1), axis=0))

#Mean, Var, and Std

import numpy
import numpy as np
np.set_printoptions(sign=' ')
[n,m]=list(map(int,input().split()))
A=np.array([input().split() for _ in range(n)],int)
print(np.mean(A, axis=1))
print(np.var(A, axis=0))
print(numpy.around(np.std(A),decimals=12))

#Dot and Cross

import numpy as np
n=int(input())
A=np.array([input().split() for _ in range(n)],int)
B=np.array([input().split() for _ in range(n)],int)
print(np.dot(A, B))

#Inner and Outer

import numpy as np

A=np.array(input().split(),int)
B=np.array(input().split(),int)
print (np.inner(A, B))
print (np.outer(A, B))

#Polynomials

import numpy as np
p=np.array(input().split(),float)
x=int(input())
print(np.polyval(p,x))

#Linear Algebra

import numpy as np
n=int(input())
A=np.array([input().split() for _ in range(n)],float)
print(round(np.linalg.det(A),2))

#Problem 2
#Birthday Cake Candles

import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    mx=0
    counts=0
    for i in range(len(candles)):
        if candles[i]>mx:
            mx=candles[i]
            counts=1
        elif candles[i]==mx:
            counts +=1
    return(counts)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#Number Line Jumps

import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    if (v1 - v2) > 0:
        if (x1 - x2) % (v1 - v2) == 0:
            return ("YES")
        else:
            return ("NO")
    else:
        return ("NO")


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#Viral Advertising

import math
import os
import random
import re
import sys

def viralAdvertising(n):
    C=0
    pop=5
    for i in range(n):
        like=pop//2
        C +=like
        pop=like*3
    return(C)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#Recursive Digit Sum

# !/bin/python3

import math
import os
import random
import re
import sys

def superDigit(n, k):
    while int(n) >= 10:
        s = sum(int(i) for i in str(n))
        n = s
    sd = n * k
    # the superdigit of k repitation of number n will be superdigit of k times of superdigit of n
    while sd >= 10:
        s = sum(int(i) for i in str(sd))
        sd = s
    return (sd)


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#Insertion Sort - Part 1

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort1(n, arr):
    check=arr[-1]
    for i in range(n-2,-1,-1):
        if arr[i]>check:
            arr[i+1]=arr[i]
            print(*arr)
        else:
            arr[i+1]=check
            print(*arr)
            return
    arr[0]=check
    print(*arr)
    return
if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#Insertion Sort - Part 2

import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
 for i in range(n):
    if i==0:
        continue
    for j in range(0,i):
        if arr[j]>arr[i]:
            s=arr[i]
            arr[i]=arr[j]
            arr[j]=s
        else:
            continue
    print(*arr)


if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)

#FINALLYYYYYYYY THE ENDDDDDDDDD :((