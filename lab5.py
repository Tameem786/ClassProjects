# a = 123
# b = 123.456
# c = "Hello, Tameem!"
# d = True
# e = False
# g = 1.2e5
# h = 1.2e-5
#
# print(a)
# print(type(a))
# print(d+e)

# a = 3; b = 4
# c = a + b
# print(c)
#
# d = a < b
# print(d)
#
# e = 3 < 4 and 4 < 5
# print(e)
#
# a = "Hello" + "Tameem"
# print(a)
# b = [1,2,3] + [4,5,6]
# print(b)

# a = [1,2,3,4,5]
# b = a[2]
# print(b)
# a.append(6)
# print(a)
# a[2] = 7
# print(a)
#
# a = (1,2,3)
# b = a[2]
# print(b)
# a1,a2,a3 = a
# print(a1,a2,a3)

# a = {"Apple": 3, "Pineapple": 4}
# print(a["Apple"])
# a["Pineapple"] = 7
# print(a["Pineapple"])
#
# a["Melon"] = 3
# print(a)

# a = 7
# if a < 12:
#     print("Good morning!")
# elif a < 17:
#     print("Good afternoon!")
# elif a < 21:
#     print("Good evening!")
# else:
#     print("Good night!")

# for i in [4, 7, 10]:
#     print(i)
# for i in range(3):
#     print(i)
# a = 0
# while a < 3:
#     print(a)
#     a += 1

# a = [1,2,3,4,5,6,7]
# b = [c*2 for c in a]
# print(b)
# b = [c*2 for c in a if c < 5]
# print(b)

# def add1(a, b):
#     c = a + b
#     return c
#
# def add2(a, b=4):
#     c = a + b
#     return c
#
# def add3(a, b, c):
#     d = a + b + c
#     return d
#
# print(add1(3,4))
# print(add2(3))
# t = (1,2,3)
# print(add3(*t))

class Parent:
    def __init__(self, a):
        self.a = a
    def add(self, b):
        print(self.a + b)
    def multiply(self, b):
        print(self.a * b)

p = Parent(3)
p.add(4)
p.multiply(4)

class Child(Parent):
    def subtract(self, b):
        print(self.a - b)
    def divide(self, b):
        print(self.a / b)

c = Child(4)
c.add(3)
c.multiply(3)
c.subtract(3)
c.divide(3)
















