## Python基础入门

本教程将介绍Python编程的基础知识。完成本教程后，你将能够编写简单的Python程序并理解基本概念。

### Python简介

Python是一种高级、解释型编程语言(high-level, interpreted programming language)，以其可读性和易用性而闻名。它被广泛应用于数据分析、人工智能、网络开发等领域。让我们从一些基本概念开始。

### 变量 (Variables) 和数据类型 (Data Types)

在Python中，我们可以使用变量来存储数据。

```python
# 整数（Integer）
age = 25
print("年龄:", age)

# 浮点数（Float）
height = 170.5
print("身高:", height)

# 字符串（String）
name = "小明"
print("姓名:", name)

# 布尔值（Boolean）
is_student = True
print("是否是学生:", is_student)
```

### 基本运算

Python可以执行基本的算术运算。

```python
# 加法
print("5 + 3 =", 5 + 3)

# 减法
print("10 - 2 =", 10 - 2)

# 乘法
print("4 * 3 =", 4 * 3)

# 除法
print("16 / 2 =", 16 / 2)

# 取余（求余数）
print("10 除以 3 的余数是:", 10 % 3)
```

### 列表（List）

列表用于在单个变量中存储多个项目(elements)。

```python
my_list = [1, 2, 3, 4, 5]
print("我的列表:", my_list)

# 访问元素
print("第一个元素:", my_list[0])  # 注意：Python的索引从0开始
print("最后一个元素:", my_list[-1])

# 添加元素
my_list.append(6)
print("添加元素后:", my_list)

# 删除元素
my_list.remove(3)
print("删除元素后:", my_list)
```

### 循环 (Loops)

循环用于重复执行一块代码。

#### `for` 循环

```python
print("使用for循环打印0到4:")
for i in range(5):
    print(i)
```

#### `while` 循环

```python
print("使用while循环打印0到4:")
count = 0
while count < 5:
    print(count)
    count += 1  # 这相当于 count = count + 1
```

### 函数 (Functions)

函数是执行特定任务的代码模块。

```python
def greet(name):
    return f"你好，{name}！"

print(greet("小明"))
print(greet("小红"))
```

### 导入库（Import Libraries）
> 库 (Library)，包 (packages)，模块(modules)

在Python中，我们通常使用 import 语句来导入这些资源。例如：
```python
import math  # 导入标准库 (Importing a standard library)
from datetime import datetime  # 从库中导入特定模块 (Importing a specific module from a library)
import numpy as np  # 导入第三方库并使用别名 (Importing a third-party library with an alias)
```

Python有丰富的库，例如math，你可以导入并在程序中使用它们。



```python
import math

print("16的平方根是:", math.sqrt(16))
print("圆周率π约等于:", math.pi)
```


### 类和面向对象编程 (Classes and Object-Oriented Programming)

面向对象编程（OOP）是一种编程范式，它使用"对象"来组织代码。在Python中，我们使用类来创建对象。

#### 类的基本概念 (Basic Concepts of Classes)

类是一种用户自定义的数据类型，它定义了一类对象的属性和方法。

```python
class Dog:
    # 初始化方法（构造函数）
    def __init__(self, name, age):
        self.name = name  # 属性
        self.age = age

    # 方法
    def bark(self):
        return f"{self.name}正在汪汪叫！"

    def get_info(self):
        return f"{self.name}今年{self.age}岁了。"

# 创建Dog类的实例
my_dog = Dog("小黑", 3)

# 使用对象的方法
print(my_dog.bark())
print(my_dog.get_info())
```

#### 继承 (Inheritance)

继承允许我们基于一个已存在的类创建新类。

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

class Cat(Animal):
    def speak(self):
        return f"{self.name}喵喵叫！"

class Bird(Animal):
    def speak(self):
        return f"{self.name}吱吱叫！"

# 创建实例
my_cat = Cat("咪咪")
my_bird = Bird("小鸟")

print(my_cat.speak())
print(my_bird.speak())
```

#### 封装 (Encapsulation)

封装是将数据和操作数据的方法绑定在一起的一种方式。

```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # 私有属性

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return True
        return False

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return True
        return False

    def get_balance(self):
        return self.__balance

# 使用BankAccount类
my_account = BankAccount(1000)
my_account.deposit(500)
my_account.withdraw(200)
print(f"当前余额：{my_account.get_balance()}元")
```

#### 多态 (Polymorphism)

多态允许使用一个接口来引用不同类型的对象。

```python
def animal_speak(animal):
    print(animal.speak())

# 使用之前定义的Cat和Bird类
cat = Cat("咪咪")
bird = Bird("小鸟")

animal_speak(cat)  # 输出：咪咪喵喵叫！
animal_speak(bird)  # 输出：小鸟吱吱叫！
```

### 面向对象编程的优点

1. **代码重用**：通过继承，我们可以重用已有的代码。
2. **封装**：数据可以隐藏在对象内部，提高了安全性。
3. **模块化**：对象可以看作是程序的构建块，使大型程序更容易管理。
4. **灵活性**：通过多态，我们可以用统一的方式处理不同类型的对象。

### 实践练习

1. 创建一个包含你最喜欢的水果名称的列表，然后打印出来。(Create a list of your favorite fruits and print it.)
2. 编写一个函数，接受一个数字作为参数，返回该数字的平方。(Write a function that takes a number as a parameter and returns its square.)
3. 使用for循环打印出1到10的所有偶数。(Use a for loop to print all even numbers from 1 to 10.)
4. 创建一个简单的计算器程序，能够进行加、减、乘、除运算。(Create a simple calculator program that can perform addition, subtraction, multiplication, and division.)
   
**类与面向对象**
1. 创建一个Student类，包含姓名、年龄和成绩属性，以及一个打印学生信息的方法。(Create a Student class with name, age, and grade attributes, and a method to print student information.)
2. 基于Student类，创建一个GraduateStudent子类，添加研究方向属性。(Based on the Student class, create a GraduateStudent subclass, adding a research field attribute.)
3. 创建一个简单的图书管理系统，包含Book和Library类。(Create a simple library management system with Book and Library classes.)

理解面向对象编程需要一些时间和实践。随着你编写更多的Python代码，这些概念会变得更加清晰。记住，编程是一项需要不断练习的技能，所以要多多实践！

### 小贴士

- Python对缩进很敏感，它用来划分代码块。确保你的代码缩进正确。
- 使用`#`来添加注释，解释你的代码。
- 多练习！编程是一项需要不断实践的技能。

祝你编程愉快！随着练习的增加，你会发现Python是一个强大而有趣的编程语言。或许会出现即便掌握了其他语言，能用Python还是用Python的情况。


### 练习示例

1. 创建一个包含你最喜欢的水果名称的列表，然后打印出来。

```python
favorite_fruits = ["苹果", "香蕉", "草莓", "芒果", "葡萄"]
print("我最喜欢的水果有：")
for fruit in favorite_fruits:
    print(fruit)
```

2. 编写一个函数，接受一个数字作为参数，返回该数字的平方。

```python
def square(number):
    return number ** 2

# 测试函数
print(f"5的平方是：{square(5)}")
print(f"9的平方是：{square(9)}")
```

3. 使用for循环打印出1到10的所有偶数。

```python
print("1到10的偶数是：")
for num in range(1, 11):
    if num % 2 == 0:
        print(num)
```

4. 创建一个简单的计算器程序，能够进行加、减、乘、除运算。

```python
def calculator():
    num1 = float(input("请输入第一个数字："))
    num2 = float(input("请输入第二个数字："))
    operation = input("请选择运算（+、-、*、/）：")

    if operation == '+':
        result = num1 + num2
    elif operation == '-':
        result = num1 - num2
    elif operation == '*':
        result = num1 * num2
    elif operation == '/':
        if num2 != 0:
            result = num1 / num2
        else:
            return "错误：除数不能为0"
    else:
        return "错误：无效的运算符"

    return f"{num1} {operation} {num2} = {result}"

print(calculator())
```

5. 创建一个Student类，包含姓名、年龄和成绩属性，以及一个打印学生信息的方法。

```python
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def print_info(self):
        print(f"学生姓名：{self.name}")
        print(f"年龄：{self.age}")
        print(f"成绩：{self.grade}")

# 测试Student类
student1 = Student("张三", 18, 85)
student1.print_info()
```

6. 基于Student类，创建一个GraduateStudent子类，添加研究方向属性。

```python
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def print_info(self):
        print(f"学生姓名：{self.name}")
        print(f"年龄：{self.age}")
        print(f"成绩：{self.grade}")

class GraduateStudent(Student):
    def __init__(self, name, age, grade, research_field):
        super().__init__(name, age, grade)
        self.research_field = research_field

    def print_info(self):
        super().print_info()
        print(f"研究方向：{self.research_field}")

# 测试GraduateStudent类
grad_student = GraduateStudent("李四", 24, 90, "人工智能")
grad_student.print_info()
```

7. 创建一个简单的图书管理系统，包含Book和Library类。

```python
class Book:
    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_borrowed = False

    def __str__(self):
        return f"{self.title} by {self.author} (ISBN: {self.isbn})"

class Library:
    def __init__(self):
        self.books = []

    def add_book(self, book):
        self.books.append(book)
        print(f"已添加书籍：{book}")

    def remove_book(self, isbn):
        for book in self.books:
            if book.isbn == isbn:
                self.books.remove(book)
                print(f"已移除书籍：{book}")
                return
        print("未找到该书籍")

    def borrow_book(self, isbn):
        for book in self.books:
            if book.isbn == isbn:
                if not book.is_borrowed:
                    book.is_borrowed = True
                    print(f"已借出书籍：{book}")
                    return
                else:
                    print("该书已被借出")
                    return
        print("未找到该书籍")

    def return_book(self, isbn):
        for book in self.books:
            if book.isbn == isbn:
                if book.is_borrowed:
                    book.is_borrowed = False
                    print(f"已归还书籍：{book}")
                    return
                else:
                    print("该书未被借出")
                    return
        print("未找到该书籍")

    def list_books(self):
        if not self.books:
            print("图书馆暂无藏书")
        else:
            print("图书馆藏书列表：")
            for book in self.books:
                status = "已借出" if book.is_borrowed else "可借阅"
                print(f"{book} - {status}")

# 测试图书管理系统
library = Library()

book1 = Book("Python编程", "张三", "1234567890")
book2 = Book("数据结构与算法", "李四", "0987654321")

library.add_book(book1)
library.add_book(book2)

library.list_books()

library.borrow_book("1234567890")
library.list_books()

library.return_book("1234567890")
library.list_books()
```

这些示例应该能帮助学生更好地理解和实践Python编程的基本概念和面向对象编程。鼓励学生修改这些代码，尝试添加新功能或改进现有功能，以加深他们的理解。