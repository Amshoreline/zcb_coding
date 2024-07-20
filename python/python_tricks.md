# Python编程的一些tricks

1. iPython是好东西，打开后输入`dir(object)`可以查看所有方法
2. 使用tuple的时候记得打逗号，尤其是只有一个元素时
3. 字符串

    ```python
    pound = 9
    pence = 15
    # First format
    'The price of the T-shirt is %d pounds and %d pence' % (pound, pence)
    # Second format
    'The price of the T-shirt is {} pounds and {} pence'.format(pound, pence)
    # Third format
    f'The price of the T-shirt is {pound} pounds and {pence} pence'
    # Number
    loss = 0.123456
    print(f'loss is {loss:.4f}')    # loss is 0.1235
    number = 123_456
    print(f'number is {number:,}')  # number is 123,456
    # List
    names = ['Alice', 'Bob', 'Carol']
    str_names = ','.join(names)
    # Auto formatting
    a_list = [1, 2, 3]
    assert a_list == eval(str(a_list))
    ```

4. 赋值

    ```python
    a, *b = (1, 2, 3)       # b = [2, 3]
    a, *b = (1, )           # b = []
    a, *b, c = (1, 2, 3, 4) # b = [2, 3]
    ```

5. 条件语句

   ```python
   condition = True
   x = 1 if condition else 0
   ```

6. 读写

    ```python
    with open('filename', 'r') as reader:
        content = reader.read()
    with open('filename', 'w') as writer:
        writer.write(content)
    ```

7. 排序

   ```python
   a_list = [(20, 'Alice'), (18, 'Bob'), (19, 'Carol')]
   a_list.sort(key=lambda x: x[0], reverse=True)
   ```

8. 迭代器

    ```python
    names = ['Corey', 'Chris', 'Dave', 'Travis']
    # Build a list
    lower_names = [name.lower for name in names]
    # Use "enumreate"
    for index, name in enumerate(names):
        print(index, name)
    # Use "zip"
    ages = [16, 32, 21, 20]
    for index, (name, age) in enumerate(zip(names, ages)):
        print(index, name, age)
    # Use "zip" and "*"
    name_age_list = list(zip(*(names, ages)))   # [(Corey, 16), ...]
    ```

9. 字典

    ```python
    info = {'a': 1, 'b': 2, 'c': 3}
    info = dict([('a', 1), ('b', 2), ('c', 3)])
    for key in info:    # or info.keys()
        print(key)
    for key, value in info.items():
        print(key, value)
    ```

10. 数组

    ```python
    import numpy as np
    import torch

    # Invert an array
    array = np.arange(100)
    array[:: -1]
    tensor = torch.arange(100)  # negative step is not supported
    # Use "..."
    array = np.arange(32).reshape(2, 2, 2, 2, 2)
    array[0, ..., 0]
    tensor = torch.arange(32).view(2, 2, 2, 2, 2)
    tensor[0, ..., 0]
    ```

11. 类的属性

    ```python
    class Person():
        pass
    person = Person()
    person_info = {'first': 'Corey', 'last': 'Schafer'}
    for key, value in person_info.items():
        setattr(person, key, value)
    for key in person_info.keys():
        print(getattr(person, key))
    ```

12. 密码

    ```python
    from getpass import getpass

    username = input('username: ')
    password = getpass('password: ')
    ```