# Multiprocess

> https://stackoverflow.com/questions/8533318/multiprocessing-pool-when-to-use-apply-apply-async-or-map

## multiprocessing

```python
from multiprocessing import Pool


def func1(a):
    return a + 1

def func2(a, b):
    return a + b

pool = Pool(4)
case = 2
if case == 1:
    res = pool.map(func1, [1, 2, 3])
elif case == 1:
    res = pool.starmap(func2, zip(range(10), range(100, 110)))
pool.close()
pool.join()
```