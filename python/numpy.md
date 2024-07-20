# Numpy tips and tricks
> https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md
> https://medium.com/better-programming/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d
1. numpy插入的时间复杂度是`O(n)`，因为需要重新分配内存，而list插入的时间复杂度是`O(1)`，因为实现方式是指针
2. 初始化
    ```python
    np.zeros, np.zeros_like
    np.ones, np.ones_like
    np.empty, np.empty_like
    np.full(shape, fill_value), np.full_like(array, fill_value)
    ```
3. arange和linspace
    ```python
    np.arange(start, stop, step)
    np.linspace(start, stop, num)
    ```
4. random
    ```python
    np.random.randint(high)
    np.random.randint(low, high=None, size=None)
    np.random.rand(*size)
    np.random.randn(*size)
    np.random.uniform(low=0.0, high=1.0, size=None)
    ```
5. where
    ```python
    np.where(a > 5) <=> np.nonzero(a > 5)
    np.where(a > 5, 1, 0) <=> a[a <= 5] = 0, a[a > 5] = 1
    ```
6. 数组中找一个元素
    ```python
    np.searchsorted(array, item)  # 要求数组有序
    ```
7. 取整
    ```python
    np.floor()  # -inf
    np.ceil()   # inf
    np.round()  # nearest
    np.copysign(np.ceil(np.abs(array)), array)  # zero
    ```
8. 比较数组
    ```python
    np.allclose(A, B)
    np.array_equal(A, B)
    ```
9. 只读
    ```python
    array.flags.writeable = False
    ```
10. meshgrid
    ```python
    x, y = np.meshgrid(
        np.linspace(0,1,5),
        np.linspace(0,1,5)
    )
    ```
11. 数据类型相关信息
    ```python
    for dtype in [np.int8, np.int32, np.int64]:
        print(np.iinfo(dtype).min)
        print(np.iinfo(dtype).max)
    for dtype in [np.float32, np.float64]:
        print(np.finfo(dtype).min)
        print(np.finfo(dtype).max)
        print(np.finfo(dtype).eps)
    ```
12. repeat
    ```python
    C = np.bincount([1,1,2,3,4,4,6])
    A = np.repeat(np.arange(len(C)), C)
    ```
13. 前n大的数
    ```python
    # Slow
    print (Z[np.argsort(Z)[-n :]])
    # Fast
    print (Z[np.argpartition(-Z, n)[: n]])
    ```
14. contiguous问题
    ```python
    # Use array.flags['C_CONTIGUOUS'] to check
    a = np.ones((4, 4))             # True
    b = a[1 : 3, 2 :]               # False
    c = a[1 : 3, 2 :].copy()        # True
    d = a[1 : 3, 2 :].reshape(-1)   # True
    ```