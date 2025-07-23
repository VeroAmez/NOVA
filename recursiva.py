def recursiva(n):
    print(n)
    if n==1:
        return 0
    else:
        return recursiva(n-1)

print(recursiva(10))
