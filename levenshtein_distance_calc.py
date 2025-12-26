def lev(a, b):
    if (len(a) == 0): return len(b)
    if (len(b) == 0): return len(a)
    if (a[0] == b[0]): return lev(a[1:], b[1:])
    else:
        return (1 + min(lev(a, b[1:]), lev(a[1:], b), lev(a[1:], b[1:])))

a, b = input().strip().split()
print(a, b)
print(lev(a, b))