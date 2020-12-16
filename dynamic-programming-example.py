import random
import functools

# houses=[random.randint(2,255) for i in range(100)]
# print(houses)

houses=[188, 50, 214, 33, 103, 117, 2, 176, 111, 253, 94, 226, 252, 208, 164, 20, 238, 196, 98, 252, 239, 175, 34, 255, 52, 9, 221, 159, 193, 37, 40, 210, 78, 244, 192, 78, 79, 127, 138, 105, 216, 134, 138, 193, 87, 35, 40, 162, 196, 160, 80, 44, 19, 252, 20, 75, 94, 211, 13, 181, 105, 141, 122, 47, 189, 249, 10, 66, 115, 248, 105, 201, 68, 101, 143, 143, 246, 61, 159, 115, 230, 136, 169, 166, 82, 255, 234, 129, 68, 100, 106, 209, 230, 45, 17, 110, 205, 219, 218, 196]

# dynamic approach
def dp(h):
    r=n=0
    for v in h:
        n,r=max(r,n),v+n
    return max(r,n)

@functools.lru_cache(maxsize=3)
def solve(h):
    if not h:
        return 0
    a,*h=h
    v1=solve(tuple(h[1:]))+a
    v2=solve(tuple(h))
    return max(v1,v2)

print(solve(tuple(houses)))
print(dp(houses))