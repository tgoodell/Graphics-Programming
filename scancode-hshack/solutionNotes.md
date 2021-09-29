# Computers can See 3 Solution Notes

The trick is to figure out where the data is stored. In this case, all of the data is stored in the 13 x 13 square in the center. From there, just know that each character is stored in binary. So, the really secret sauce is:

``` 
def decode_binary_string(s):
    return ''.join(chr(int(s[i*8:i*8+8],2)) for i in range(len(s)//8))
```

The flag can either be extracted by hand or via program. 

Points: 500
Flag: FLAG{Denso}
