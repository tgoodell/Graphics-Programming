# ~ import base64

# ~ msg=b"This is a test"

# ~ #print(base64.a85_encode(msg))
# ~ #print(base64.b85_encode(msg))
# ~ print(base64.b64_encode(msg))
# ~ print(base64.b32_encode(msg))
# ~ print(base64.b16_encode(msg))

# ~ #print(base64.b85decode(msg1))

# ~ for letter in msg:
	# ~ print(int(letter))

# ~ for letter in msg:
	# ~ print(hex(letter))
	
# ~ for letter in msg:
	# ~ print(oct(letter))
	

# ~ for letter in msg:
	# ~ print(bin(letter))
	
# ~ nums=[84,104,105]

# ~ for num in nums:
	# ~ print(chr(num))
	
# ~ nums=[124,150,151,163,40]
# ~ nums=nums.split(",")
# ~ print(nums)
# ~ for num in nums:
	# ~ x=int(num,8) # Base8
	# ~ print(chr(x))


# Caesar Cipher / Rot13
msg="TGG KUSJWV QGM"
alphabet="abcdefghijklmnopqrstuvwxyz"
shift=5

def rotn(msg,shift):
	output=""
	for letter in msg:
		if letter in alphabet:
			index=alphabet.find(letter)
			index+=shift
			index%=26
			output+=alphabet[index]
		else:
			output+=letter
	return output
	
print(rotn(msg,1))

# ~ for i in range(26):
	# ~ print(str(i) + ": " + rotn(msg,i))
	
	
# Xor	
#msg=[0x21,0x30,0x36,0x2c,0x31,0x33,0x37,0x3F]
#key=b"SQUIRREL"

#for num, letter in zip(msg,key):
#	print(num ^ letter)
