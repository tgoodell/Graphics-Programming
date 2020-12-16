import numpy as np

yo=10 # Initial Height
vo=10 # Initial Velocity
x=10 # Target Distance

# Assumptions
xo=0 # Initial Distance
y=0 # Final Height
g=9.81 # Gravity

def getDistance(a):
    theta = a * np.pi / 180  # Using i to get theta so Numpy can work with it
    vox = vo * np.cos(theta)  # Calculating X component of Velocity
    voy = vo * np.sin(theta)  # Calculating Y component of Velocity
    t = (np.sqrt(2 * g * yo + np.square(voy)) + voy) / g  # Calculating time from the vertical Kinematics equation
    testx = xo + vox * t  # Calculating the distance reached with the horizontal Kinematics equation using the test angle
    return testx

def getTopAngle():
    a=0
    topAngle=0
    while(a<90):
        if(getDistance(a)>getDistance(topAngle)):
            topAngle=a
        a=a+1
    return topAngle

def getAngle1():
    a=0
    tangle=0
    i=1
    topAngle=getTopAngle()
    while (a < topAngle):
        if (np.abs(x-getDistance(a))<np.abs(x-getDistance(tangle))):
            tangle=a
        a+=i
    return tangle

def getAngle2():
    a=getTopAngle()
    tangle=0
    i=1
    while (a < 90):
        if (np.abs(x-getDistance(a))<np.abs(x-getDistance(tangle))):
            tangle=a
        a+=i
    return tangle

def newton(theta):
    otheta=theta-(yo*np.square(np.cos(theta*np.pi/180))+0.5*x*np.sin(2*theta*np.pi/180)-(g*np.square(x))/2*np.square(vo))/(-2*yo*np.sin(theta*np.pi/180)*np.cos(theta*np.pi/180)+x*np.cos(2*theta*np.pi/180)-((4*np.square(vo)*g*x*4*g*np.square(x)*vo)/4*np.power(vo,4)))
    return otheta

otheta=getAngle1()
for i in range(1000):
    otheta=newton(otheta)

if(otheta>0):
    print(otheta)

otheta=getAngle2()
for i in range(1000):
    otheta=newton(otheta)

if(otheta>0):
    print(otheta)


#3.9999894952713615
#69.99998949513461
