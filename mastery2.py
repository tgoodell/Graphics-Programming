# Import Numpy for Math
import numpy as np

# Intitials
yo=.8 # Initial Height
vo=5 # Initial Velocity
x=1.23 # Target Distance

m=0.0005 # The margin of error of the angle
p=0.001 # The precision of the angle

# Assumptions
xo=0 # Initial Distance
y=0 # Final Height
g=9.81 # Gravity

angles=[] # Build the array for launch angles to be stored in

def getDistance(a): # Function that spits out distance given an angle, using the initial conditions set above
    theta = a*np.pi/180 # Using i to get theta so Numpy can work with it
    vox = vo*np.cos(theta) # Calculating X component of Velocity
    voy = vo*np.sin(theta) # Calculating Y component of Velocity
    t = (np.sqrt(2*g*yo+np.square(voy))+voy)/g # Calculating time from the vertical Kinematics equation
    testx = xo+vox*t # Calculating the distance reached with the horizontal Kinematics equation using the test angle
    return testx

def bruteforce(): # Function that bruteforces the launch angle to hit the target, using the initial conditions set above
    a = 0  # Declare Angle of Projectile Launcher
    while (a <= 90):  # Increment from 0 to 90 using p
        testx=getDistance(a) # Use getDistance function to get a distance to test against target distance.
        if (testx > x-m and testx < x+m):  # Comparing the testx to target distance with desired margin of error
            if len(angles)==0: # If there are no angles in angles[]
                angles.append(a) # Add the angle to angles[]
            elif len(angles)==1: # If there is already an angle in angles[]
                if np.abs(a-angles[0])<=1: # And the angle is not unique (already in angles[])
                    pass # Do nothing
                else: # If the angle is unique
                    angles.append(a) # Add the angle to angles[]
            elif len(angles)==2: # If there are already two angles in angles[]
                if np.abs(a-angles[0])<=1 or np.abs(a-angles(1))<=1: # And the angle is not unique (already in angles[])
                    pass # Do nothing
                else: # If the angle is unique
                    angles.append(a) # Add the angle to angles[]
        a += p  # Increment by the precision

    if not angles: # If no angles were recorded
        print("Launch is not possible.")
    else: # If angles were recorded
        print(angles)

    return angles

bruteforce() # Execute bruteforce function
