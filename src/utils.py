import numpy as np

# Calculation for points angles 11 , 13 and 15 - calculate angles of any 3 points

def calculate_angle(a,b,c):
    a=np.array(a) # Point 11 First
    b=np.array(b) # Point 13 Mid
    c=np.array(c) # Point 15 End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0]) # Y end - y mid and x value then first and mid point.
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360 -angle # adjust angles as our joints cant rotate more than 180


    return angle
