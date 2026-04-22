import cv2
import numpy as np

def get_box():
    return (1, 2, 3, 4)

def fallback():
    eyes = [(1,1,1,1), (2,2,2,2)]
    mouth = (3,3,3,3)
    return eyes, mouth

def detect():
    # Simulate the landmark case
    return [get_box(), get_box()], get_box()

def test():
    print("Testing fallback...")
    eyes, mouth = fallback()
    print(f"eyes: {eyes}")
    print(f"mouth: {mouth}")
    x1, y1, x2, y2 = mouth
    print("Unpacked mouth OK")

    print("\nTesting detect...")
    eyes, mouth = detect()
    print(f"eyes: {eyes}")
    print(f"mouth: {mouth}")
    x1, y1, x2, y2 = mouth
    print("Unpacked mouth OK")

test()
