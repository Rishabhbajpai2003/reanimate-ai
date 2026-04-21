import sys
from pathlib import Path
import cv2
import numpy as np

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from pipeline.animate import AnimateModule

def test_animate_no_face():
    anim = AnimateModule()
    # Create an image that will likely not have a face
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite("scratch/no_face.png", img)
    
    try:
        print("Running animate.process (no face)...")
        anim.process("scratch/no_face.png", "scratch/no_face_out.gif")
        print("Success!")
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_animate_no_face()
