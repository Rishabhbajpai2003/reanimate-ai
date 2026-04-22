import sys
from pathlib import Path
import cv2
import numpy as np

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from pipeline.animate import AnimateModule

def test_animate():
    anim = AnimateModule()
    # Create a dummy image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.imwrite("scratch/dummy.png", img)
    
    try:
        print("Running animate.process...")
        anim.process("scratch/dummy.png", "scratch/dummy_out.gif")
        print("Success!")
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_animate()
