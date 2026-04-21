import sys
from pathlib import Path
import cv2
import numpy as np

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from pipeline.animate import AnimateModule

def test_animate_real():
    anim = AnimateModule()
    img_path = "backend/static/uploads/03870e08d136408fb6af4bc80780364d_input.jpeg"
    
    try:
        print(f"Running animate.process for {img_path}...")
        anim.process(img_path, "scratch/real_out.gif")
        print("Success!")
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_animate_real()
