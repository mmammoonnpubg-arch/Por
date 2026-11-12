!pip install -q fal-client pillow
import os
os.environ["FAL_KEY"] = "ضع_مفتاحك_هنا"
!python barca_girl.py --in IMG_7072.png --out ./out --video
from IPython.display import Image as I, display
import glob
for f in sorted(glob.glob("out/*.png")):
    print(f); display(I(f, width=300))