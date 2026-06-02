from pathlib import Path
from PIL import Image
from collections import Counter

dirs = [
    "data/train/degraded",
    "data/train/clean",
    "data/test/degraded",
    "data/test_release/degraded",
    "data/test",
]
for d in dirs:
    p = Path(d)
    if not p.exists():
        continue
    imgs = list(p.glob("*.png"))
    if not imgs:
        continue
    sizes = Counter(Image.open(x).size for x in imgs)
    print(f"\n{d}: {len(imgs)} images")
    for size, count in sorted(sizes.items()):
        print(f"  {size[0]}x{size[1]}: {count}")
