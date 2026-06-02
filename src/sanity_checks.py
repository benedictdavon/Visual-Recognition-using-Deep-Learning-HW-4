from types import SimpleNamespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataset_utils import HW4RestorationDataset, HW4TestDataset

args = SimpleNamespace(
    hw4_data_root='data',
    hw4_split_file='splits/hw4_split_seed42.json',
    hw4_val_per_task=160,
    hw4_seed=42,
    patch_size=128,
)

train_set = HW4RestorationDataset(args, split='train')
val_set = HW4RestorationDataset(args, split='val')
test_set = HW4TestDataset('data/test/degraded')
print('train:', len(train_set))
print('val:', len(val_set))
print('test:', len(test_set))
print('first train sample:', train_set[0][0])
print('first val sample:', val_set[0][0])
print('first test sample:', test_set[0][0])
