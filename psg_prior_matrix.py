import numpy as np
import json

NUM_CLASSES = 133

with open('./data/psg/psg.json', 'r') as f:
    data = json.load(f)

if isinstance(data, dict) and "data" in data:
    data = data["data"]

co_occurrence = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)

for ann in data:
    obj_cats = [obj['category_id'] for obj in ann['annotations']]
    for rel in ann['relations']:
        sub_idx, obj_idx, _ = rel
        if sub_idx >= len(obj_cats) or obj_idx >= len(obj_cats):
            continue
        sub_cls = obj_cats[sub_idx]
        obj_cls = obj_cats[obj_idx]
        if sub_cls >= NUM_CLASSES or obj_cls >= NUM_CLASSES:
            continue
        co_occurrence[sub_cls, obj_cls] += 1

np.save('psg_co_occurrence_matrix.npy', co_occurrence)

min_val = co_occurrence.min()
max_val = co_occurrence.max()
prior_matrix = 2 * (co_occurrence - min_val) / (max_val - min_val + 1e-6) - 1
np.save('psg_prior_matrix.npy', prior_matrix)

