import os
import sys
import dbm
import time
import pickle
from loguru import logger
from transformers import BertTokenizer, BertModel
import torch

def replace_name(text):
    if '-stuff' in text:
        text = text.replace('-stuff', '')
    if '-merged' in text:
        text = text.replace('-merged', '')
    if '-other' in text:
        text = text.replace('-other', '')
    return text

object_categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
                     'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
object_categories = [replace_name(x) for x in object_categories]
relation_categories = ['over', 'in front of', 'beside', 'on', 'in', 'attached to', 'hanging from', 'on back of', 'falling off', 'going down', 'painted on', 'walking on', 'running on', 'crossing', 'standing on', 'lying on', 'sitting on', 'flying over', 'jumping over', 'jumping from', 'wearing', 'holding', 'carrying', 'looking at', 'guiding', 'kissing',
                       'eating', 'drinking', 'feeding', 'biting', 'catching', 'picking', 'playing with', 'chasing', 'climbing', 'cleaning', 'playing', 'touching', 'pushing', 'pulling', 'opening', 'cooking', 'talking to', 'throwing', 'slicing', 'driving', 'riding', 'parked on', 'driving on', 'about to hit', 'kicking', 'swinging', 'entering', 'exiting', 'enclosing', 'leaning on']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("./work_dirs/checkpoints/bert-base-uncased")
model = BertModel.from_pretrained("./work_dirs/checkpoints/bert-base-uncased").to(device).eval()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return embedding

def process(text_dir, embed_dir, bin=1, part=0):
    if not os.path.exists(text_dir):
        logger.error("text_dir: {} not exists".format(text_dir))
        return
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)
    text_kv_db = dbm.open(os.path.join(text_dir, 'kv_triplet.db'), 'r')
    embed_kv_db = dbm.open(os.path.join(embed_dir, 'kv_triplet.db'), 'c')

    skipped_keys_path = os.path.join(embed_dir, "skipped_triplet_keys.txt")
    skipped_f = open(skipped_keys_path, "w", encoding="utf-8")

    total = len(object_categories) * len(object_categories) * len(relation_categories) // bin
    finished = 0
    skipped = 0
    t0 = time.time()
    for i, subject in enumerate(object_categories):
        if not (i >= (part * len(object_categories) // bin) and i < ((part + 1) * len(object_categories) // bin)):
            continue
        for j, object in enumerate(object_categories):
            for k, relation in enumerate(relation_categories):
                key = f"{subject}#{object}#{relation}"
                key_bytes = key.encode()
                if key_bytes in embed_kv_db:
                    continue
                try:
                    text = pickle.loads(text_kv_db[key_bytes])
                except Exception as e:
                    logger.warning(f"读取三元组描述失败，跳过：{key}，错误：{e}")
                    skipped_f.write(key + "\n")
                    skipped += 1
                    continue
                if not isinstance(text, str) or not text.strip():
                    logger.warning(f"三元组描述为空，跳过：{key}")
                    skipped_f.write(key + "\n")
                    skipped += 1
                    continue
                try:
                    embed = get_embedding(text)
                    embed_kv_db[key_bytes] = pickle.dumps(embed)
                    finished += 1
                except Exception as e:
                    logger.warning(f"编码失败，跳过：{key}，错误：{e}")
                    skipped_f.write(key + "\n")
                    skipped += 1
                    continue

                # 进度显示
                t1 = time.time()
                speed = finished / max(t1 - t0, 1e-6)
                left = (total - (i * len(object_categories) * len(relation_categories) + j * len(relation_categories) + k + 1)) / max(speed, 1e-6)
                logger.info(
                    f"speed: {speed:.4f}, left: {left:.2f}s, "
                    f"{subject}: {i}/{len(object_categories)}, {object}: {j}/{len(object_categories)}, {relation}: {k}/{len(relation_categories)} "
                    f"({finished}/{total}), skipped: {skipped}"
                )

    skipped_f.close()
    text_kv_db.close()
    embed_kv_db.close()

if __name__ == '__main__':
    text_dir = sys.argv[1]
    embed_dir = sys.argv[2]
    bin = int(float(sys.argv[3])) if len(sys.argv) > 3 else 1
    part = int(float(sys.argv[4])) if len(sys.argv) > 4 else 0
    process(text_dir, embed_dir, bin, part)
#CUDA_VISIBLE_DEVICES=1 PYTHONPATH='.':$PYTHONPATH python tools/bert_embed.py ./pair_llm_db/qwen2.5_triplet_db1_merged ./pair_llm_db/qwen2.5_triplet_db1_test1_embed 1 0
#CUDA_VISIBLE_DEVICES=1 PYTHONPATH='.':$PYTHONPATH python tools/bert_embed.py ./pair_llm_db/qwen2.5_triplet_db ./pair_llm_db/qwen2.5_triplet_db_embed 1 0