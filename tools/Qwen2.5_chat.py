import os
import sys
import dbm
import time
import pickle
import requests
import json
from loguru import logger
from retry import retry



# LM Studio API配置
API_BASE_URL = "http://localhost:1234/v1"  # LM Studio默认API地址
MAX_RETRIES = 3
TIMEOUT = 300  # 请求超时时间（秒），根据模型响应速度调整

# 保留原有参数配置
temperature = 0.6
top_p = 0.9
max_gen_len = 256  # 限制生成文本长度


def replace_name(text):
    if '-stuff' in text:
        text = text.replace('-stuff', '')
    if '-merged' in text:
        text = text.replace('-merged', '')
    if '-other' in text:
        text = text.replace('-other', '')
    return text


# 类别列表保持不变
object_categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
                     'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
object_categories = [replace_name(x) for x in object_categories]
relation_categories = ['over', 'in front of', 'beside', 'on', 'in', 'attached to', 'hanging from', 'on back of', 'falling off', 'going down', 'painted on', 'walking on', 'running on', 'crossing', 'standing on', 'lying on', 'sitting on', 'flying over', 'jumping over', 'jumping from', 'wearing', 'holding', 'carrying', 'looking at', 'guiding', 'kissing',
                       'eating', 'drinking', 'feeding', 'biting', 'catching', 'picking', 'playing with', 'chasing', 'climbing', 'cleaning', 'playing', 'touching', 'pushing', 'pulling', 'opening', 'cooking', 'talking to', 'throwing', 'slicing', 'driving', 'riding', 'parked on', 'driving on', 'about to hit', 'kicking', 'swinging', 'entering', 'exiting', 'enclosing', 'leaning on']



@retry(tries=MAX_RETRIES, delay=5)
def get_triplet_level_description(subject, object, relation):
    messages = [
        {
            "role": "system", 
            "content": "You are asked to play the role of a relation judger. Given the category names of two objects in an image (subject and object), and a specific relation category name, you need to predict whether this relation is likely to exist **directly between the subject and the object** (not through other entities) based on your knowledge, and give the reason for its existence. Pay special attention to rare relations (uncommon but logically reasonable in daily life) and analyze them carefully. Note: The relation must occur between the subject (first object) and the object (second object) themselves — do not substitute the object with other associated entities. Please give me an example. Output ONLY in English, no Chinese."
        },
        {
            "role": "user", 
            "content": "For example, the input is: the subject is a 'person', the object is a 'bench' and the relation is 'lying on'. The output should be Yes, the relation is likely to exist in the image. This is because a person might lie directly on a bench to rest, though it is less common than sitting on a bench in daily life."
        },
        {
            "role": "assistant",
            "content": "Ok, I got it. Please give me the subject, object and relation names. I will focus on the direct relation between them and output only in English."
        },
        {
            "role": "user", 
            "content": "The subject is a {}, the object is a {}, and the relation is {}".format(subject, object, relation)
        }
    ]

    payload = {
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_gen_len,
        "stream": False
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=TIMEOUT
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API请求失败: {response.status_code}, {response.text}")


def process(db_dir, bin=36, part=0):
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    kv_db = dbm.open(os.path.join(db_dir, 'kv_triplet.db'), 'c')
    t0 = time.time()
    for i, subject in enumerate(object_categories):
        if not (i >= (part * len(object_categories) // bin) and i < ((part + 1) * len(object_categories) // bin)):
            continue
        for j, object in enumerate(object_categories):
            for k, relation in enumerate(relation_categories):
                key = f"{subject}#{object}#{relation}"
                if key in kv_db:
                    continue
                t1 = time.time()
                speed = 1 / max(t1 - t0, 1e-6)
                left = ((len(object_categories) - i - 1) * len(object_categories) * len(relation_categories) // bin +
                        (len(object_categories) - j - 1) * len(relation_categories) +
                        len(relation_categories) - k - 1) * max(t1 - t0, 1e-6)
                logger.info(f"speed: {speed:.4f}, left: {left:.2f}s, {subject}: {i}/{len(object_categories)}, {object}: {j}/{len(object_categories)}, {relation}: {k}/{len(relation_categories)}")
                t0 = time.time()
                try:
                    description = get_triplet_level_description(subject, object, relation)
                    kv_db[key] = pickle.dumps(description)
                    with open(os.path.join(db_dir, "triplet_llm_answers.txt"), "a", encoding="utf-8") as f:
                        f.write(f"{subject}\t{object}\t{relation}\t{description.strip()}\n")
                except Exception as e:
                    logger.warning(f"请求失败: {e}")
                    time.sleep(10)
                    continue
    kv_db.close()


if __name__ == '__main__':
    db_dir = sys.argv[1]
    bin = int(float(sys.argv[2]))
    part = int(float(sys.argv[3]))
    process(db_dir, bin, part)

#CUDA_VISIBLE_DEVICES=0 PYTHONPATH='.':$PYTHONPATH python tools/Qwen2.5_chat.py ./pair_llm_db/qwen2.5_triplet_db 1 0

