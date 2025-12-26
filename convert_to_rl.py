import json
import os
import re
from datetime import datetime

# 配置路径
INPUT_FILE = "dataset_output_v1/dataset.json"
DATA_SOURCE_NAME = "chart_vqa_rl_train"
# 使用 os.getcwd() 获取当前工作目录
WORKSPACE_ROOT = os.getcwd()
TODAY_DATE = datetime.now().strftime("%Y%m%d")

OUTPUT_FILE = f"rl_dataset_{TODAY_DATE}.jsonl"

def extract_answer_letter(gpt_response):
    """从 GPT 回复中提取选项字母 (例如 'The correct answer is C.')"""
    match = re.search(r"The correct answer is ([A-F])", gpt_response)
    if match:
        return match.group(1)
    return "A"  # Fallback

def convert_to_rl():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        raw_data = json.load(f)

    # 修改为写入 jsonl
    with open(OUTPUT_FILE, 'w') as f_out:
        for idx, item in enumerate(raw_data):
            # 1. 获取基础信息
            user_input_full = item['conversations'][0]['value'] 
            gpt_response = item['conversations'][1]['value']
            
            # 2. 提取正确答案 (Ground Truth)
            final_answer = extract_answer_letter(gpt_response)
            
            # 3. 构建 Image Path (使用第一张无红线的原图)
            img_filename = item['image'][0] 
            img_abs_path = os.path.join(WORKSPACE_ROOT, "dataset_output_v1", "images", img_filename)

            # 4. 构建 RL Entry
            entry = {
                "data_source": DATA_SOURCE_NAME,
                "prompt": [
                    {
                        "role": "user",
                        "content": user_input_full
                    }
                ],
                "reward_model": {
                    "ground_truth": [final_answer], # 答案列表
                    "style": "rule"
                },
                "image": [img_abs_path],
                "image_search_title_list": [],
                "image_search_thumbnail_list": [],
                "id": f"{DATA_SOURCE_NAME}_{idx+1}"
            }
            
            f_out.write(json.dumps(entry) + '\n')

    print(f"Successfully converted {len(raw_data)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_to_rl()
