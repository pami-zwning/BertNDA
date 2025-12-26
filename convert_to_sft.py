import json
import os
import re
from datetime import datetime

# 配置路径
INPUT_FILE = "dataset_output_v1/dataset.json"
# 使用 os.getcwd() 获取当前工作目录
WORKSPACE_ROOT = os.getcwd()
TODAY_DATE = datetime.now().strftime("%Y%m%d")

OUTPUT_FILE = f"sft_dataset_{TODAY_DATE}.jsonl"

SYSTEM_PROMPT = "You are an AI assistant specialized in visual analysis and chart interpretation. You have access to a tool named 'draw_line' to perform precise measurements."

def extract_answer_letter(gpt_response):
    """从 GPT 回复中提取选项字母 (例如 'The correct answer is C.')"""
    match = re.search(r"The correct answer is ([A-F])", gpt_response)
    if match:
        return match.group(1)
    return "A"  # Fallback

def generate_cot_and_tool(item, question_text):
    """生成思维链 (CoT) 和工具调用"""
    coords = item['line_pixel_coords']
    
    # 构建思维链内容
    # 这里的 Global Context 是简化的，因为原始 json 没有存 metadata
    # 实际生产中最好在 generate_charts.py 中就把 metadata 存下来
    cot_thinking = (
        f"<thinking>\n"
        f"**Global Context:** The chart visualizes a data relationship plotted on a Cartesian coordinate system. "
        f"The visual elements include axes, data curves/points, and a legend.\n\n"
        f"**Target Identification:** The user is asking a specific question regarding values or comparisons at a specific coordinate. "
        f"I need to locate the relevant data point described in the query: \"{question_text.splitlines()[0]}\".\n\n"
        f"**Visual Action:** To determine the precise value or relationship, I will draw an auxiliary line at the target position. "
        f"The calculated pixel coordinates for this reference line are {coords}. "
        f"I will now execute the tool to project this visual aid.\n"
        f"</thinking>"
    )
    
    tool_call = {
        "name": "draw_line",
        "arguments": {
            "coords": coords,
            "image_index": 1 # 通常 1 代表第一张图
        }
    }
    
    return cot_thinking, json.dumps(tool_call)

def convert_to_sft():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        raw_data = json.load(f)

    # 修改为写入 jsonl
    with open(OUTPUT_FILE, 'w') as f_out:
        for item in raw_data:
            # 1. 获取基础信息
            user_input_full = item['conversations'][0]['value'] 
            gpt_response = item['conversations'][1]['value']
            
            # 分离出纯文本问题 (去掉 <image>\n) 用于 CoT 生成
            question_text = user_input_full.replace("<image>\n", "").strip()
            
            # 提取答案字母
            final_answer = extract_answer_letter(gpt_response)
            
            # 生成 CoT 和 Tool Call
            thinking_part, tool_json = generate_cot_and_tool(item, question_text)
            
            # 2. 构建 Messages
            messages = [
                # System Message
                {
                    "role": "system", 
                    "content": SYSTEM_PROMPT
                },
                # User Question
                {
                    "role": "user", 
                    "content": user_input_full
                },
                # Assistant Step 1: Thinking + Tool Call
                {
                    "role": "assistant", 
                    "content": f"{thinking_part}\n<tool_call>\n{tool_json}\n</tool_call>"
                },
                # User Tool Response (Simulated)
                {
                    "role": "user", 
                    "content": "<tool_response>\nAuxiliary line drawn successfully.</tool_response>"
                },
                # Assistant Step 2: Final Conclusion
                {
                    "role": "assistant", 
                    "content": (
                        f"<thinking>\n"
                        f"**Conclusion:** The visual evidence, enhanced by the auxiliary line, clearly points to the answer. "
                        f"Checking the options provided, the value corresponds to option {final_answer}.\n"
                        f"</thinking>\n"
                        f"<answer>\n{final_answer}\n</answer>"
                    )
                }
            ]

            # 3. 构建 Image Path (使用第一张无红线的原图)
            img_filename = item['image'][0] 
            img_abs_path = os.path.join(WORKSPACE_ROOT, "dataset_output_v1", "images", img_filename)

            # 4. 组合最终 Entry
            entry = {
                "messages": messages,
                "images": [img_abs_path]
            }
            
            f_out.write(json.dumps(entry) + '\n')

    print(f"Successfully converted {len(raw_data)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_to_sft()
