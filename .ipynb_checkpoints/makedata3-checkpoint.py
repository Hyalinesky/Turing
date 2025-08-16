import json
import re
from openai import OpenAI
from tqdm import tqdm
import time

def get_openai_response(prompt, api_key, model_name):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.wlai.vip/v1/"
    )

    content = [{"type": "text", "text": prompt}]

    try:
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": content
            }],
            model=model_name
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def extract_query_from_prompt(prompt):
    """从prompt中提取用户问题"""
    pattern = r"用户问题：(.*?)\n\n"
    match = re.search(pattern, prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def is_valid_json_response(response):
    """检查response是否为有效的JSON格式并包含所需字段"""
    try:
        llm_result = json.loads(response.strip())
        origin_address = llm_result.get("origin_address", "")
        target = llm_result.get("target", "")
        return True, llm_result
    except (json.JSONDecodeError, Exception) as e:
        return False, None

def process_data():
    # 配置参数
    model_name = "gpt-4o"
    api_key = "sk-kxHck7UPIZVBi4TP3d13366b45F24aC1B50a5dB3DaA100Fb"
    
    # 读取原始数据
    input_file = "data/地图任务路由.jsonl"
    output_file = "data/目的地推荐.jsonl"
    
    # 读取所有数据并筛选包含"目的地推荐"的数据
    filtered_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if "目的地推荐" in data.get("response", ""):
                filtered_data.append(data)
    
    print(f"找到 {len(filtered_data)} 条包含'目的地推荐'的数据")
    
    # 处理数据
    processed_count = 0
    skipped_count = 0
    
    # 清空输出文件（如果存在）
    open(output_file, 'w', encoding='utf-8').close()
    
    for data in tqdm(filtered_data, desc="处理数据"):
        # 提取用户问题
        query = extract_query_from_prompt(data["prompt"])
        if not query:
            print(f"无法提取用户问题，跳过: {data['prompt'][:50]}...")
            skipped_count += 1
            continue
        
        # 构建新的prompt
        new_prompt = f"""你是一个旅游专家，现在用户有一个旅游相关的问题，这个问题需要请求地图，获取某个地点周边的某些内容。
    你需要从问题中分析用户想要搜索的位置，和想要搜索的内容（例如周边的美食/宾馆/商店），并返回json格式的输出: {{"origin_address": "", "target": ""}}，其中"origin_address"存储想要搜索的位置，"target"存储想要搜索的内容。如果你提取不到想要搜索的位置，在origin_address中填写-1。如果你提取不到想要搜索的内容，在target中填写-1。

    用户问题：{query}

    请直接返回json格式的输出 {{"origin_address": "", "target": ""}}。"""
        
        # 尝试获取有效response，最多重试3次
        success = False
        for attempt in range(3):
            try:
                response = get_openai_response(new_prompt, api_key, model_name)
                
                if response.startswith("Error:"):
                    print(f"API错误: {response}")
                    time.sleep(1)  # 等待1秒后重试
                    continue
                
                # 检查JSON是否有效
                is_valid, parsed_json = is_valid_json_response(response)
                
                if is_valid:
                    # 构建结果数据
                    result_data = {
                        "system": "",
                        "prompt": new_prompt,
                        "response": response.strip()
                    }
                    
                    # 追加写入文件
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
                    
                    processed_count += 1
                    success = True
                    break
                else:
                    print(f"无效JSON响应 (尝试 {attempt + 1}/3): {response}")
                    time.sleep(0.5)  # 短暂等待后重试
                    
            except Exception as e:
                print(f"处理出错 (尝试 {attempt + 1}/3): {str(e)}")
                time.sleep(1)
        
        if not success:
            print(f"跳过数据，3次尝试均失败: {query[:50]}...")
            skipped_count += 1
        
        # 避免API调用过于频繁
        time.sleep(0.1)
    
    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 条")
    print(f"跳过: {skipped_count} 条")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    process_data()