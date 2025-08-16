import json
import requests
import time
import os
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import jieba
from collections import Counter
import re
from math import radians, cos, sin, asin, sqrt
from openai import OpenAI

# 百度地图 API Key
AK = "gehmkw7PwZ0TCcHIqHsqa2IoAwWDQKbI"

class TravelAssistant:
    def __init__(self, 
                 model_path="../models/Qwen/Qwen2.5-7B-Instruct",
                 use_test_mode=False,
                 api_key="sk-8uZDRBjjuuwpdArZWdCpo7EP2iKhXBHBnvS5x2ajUbWr8u6t",
                 api_model="gpt-4o",
                 api_base_url="https://api.wlai.vip/v1/"):
        """
        初始化旅游助手
        
        Args:
            model_path: 本地模型路径
            use_test_mode: 是否使用测试模式（True=使用闭源API，False=使用开源模型）
            api_key: API密钥
            api_model: API模型名称
            api_base_url: API基础URL
        """
        self.use_test_mode = use_test_mode
        
        if self.use_test_mode:
            # 测试模式：使用闭源API
            print("启用测试模式：使用闭源模型API")
            self.api_key = api_key
            self.api_model = api_model
            self.api_base_url = api_base_url
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url
            )
            self.model = None
            self.tokenizer = None
        else:
            # 非测试模式：使用开源模型
            print("使用开源模型")
            print(f"正在加载模型: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.client = None
        
        # 初始化结果记录
        self.results = []
    
    def get_openai_response(self, prompt: str, system_prompt: str = None) -> str:
        """使用OpenAI API获取响应"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.api_model,
                temperature=0.7,
                max_tokens=512
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"API调用错误: {str(e)}"
    
    def get_local_model_response(self, prompt: str, system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") -> str:
        """使用本地模型获取响应"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
        
    def llm_generate(self, prompt: str, system_prompt: str = "") -> str:
        """调用LLM生成响应（根据test_mode选择模型）"""
        if self.use_test_mode:
            return self.get_openai_response(prompt, system_prompt)
        else:
            return self.get_local_model_response(prompt, system_prompt)
    
    def classify_query(self, query: str) -> Tuple[bool, bool]:
        """判断query属于哪个类别"""
        prompt = f"""你是一个旅游专家，现在用户有一个旅游相关的问题，请你判断这个问题属于以下哪个类别：静态攻略，地图上规划。
静态攻略指的是：攻略推文、美食酒店调研（点评网站）
地图上规划指的是：需要地图信息（例如离当前最近的目的地，距离景点最近的酒店，如何从当前位置到目的地）

用户问题：{query}

请直接回答"静态攻略"或"地图上规划"或"静态攻略, 地图上规划"。"""
        
        response = self.llm_generate(prompt)
        
        # 记录步骤
        self.results.append({
            "step": "query_classification",
            "input": prompt,
            "output": response,
            "model_used": "API" if self.use_test_mode else "Local"
        })
        
        print(f"[步骤1 - Query分类]\n输入: {query}\n输出: {response}\n")
        
        # 解析响应
        use_rag = "静态攻略" in response
        use_map = "地图上规划" in response
        
        return use_rag, use_map
    
    def bm25_search(self, query: str, docs: List[str], top_k: int = 8) -> List[tuple]:
        """使用BM25算法搜索相关文档，返回(文档, 分数)的列表"""
        # 分词
        query_tokens = list(jieba.cut(query))
        
        # 计算每个文档的BM25分数
        docs_with_scores = []
        for doc in docs:
            doc_tokens = list(jieba.cut(doc))
            score = 0
            for token in query_tokens:
                if token in doc_tokens:
                    score += doc_tokens.count(token)
            
            if score > 0:  # 只保留有相关性的文档
                docs_with_scores.append((doc, score))
        
        # 按分数排序
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return docs_with_scores[:top_k]
    
    def retrieve_from_rag(self, query: str) -> List[str]:
        """从RAG文档中检索相关内容"""
        all_docs_with_scores = []  # 存储所有文档和分数
        files_to_search = []
        
        # 确定要搜索的文件
        if any(keyword in query for keyword in ["攻略", "去哪玩", "景点", "游玩"]):
            files_to_search.append("docs/xhs_rag.txt")
        
        if any(keyword in query for keyword in ["住在哪", "酒店", "宾馆", "民宿", "青旅", "住宿"]):
            files_to_search.append("docs/酒店_rag.txt")
            if "docs/xhs_rag.txt" not in files_to_search:
                files_to_search.append("docs/xhs_rag.txt")
        
        if any(keyword in query for keyword in ["美食", "吃", "夜宵", "菜", "料", "餐厅", "饭店"]):
            files_to_search.append("docs/美食_rag.txt")
            if "docs/xhs_rag.txt" not in files_to_search:
                files_to_search.append("docs/xhs_rag.txt")
        
        # 如果没有匹配到特定类别，默认搜索所有文件
        if not files_to_search:
            files_to_search = ["docs/xhs_rag.txt", "docs/酒店_rag.txt", "docs/美食_rag.txt"]
        
        print(f"[RAG检索] 搜索文件: {files_to_search}")
        
        # 从每个文件中检索
        for file_path in files_to_search:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        docs = f.readlines()
                    docs = [doc.strip() for doc in docs if doc.strip()]
                    
                    # 使用BM25检索，获取文档和分数
                    docs_with_scores = self.bm25_search(query, docs, top_k=10)  # 每个文件多取一些
                    all_docs_with_scores.extend(docs_with_scores)
                else:
                    print(f"警告: 文件 {file_path} 不存在")
            except Exception as e:
                print(f"读取文件 {file_path} 出错: {e}")
        
        # 按分数排序所有文档
        all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 限制总数为8个，只保留文档内容
        retrieved_docs = [doc for doc, score in all_docs_with_scores[:8]]
        
        # 记录步骤
        self.results.append({
            "step": "rag_retrieval",
            "input": {"query": query, "files_searched": files_to_search},
            "output": retrieved_docs
        })
        
        print(f"[步骤2 - RAG检索]\n检索到 {len(retrieved_docs)} 个相关文档\n")
        
        return retrieved_docs
    
    def classify_map_task(self, query: str) -> str:
        """判断地图任务类型"""
        prompt = f"""你是一个旅游专家，现在用户有一个旅游相关的问题，这个问题需要请求地图。
若你认为这个任务是获得多个推荐的目的地，输出"目的地推荐"
若你认为这个任务是获得到达目的地的路线，输出"路线推荐"

用户问题：{query}

请直接回答"目的地推荐"或"路线推荐"。"""
        
        response = self.llm_generate(prompt)
        
        # 记录步骤
        self.results.append({
            "step": "map_task_classification",
            "input": prompt,
            "output": response,
            "model_used": "API" if self.use_test_mode else "Local"
        })
        
        print(f"[步骤2 - 地图任务分类]\n输出: {response}\n")
        
        return response
    
    def geocode_address(self, address: str) -> Dict:
        """根据地址获取经纬度"""
        url = "https://api.map.baidu.com/geocoding/v3/"
        params = {"address": address, "ak": AK, "output": "json"}
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == 0:
                location = data["result"]["location"]
                return {"lng": location["lng"], "lat": location["lat"]}
            else:
                return {"error": data.get("msg", "未知错误")}
        except Exception as e:
            return {"error": str(e)}
    
    def haversine(self, lat1, lng1, lat2, lng2):
        """计算两点直线距离（米）"""
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlng/2)**2
        c = 2*asin(sqrt(a))
        return c * 6371000
    
    def fetch_poi_recommendations(self, query: str) -> Dict:
        """获取POI推荐"""
        # 使用LLM提取位置和目标信息
        prompt = f"""你是一个旅游专家，现在用户有一个旅游相关的问题，这个问题需要请求地图，获取某个地点周边的某些内容。
    你需要从问题中分析用户想要搜索的位置，和想要搜索的内容（例如周边的美食/宾馆/商店），并返回json格式的输出: {{"origin_address": "", "target": ""}}，其中"origin_address"存储想要搜索的位置，"target"存储想要搜索的内容。如果你提取不到想要搜索的位置，在origin_address中填写-1。如果你提取不到想要搜索的内容，在target中填写-1。

    用户问题：{query}

    请直接返回json格式的输出 {{"origin_address": "", "target": ""}}。"""
        
        response = self.llm_generate(prompt)
        
        # 尝试解析JSON
        try:
            import json
            llm_result = json.loads(response.strip())
            origin_address = llm_result.get("origin_address", "")
            target = llm_result.get("target", "")
        except (json.JSONDecodeError, Exception) as e:
            return {"error": "请重新组织指令"}
        
        # 检查提取结果
        if origin_address == "-1":
            return {"error": "您好！请说明您的当前位置或想要查询的位置"}
        
        if target == "-1":
            return {"error": "请重新组织指令"}
        
        # 如果是当前位置，使用默认位置（这里假设为百度科技园）
        if origin_address in ["当前位置", "我的位置", "这里"]:
            origin_address = "百度科技园"
        
        # 获取原点坐标
        origin = self.geocode_address(origin_address)
        if "error" in origin:
            return {"error": f"地理编码失败: {origin}"}
        
        # 处理多个目标，用空格分隔
        targets = target.split()
        poi_queries = []
        for t in targets:
            poi_queries.append({"query": t.strip(), "tag": ""})
        
        # 搜索POI
        url = "http://api.map.baidu.com/place/v2/search"
        all_poi_data = []
        
        for pq in poi_queries:
            params = {
                "ak": AK,
                "query": pq["query"],
                "tag": pq["tag"],
                "location": f"{origin['lat']},{origin['lng']}",
                "radius": 1000,
                "output": "json",
                "page_size": 10  # 增加每个类别的搜索数量
            }
            
            try:
                resp = requests.get(url, params=params)
                data = resp.json()
                if data.get("status") == 0:
                    results = data.get("results", [])
                    for poi in results:
                        poi_lat = poi["location"]["lat"]
                        poi_lng = poi["location"]["lng"]
                        distance = self.haversine(origin["lat"], origin["lng"], poi_lat, poi_lng)
                        
                        # 提取必要信息
                        poi_info = {
                            "name": poi.get("name", ""),
                            "address": poi.get("address", ""),
                            "telephone": poi.get("telephone", ""),
                            "distance_m": round(distance, 2),
                            "category": pq["query"]  # 添加类别信息
                        }
                        all_poi_data.append(poi_info)
            except Exception as e:
                print(f"POI搜索出错: {e}")
        
        # 如果有多个POI类别，确保每个类别都有推荐
        if len(poi_queries) > 1:
            # 按类别分组
            category_groups = {}
            for poi in all_poi_data:
                category = poi["category"]
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append(poi)
            
            # 从每个类别中选取推荐，确保多样性
            balanced_recommendations = []
            max_per_category = max(1, 8 // len(poi_queries))  # 每个类别最多取的数量
            
            for category, pois in category_groups.items():
                # 按距离排序，取前几个
                pois.sort(key=lambda x: x["distance_m"])
                balanced_recommendations.extend(pois[:max_per_category])
            
            # 按距离重新排序
            balanced_recommendations.sort(key=lambda x: x["distance_m"])
            final_recommendations = balanced_recommendations[:8]
        else:
            # 单一类别，按距离排序取前5个
            all_poi_data.sort(key=lambda x: x["distance_m"])
            final_recommendations = all_poi_data[:5]
        
        return {
            "origin_address": origin_address,
            "target_categories": [pq["query"] for pq in poi_queries],
            "recommendations": final_recommendations
        }
    
    def get_route_recommendations(self, query: str) -> Dict:
        """获取路线推荐"""
        # 使用LLM提取起始地点和目标地点
        prompt = f"""你是一个旅游专家，现在用户有一个旅游相关的问题，这个问题需要请求地图，获取某个起始地点的到目标地点的路线。
    你需要从问题中分析用户的起始地点，和目标地点，并返回json格式的输出: {{"origin": "", "destination": ""}}，其中"origin"存储起始地点，"destination"存储目标地点。如果你提取不到起始地点，在origin中填写-1。如果你提取不到目标地点，在destination中填写-1。如果起始地点很模糊，例如"我的位置"，"当前地点"，"这里"，也认为提取不到起始地点，在origin中填写-1。

    用户问题：{query}

    请直接返回json格式的输出 {{"origin": "", "destination": ""}}。"""
        
        response = self.llm_generate(prompt)
        
        # 尝试解析JSON
        try:
            import json
            llm_result = json.loads(response.strip())
            origin = llm_result.get("origin", "")
            destination = llm_result.get("destination", "")
        except (json.JSONDecodeError, Exception) as e:
            return {"error": "请重新组织指令"}
        
        # 检查提取结果
        if origin == "-1" or destination == "-1":
            return {"error": "您好！请说明您的起始地点和目标地点"}
        
        # 如果是当前位置，使用默认位置（这里假设为百度科技园）
        if origin in ["当前位置", "我的位置", "这里"]:
            return {"error": "您好！请说明您的起始地点和目标地点"}
        
        # 获取坐标
        origin_coords = self.geocode_address(origin)
        dest_coords = self.geocode_address(destination)
        
        if "error" in origin_coords or "error" in dest_coords:
            return {"error": "地理编码失败"}
        
        # 获取多种交通方式的路线
        modes = ["driving", "transit", "riding", "walking"]
        mode_names = {"driving": "驾车", "transit": "公交", "riding": "骑行", "walking": "步行"}
        routes = []
        
        for mode in modes:
            base_url = f"https://api.map.baidu.com/directionlite/v1/{mode}"
            params = {
                "ak": AK,
                "origin": f"{origin_coords['lat']},{origin_coords['lng']}",
                "destination": f"{dest_coords['lat']},{dest_coords['lng']}"
            }
            
            try:
                resp = requests.get(base_url, params=params)
                data = resp.json()
                
                if data.get("status") == 0:
                    route = data["result"]["routes"][0]
                    distance_m = route.get("distance", 0)
                    duration_s = route.get("duration", 0)
                    
                    # 保留完整的路线描述
                    steps = route.get("steps", [])
                    all_steps = []
                    for step in steps:
                        if isinstance(step, dict) and "instruction" in step:
                            all_steps.append(step["instruction"])
                    
                    route_info = {
                        "mode": mode_names[mode],
                        "distance_km": round(distance_m / 1000, 2),
                        "duration_min": round(duration_s / 60, 1),
                        "steps": all_steps,  # 保留完整步骤
                        "steps_summary": " → ".join(all_steps[:3]) + ("..." if len(all_steps) > 3 else "") if all_steps else "直达"  # 添加摘要用于显示
                    }
                    routes.append(route_info)
                    
            except Exception as e:
                print(f"获取{mode}路线出错: {e}")
            
            time.sleep(0.1)
        
        return {
            "origin": origin,
            "destination": destination,
            "routes": routes
        }
    
    def handle_map_query(self, query: str) -> Dict:
        """处理地图相关查询"""
        task_type = self.classify_map_task(query)
        
        if "路线推荐" in task_type:
            result = self.get_route_recommendations(query)
        elif "目的地推荐" in task_type:
            result = self.fetch_poi_recommendations(query)
        else:
            # 默认为目的地推荐
            result = self.fetch_poi_recommendations(query)
        
        # 记录步骤
        self.results.append({
            "step": "map_api_call",
            "input": {"query": query, "task_type": task_type},
            "output": result
        })
        
        print(f"[步骤3 - 地图API调用]\n获取到结果\n")
        
        return result
    
    def generate_final_response(self, query: str, rag_docs: List[str] = None, map_info: Dict = None) -> str:
        """生成最终响应"""
        # 构建prompt
        system_prompt = "你是一个专业的旅游助手，请根据提供的信息为用户提供详细、有用的旅游建议。"
        
        prompt = f"{system_prompt}\n\n用户问题：{query}\n\n"
        
        if rag_docs:
            prompt += "相关文档：\n"
            for i, doc in enumerate(rag_docs):
                prompt += f"{i+1}. {doc}\n"
            prompt += "\n"
        
        if map_info:
            prompt += f"从地图上获得的信息：{json.dumps(map_info, ensure_ascii=False, indent=2)}\n\n"
        
        prompt += "请根据以上信息，为用户提供详细的回答。如果用于回答问题的信息不够充分，请先温和地提示用户补充输入，然后再根据自身的知识给出一个答案，并在答案结尾声明\"以上回答截止至模型训练时所获取的知识，如果需要更靠谱的知识，请用更具体的提问方式问我吧～\""
        
        response = self.llm_generate(prompt)
        
        # 记录步骤
        self.results.append({
            "step": "final_response_generation",
            "input": prompt,
            "output": response,
            "model_used": "API" if self.use_test_mode else "Local"
        })
        
        print(f"[步骤4 - 最终响应生成]\n{response}\n")
        
        return response
    
    def process_query(self, query: str) -> str:
        """处理用户查询的主函数"""
        print(f"\n{'='*50}")
        print(f"处理查询: {query}")
        print(f"模型模式: {'API测试模式' if self.use_test_mode else '本地模型模式'}")
        print(f"{'='*50}\n")
        
        # 步骤1：分类查询
        use_rag, use_map = self.classify_query(query)
        
        rag_docs = None
        map_info = None
        
        # 步骤2：根据分类执行相应操作
        if use_rag:
            rag_docs = self.retrieve_from_rag(query)
        
        if use_map:
            map_info = self.handle_map_query(query)
        
        # 步骤3：生成最终响应
        final_response = self.generate_final_response(query, rag_docs, map_info)
        
        # 保存结果
        self.save_results()
        
        return final_response
    
    def save_results(self):
        """保存所有步骤的结果"""
        filename = f"results_{'api' if self.use_test_mode else 'local'}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到 {filename}")


# 主函数
def main():  
    # 配置参数
    USE_TEST_MODE = True  # 修改这个参数来切换模式：True=使用API，False=使用本地模型
    API_KEY = "sk-8uZDRBjjuuwpdArZWdCpo7EP2iKhXBHBnvS5x2ajUbWr8u6t"
    API_MODEL = "gpt-4o"
    API_BASE_URL = "https://api.wlai.vip/v1/"
    LOCAL_MODEL_PATH = "../models/Qwen/Qwen2.5-7B-Instruct"
    
    # 初始化助手
    assistant = TravelAssistant(
        model_path=LOCAL_MODEL_PATH,
        use_test_mode=USE_TEST_MODE,
        api_key=API_KEY,
        api_model=API_MODEL,
        api_base_url=API_BASE_URL
    )
    
    # 测试不同类型的查询
    test_queries = [
    "北京有哪些必去的景点？",
    "从首都机场到天安门最快的路线是什么？",
    "北京最好的烤鸭店",
    "离故宫最近的地铁站",
    "北京三日游攻略推荐",
    "从王府井到长城怎么走？",
    "北京有什么特色小吃",
    "距离西二旗最近的景点",
    "798艺术区有哪些值得看的展览？",
    "三里屯附近有什么好吃的餐厅推荐吗？",
    "中国国家博物馆需要提前预约吗？",
    "什刹海租船多少钱一小时？",
    "八达岭长城最佳观景点在哪里？",
    "前门大街晚上好玩吗？",
    "北京大学校园可以自由参观吗？",
    "北京CBD附近的高档酒店有哪些？",
    "北京东城区有什么特色餐厅？",
    "丰台区周末适合去哪玩？",
    "北京使馆区有哪些异国餐厅？",
    "北京动物园看大熊猫的最佳时间是什么时候？",
    "北京后海晚上有什么活动？",
    "大兴区有哪些亲子游景点？",
    "北京奥林匹克公园夜景好看吗？",
    "密云区秋天适合去哪玩？",
    "延庆区冬天滑雪推荐哪里？",
    "怀柔区哪些地方适合露营？",
    "昌平区除了十三陵还有什么好玩的？",
    "朝阳区哪里有好逛的商场？",
    "首都机场周边有什么酒店推荐？",
    "北京欢乐谷门票多少钱？",
    "海淀区有什么好吃的餐馆？",
    "石景山区有什么特色景点？",
    "北京老城区适合citywalk的路线有哪些？",
    "北京胡同区适合拍照的地方有哪些？",
    "西城区周末适合去哪？",
    "通州区有什么好玩的地方？",
    "北京郊区自驾一日游路线推荐",
    "顺义区有哪些采摘园？",
    "十三陵景区需要预约吗？",
    "北京环球影城有哪些热门项目？",
    "北海公园荷花季什么时候？",
    "南锣鼓巷附近有什么特色小吃？",
    "国家大剧院有什么演出推荐？",
    "圆明园的最佳游览路线是什么？",
    "天坛公园适合晨练吗？",
    "天安门广场升旗仪式几点开始？",
    "奥体中心附近有什么酒店？",
    "慕田峪长城人多吗？",
    "景山公园看故宫全景的机位在哪？",
    "水立方晚上会亮灯吗？",
    "王府井步行街营业到几点？",
    "簋街晚上几点最热闹？",
    "长城秋天去合适吗？",
    "雍和宫求签灵吗？",
    "颐和园划船多少钱？",
    "鸟巢可以进去参观吗？",
    "清华大学需要预约参观吗？",
    "中国人民大学附近有什么餐馆？",
    "北京航空航天大学校园景色怎么样？",
    "北京师范大学的图书馆可以外借吗？",
    "北京理工大学周边有什么好吃的？",
    "中国农业大学周末开放吗？",
    "中央民族大学的民族文化展在哪里？",
    "北京交通大学附近的美食街叫什么？",
    "北京科技大学有什么特色活动？",
    "北京外国语大学的语言博物馆对外开放吗？",
    "中国政法大学附近的地铁站是哪个？",
    "798艺术区附近的咖啡厅推荐",
    "三里屯夜生活怎么玩？",
    "中国国家博物馆展览时间表",
    "什刹海冰场什么时候开放？",
    "八达岭长城缆车票价多少？",
    "前门大街有哪些老字号？",
    "北京大学附近的书店推荐",
    "北京CBD拍夜景的好地方",
    "北京东城区胡同游推荐",
    "丰台区周末市集在哪？",
    "北京使馆区最美的街道是哪条？",
    "北京动物园门票价格",
    "北京后海酒吧街怎么样？",
    "大兴区适合带孩子的餐厅",
    "北京奥林匹克公园适合跑步吗？",
    "密云区水库可以钓鱼吗？",
    "延庆区夏天避暑推荐",
    "怀柔区山里民宿推荐",
    "昌平区地铁沿线景点",
    "朝阳区哪里有好玩的亲子乐园？",
    "首都机场到市区的地铁路线",
    "北京欢乐谷适合情侣玩吗？",
    "海淀区适合学习的咖啡馆",
    "石景山区爬山的地方",
    "北京老城区的摄影机位",
    "北京胡同区的美食地图",
    "西城区最好的早餐店",
    "通州区运河公园好玩吗？",
    "北京郊区露营地推荐",
    "顺义区周末自驾路线",
    "十三陵附近的餐厅推荐",
    "北京环球影城门票价格",
    "北海公园冬天好玩吗？",
    "南锣鼓巷人多吗？",
    "国家大剧院内部参观需要票吗？",
    "圆明园拍照最佳季节",
    "天坛拍日出的好位置",
    "天安门广场附近的酒店推荐",
    "奥体中心看比赛的最佳位置",
    "慕田峪长城徒步路线",
    "景山公园樱花开花时间",
    "水立方内部有什么？",
    "王府井可以买到哪些特产？",
    "簋街有素食餐厅吗？",
    "长城夜游项目有吗？",
    "雍和宫附近的茶馆推荐",
    "颐和园冬天有活动吗？",
    "鸟巢举办过哪些大赛？",
    "清华大学周边住宿推荐",
    "中国人民大学校园参观预约",
    "北京航空航天大学博物馆开放时间",
    "北京师范大学附近的早餐店",
    "北京理工大学的校园特色",
    "中国农业大学的农场开放吗？",
    "中央民族大学美食推荐",
    "北京交通大学周边住宿",
    "北京科技大学图书馆开放时间",
    "北京外国语大学校园地图",
    "中国政法大学的历史介绍",
    "798艺术区最受欢迎的展馆",
    "三里屯的最佳拍照点",
    "中国国家博物馆镇馆之宝有哪些？",
    "什刹海夏天怎么玩？",
    "八达岭长城最佳拍摄时间",
    "前门大街交通攻略",
    "北京大学的历史建筑有哪些？",
    "北京CBD停车方便吗？",
    "北京东城区的文化活动",
    "丰台区最火的餐厅",
    "北京使馆区的咖啡厅推荐",
    "北京动物园儿童票价",
    "北京后海白天好玩吗？",
    "大兴区的历史景点有哪些？",
    "北京奥林匹克公园拍照机位",
    "密云区的红叶观赏点",
    "延庆区的农家乐推荐",
    "怀柔区的漂流地点",
    "昌平区的温泉酒店",
    "朝阳区的艺术展览",
    "首都机场到北京南站怎么走？",
    "北京欢乐谷夜场活动",
    "海淀区的二手书店",
    "石景山区的特色美食",
    "北京老城区的手工艺品店",
    "北京胡同区的文化体验",
    "西城区的老字号餐厅",
    "通州区的夜景拍摄点",
    "北京郊区的滑雪场",
    "顺义区的温泉度假村",
    "十三陵的开放时间",
    "北京环球影城表演时间",
    "北海公园划船价格",
    "南锣鼓巷的文创店",
    "国家大剧院的建筑亮点",
    "圆明园的历史故事",
    "天坛的祭天大典介绍",
    "天安门广场看升旗攻略",
    "奥体中心的演唱会场地",
    "慕田峪长城索道信息",
    "景山公园门票价格",
    "水立方举办过的活动",
    "王府井的美食推荐",
    "簋街的营业时间",
    "长城的旅游季节",
    "雍和宫的开放时间",
    "颐和园的游船路线",
    "鸟巢的开放时间",
    "清华大学的著名教授",
    "中国人民大学的优势学科",
    "北京航空航天大学的专业特色",
    "北京师范大学的校园文化",
    "北京理工大学的科研成果",
    "中国农业大学的植物园",
    "中央民族大学的民族节日",
    "北京交通大学的历史沿革",
    "北京科技大学的校园环境",
    "北京外国语大学的外籍学生情况",
    "中国政法大学的著名校友",
    "从北京南站到颐和园坐地铁怎么走？",
    "从天安门广场到北京欢乐谷需要多长时间？",
    "故宫附近最近的停车场在哪里？",
    "从王府井到八达岭长城的自驾路线怎么走？",
    "长城脚下的民宿有哪些推荐？",
    "从北京首都机场到三里屯最快的地铁路线是什么？",
    "北京西站到鸟巢的公交路线有哪些？",
    "颐和园附近有哪些高评分餐馆？",
    "地铁4号线沿线有哪些景点值得一去？",
    "北京站到中国国家博物馆步行要多久？",
    "北京动物园附近的酒店有哪些？",
    "西单到圆明园的换乘方案是什么？",
    "地铁2号线可以直达哪些热门旅游景点？",
    "从国贸到798艺术区最好怎么走？",
    "北京地铁哪几条线可以到达天坛公园？",
    "北京南站附近的商务酒店有哪些？",
    "从北京西站到香山公园的公交路线推荐",
    "南锣鼓巷附近的地铁站是哪个？",
    "北京地铁1号线沿线的景点有哪些？",
    "从望京到奥林匹克公园骑行需要多久？",
    "北京环球影城附近的住宿推荐",
    "地铁5号线沿线有哪些著名景点？",
    "从首都机场到北海公园打车需要多久？",
    "什刹海附近的咖啡厅集中在哪些街道？",
    "北京站到雍和宫的地铁换乘方式",
    "地铁10号线沿线的美食街推荐",
    "北京欢乐谷附近的地铁站是哪个？",
    "北京西站到天安门广场的最佳出行方式",
    "鸟巢附近的停车场分布在哪里？",
    "朝阳公园周边的餐饮选择有哪些？",
    "北京地铁14号线经过哪些景点？",
    "奥体中心附近的快捷酒店推荐",
    "从北京南站到密云水库的自驾路线",
    "地铁6号线沿线的文化景点有哪些？",
    "北京动物园到北京植物园的公交路线",
    "从前门到颐和园的地铁乘坐方案",
    "798艺术区附近的精品酒店",
    "北京地铁8号线沿线的旅游景点",
    "天坛公园周边的早餐店推荐",
    "北京西站到长城的旅游巴士在哪里乘坐？",
    "地铁7号线沿线的特色餐厅",
    "北京站到香山公园的公交换乘方案",
    "北京CBD附近的地铁站分布",
    "从首都机场到五道口的最快路线",
    "北海公园周边的宾馆推荐",
    "北京地铁13号线沿线的景点",
    "地铁4号线从动物园到清华大学的时间",
    "北京西单到三里屯的地铁路线",
    "从北京南站到延庆滑雪场怎么走？",
    "北京站到天坛的步行路线",
    "北京地铁昌平线沿线好玩的地方",
    "从国贸到故宫最佳出行方式",
    "地铁1号线经过的购物中心有哪些？",
    "北京南站附近的快餐店推荐",
    "朝阳区的地铁站旁有哪些景点？",
    "北京西站到颐和园的骑行路线",
    "798艺术区到国家大剧院的地铁换乘",
    "北京站到大兴机场的最快交通方式",
    "天安门附近的连锁酒店",
    "地铁5号线从雍和宫到宋家庄的景点分布",
    "北京地铁房山线沿线的旅游资源",
    "从首都机场到八达岭长城的高铁路线",
    "北京西站到南锣鼓巷的地铁路线",
    "怀柔雁栖湖附近的住宿选择",
    "北京站到王府井的公交线路",
    "北京南站到水立方的最佳路线",
    "北京地铁2号线沿线的老城区景点",
    "鸟巢到颐和园的地铁方案",
    "北京地铁15号线沿线的景点",
    "北京西站到十三陵的旅游巴士",
    "环球影城到天安门的出行方案",
    "地铁1号线从四惠到苹果园的沿途景点",
    "北京站到北京欢乐谷的最佳方式",
    "王府井附近的地铁站出口分布",
    "北京南站到圆明园的地铁方案",
    "北京地铁昌平线到八达岭的换乘方式",
    "798艺术区到北京动物园的出行路线",
    "北京地铁10号线环线的主要景点",
    "地铁14号线到环球影城的路线",
    "北京西站到北海公园的地铁方案",
    "从国贸到香山公园的交通方式",
    "北京站到鸟巢的最快路线",
    "地铁5号线沿线的美食街区",
    "北京南站到南锣鼓巷的地铁方案",
    "北京地铁6号线沿线的博物馆",
    "地铁8号线到奥林匹克公园的出口",
    "北京西站到798艺术区的公交方案",
    "首都机场到王府井的最佳路线",
    "北京站到颐和园的地铁方案",
    "北京南站到故宫的最快方式",
    "地铁13号线沿线的餐饮推荐",
    "北京地铁亦庄线沿线的景点",
    "北京西站到天坛的地铁方案",
    "北京站到北海公园的出行方式",
    "地铁房山线到北京野生动物园的路线",
    "北京南站到环球影城的地铁方案",
    "北京地铁8号线沿线的餐厅推荐",
    "地铁4号线到北京动物园的出口",
    "北京西站到五道口的地铁路线",
    "798艺术区到三里屯的出行方式",
    "北京地铁10号线沿线的购物中心",
    "北京南站到八达岭长城的交通方式",
    "北京站到雍和宫的最快路线"
]
    
    for query in test_queries:
        response = assistant.process_query(query)
        print(f"\n最终回答：{response}\n")
        print("="*80)
        
        # 每次查询后清空结果，为下一次查询准备
        assistant.results = []


if __name__ == "__main__":
    main()