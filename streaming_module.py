import json
import time
from typing import List, Dict, Any, Tuple, Generator
from turing_module import TravelAssistant

class StreamingTravelAssistant(TravelAssistant):
    """支持流式处理的旅游助手"""
    
    def process_query_streaming(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """流式处理用户查询"""
        print(f"\n{'='*50}")
        print(f"流式处理查询: {query}")
        print(f"模型模式: {'API测试模式' if self.use_test_mode else '本地模型模式'}")
        print(f"{'='*50}\n")
        
        # 步骤1：分类查询
        use_rag, use_map = self.classify_query(query)
        yield {"type": "status", "content": "query_classified", "use_rag": use_rag, "use_map": use_map}
        
        rag_docs = None
        rag_sources = None
        map_info = None
        
        # 步骤2：根据分类执行相应操作
        if use_rag:
            yield {"type": "status", "content": "rag_retrieval_start"}
            rag_docs, rag_sources = self.retrieve_from_rag(query)
            yield {"type": "rag_info", "content": "rag_retrieved", "doc_count": len(rag_docs)}
        
        if use_map:
            yield {"type": "status", "content": "map_processing_start"}
            map_info = self.handle_map_query(query)
            yield {"type": "map_info", "content": "map_processed"}
        
        # 步骤3：生成最终响应（流式）
        yield {"type": "status", "content": "response_generation_start"}
        
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
        
        prompt += "注意回答需要以markdown格式返回。请根据以上RAG信息，为用户提供详细的回答。你应该优先以相关文档或从地图上获得的信息作为参考来回答，但请注意，不要出现\"根据提供的信息\"，\"根据以上信息\"这样的字样，也不需要评价以上RAG信息是否详细、是否缺失，不需要让用户知道这些信息来源于RAG。对于酒店和饮食类的问题，如果RAG信息中包含评分和评论，应该在你的回答中明确的给出和总结，例如xx饭店，评分4.2，饱受顾客好评，xx评价它是北京最好的烤鸭店，但也有一些人说该饭店的服务不太好。你是一个专注于帮助解决用户旅游问题的贴心的agent，回答应该表现得亲切。如果上述提供的用于回答问题信息不够充分，请先温和地提示用户补充输入，然后再根据自身的知识给出一个答案。如果用于回答问题的信息不够充分，需要在答案结尾声明\"以上回答截止至模型训练时所获取的知识，如果需要更靠谱的知识，请用更具体的提问方式问我吧～\"。但如果相关文档和从地图上的信息是充分的，则不应该加上这个声明。"
        
        # 流式生成响应
        if self.use_test_mode:
            # 使用OpenAI API的流式响应
            yield from self._get_openai_response_streaming(prompt, system_prompt)
        else:
            # 使用本地模型的流式响应
            response = self.get_local_model_response(prompt, system_prompt)
            for char in response:
                yield {"type": "token", "content": char}
                time.sleep(0.01)  # 添加小延迟以模拟流式效果
        
        # 结束信号
        yield {"type": "end", "content": "complete", "rag_docs": rag_docs, "rag_sources": rag_sources, "map_info": map_info}
    
    def _get_openai_response_streaming(self, prompt: str, system_prompt: str = None) -> Generator[Dict[str, Any], None, None]:
        """使用OpenAI API获取流式响应"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            print(f"正在调用OpenAI API，模型: {self.api_model}")
            print(f"消息长度: {len(str(messages))} 字符")
            
            # 使用流式API
            stream = self.client.chat.completions.create(
                messages=messages,
                model=self.api_model,
                temperature=0.7,
                max_tokens=1024,
                stream=True
            )
            
            print("OpenAI API流式响应开始")
            for chunk in stream:
                # 检查chunk.choices是否存在且不为空
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                    if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        yield {"type": "token", "content": content}
                else:
                    print(f"警告: 接收到空的chunk.choices: {chunk}")
                    
        except Exception as e:
            print(f"OpenAI API流式调用错误: {str(e)}")
            error_msg = f"抱歉，AI服务暂时不可用，请稍后再试。错误信息: {str(e)}"
            yield {"type": "token", "content": error_msg}


def process_single_query_streaming(
    query: str,
    use_test_mode: bool = False,
    api_key: str = "sk-8uZDRBjjuuwpdArZWdCpo7EP2iKhXBHBnvS5x2ajUbWr8u6t",
    api_model: str = "gpt-4.1-2025-04-14",
    api_base_url: str = "https://api.wlai.vip/v1/",
    local_model_path: str = "/root/paddlejob/workspace/env_run/chuxu/LLaMA-Factory/output/qwen2_5_lora_sft"
) -> Generator[Dict[str, Any], None, None]:
    """
    流式处理单个查询
    
    Returns:
        Generator yielding streaming responses
    """
    
    # Initialize streaming assistant
    assistant = StreamingTravelAssistant(
        model_path=local_model_path,
        use_test_mode=use_test_mode,
        api_key=api_key,
        api_model=api_model,
        api_base_url=api_base_url
    )
    
    # Process query with streaming
    yield from assistant.process_query_streaming(query)
