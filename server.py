from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import sys
import json

# Import the processing functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from streaming_module import process_single_query_streaming

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': '请输入查询内容'}), 400
        
        # 使用真正的流式响应
        def generate():
            # 流式处理查询
            stream = process_single_query_streaming(
                query=query,
                use_test_mode=True,
                api_key="sk-8uZDRBjjuuwpdArZWdCpo7EP2iKhXBHBnvS5x2ajUbWr8u6t",
                api_model="gpt-4.1-2025-04-14",
                api_base_url="https://api.wlai.vip/v1/",
                local_model_path="/root/paddlejob/workspace/env_run/chuxu/LLaMA-Factory/output/qwen2_5_lora_sft"
            )
            
            # 实时传输每个处理步骤
            for response in stream:
                yield json.dumps(response, ensure_ascii=False) + '\n'
        
        return Response(stream_with_context(generate()), content_type='application/json')
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 使用与 http.server 相同的方式
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=False)
