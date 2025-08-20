from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Import the processing function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from turing_module import process_single_query

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
        
        # Process the query - now returns a dictionary with all information
        result = process_single_query(
            query=query,
            use_test_mode=False,
            api_key="sk-8uZDRBjjuuwpdArZWdCpo7EP2iKhXBHBnvS5x2ajUbWr8u6t",
            api_model="gpt-4.1-2025-04-14",
            api_base_url="https://api.wlai.vip/v1/",
            local_model_path="/root/paddlejob/workspace/env_run/chuxu/LLaMA-Factory/output/qwen2_5_lora_sft"
        )
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 使用与 http.server 相同的方式
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=False)