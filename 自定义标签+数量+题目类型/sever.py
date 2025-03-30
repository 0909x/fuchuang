from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import asyncio
from generate import process_qa_generation, load_config
import sys  # 添加这行到文件顶部其他import语句旁边

class APIHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/generate':
            print("\n收到新的生成请求...")  # 调试日志
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data)

                print("接收到的请求数据:", request_data)  # 调试日志

                config = load_config()
                print("加载的初始配置:", {k: v for k, v in config.items() if k not in ['api_key']})  # 调试日志

                # 更新配置
                if 'types' in request_data:
                    config['QUESTION_TYPES'] = {t: 1.0 / len(request_data['types']) for t in request_data['types']}
                if 'difficulty' in request_data:
                    config['DIFFICULTY_LEVELS'] = {request_data['difficulty']: 1.0}

                # 处理主题
                if request_data.get('is_custom_topic', False):
                    custom_topic = request_data.get('topic', '').strip()
                    print(f"使用自定义主题: {custom_topic}")  # 调试日志
                    if custom_topic:
                        config['ACTIVE_TOPICS'] = [custom_topic]
                elif 'topic' in request_data and request_data['topic']:
                    print(f"使用预设主题: {request_data['topic']}")  # 调试日志
                    config['ACTIVE_TOPICS'] = [request_data['topic']]

                config['TARGET_QA_COUNT'] = max(request_data.get('count', 5), 3)
                config['QA_PER_REQUEST'] = min(5, config['TARGET_QA_COUNT'])

                print("更新后的配置:", {k: v for k, v in config.items() if k not in ['api_key']})  # 调试日志

                # 生成题目
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                qa_pairs, _, _ = loop.run_until_complete(process_qa_generation(config))
                loop.close()

                print(f"成功生成 {len(qa_pairs)} 道题目")  # 调试日志
                response = {
                    "status": "success",
                    "count": len(qa_pairs),
                    "questions": qa_pairs
                }
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                print(f"生成题目时出错: {str(e)}", file=sys.stderr)  # 调试日志
                error_response = {
                    "status": "error",
                    "message": str(e)
                }
                self.wfile.write(json.dumps(error_response).encode())


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8000), APIHandler)
    print("题库生成服务已启动")
    print("支持端点: POST /generate")
    server.serve_forever()