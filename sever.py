from http.server import HTTPServer, BaseHTTPRequestHandler
from generate import process_qa_generation, load_config  # 确保导入load_config
import json
import asyncio


class APIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
            return

        if self.path == '/generate-question':
            self.handle_generate()
        elif self.path == '/':
            self.send_html()
        else:
            self.send_error(404)

    def handle_generate(self):
        try:
            # 获取配置（现在使用正确导入的load_config）
            config = load_config()

            # 保存原始配置
            original_count = config["TARGET_QA_COUNT"]

            # 临时修改为只生成5个问题
            config["TARGET_QA_COUNT"] = 5

            # 调用原有异步函数
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            qa_pairs, _, _ = loop.run_until_complete(process_qa_generation())
            loop.close()

            # 恢复配置
            config["TARGET_QA_COUNT"] = original_count

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "question": qa_pairs[0]["question"],
                "type": qa_pairs[0]["type"]
            }).encode())
        except Exception as e:
            self.send_error(500, f"生成错误: {str(e)}")

    def send_html(self):
        try:
            with open('index.html', 'rb') as f:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f.read())
        except FileNotFoundError:
            self.send_error(404, "File not found")


def run_server():
    port = 8000
    server = HTTPServer(('0.0.0.0', port), APIHandler)
    print(f"服务器已启动: http://localhost:{port}")
    print("按 CTRL+C 停止服务")
    server.serve_forever()


if __name__ == '__main__':
    run_server()