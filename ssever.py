import logging
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import change
import asyncio
from datetime import datetime
from audit import load_latest_qa_pairs_from_dir, ModelAuditor
from generate import process_qa_generation, load_config
import sys


def setup_server_logging():
    logger = logging.getLogger('server')
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_server_logging()


class APIHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger('server')
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        self.logger.info("%s - - [%s] %s" % (
            self.address_string(),
            self.log_date_time_string(),
            format % args
        ))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/get_report':
            self.handle_get_report()
        elif self.path == '/download_report':
            self.handle_download_report()
        elif self.path == '/download_json_report':
            self.handle_download_json_report()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_get_report(self):
        try:
            report_exists = os.path.exists('report.md') and os.path.exists('report.json')
            response = {
                "status": "success" if report_exists else "error",
                "message": "Report available" if report_exists else "Report not generated",
                "markdown_url": "/download_report",
                "json_url": "/download_json_report"
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            self.send_error(500, str(e))

    def handle_download_report(self):
        try:
            if not os.path.exists('report.md'):
                raise FileNotFoundError("Markdown report not found")

            with open('report.md', 'rb') as f:
                report_content = f.read()

            self.send_response(200)
            self.send_header('Content-type', 'text/markdown')
            self.send_header('Content-Disposition', 'attachment; filename="model_test_report.md"')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(report_content)
        except FileNotFoundError:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "error", "message": "Report not generated"}
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            self.send_error(500, str(e))

    def handle_download_json_report(self):
        try:
            if not os.path.exists('report.json'):
                raise FileNotFoundError("JSON report not found")

            with open('report.json', 'rb') as f:
                report_content = f.read()

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-Disposition', 'attachment; filename="model_test_report.json"')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(report_content)
        except FileNotFoundError:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "error", "message": "Report not generated"}
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            self.send_error(500, str(e))

    async def handle_transform_request(self, request_data):
        try:
            if not isinstance(request_data, dict):
                raise ValueError("Request data must be JSON object")

            questions = request_data.get('questions', [])
            operations = request_data.get('operations', [])

            if not isinstance(questions, list):
                raise ValueError("Questions must be array")
            if not isinstance(operations, list):
                raise ValueError("Operations must be array")

            if len(questions) == 0:
                raise ValueError("At least one question required")
            if len(operations) == 0:
                raise ValueError("At least one operation required")

            transformation_system = change.TransformationSystem()
            question_objects = []
            for q in questions:
                question_objects.append(change.Question(
                    id=change.DataManager()._generate_id(q),
                    content=q,
                    batch_id=-1
                ))

            transformed_questions, _ = await transformation_system.transform_batch(
                batch_id=-1,
                operations=operations,
                concurrency=min(3, len(questions)))

            if not transformed_questions:
                raise ValueError("No transformed questions generated")

            results = []
            for original, transformed in zip(questions, transformed_questions[:5]):
                if transformed and transformed.content:
                    results.append({
                        "original": original,
                        "transformed": transformed.content,
                        "operation": operations[0]["type"] if operations else "unknown",
                        "timestamp": datetime.now().isoformat()
                    })

            return {
                "status": "success",
                "count": len(results),
                "transformed_questions": results
            }
        except Exception as e:
            self.logger.error(f"Transform request failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def do_POST(self):
        if self.path == '/generate':
            self.handle_generate_request()
        elif self.path == '/transform':
            self.handle_transform_post()
        elif self.path == '/audit':
            self.handle_audit_request()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_generate_request(self):
        self.logger.info("\nReceived new generation request...")
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data)

            # 根据前端传来的config文件名加载对应的配置
            config_file = request_data.get('config', 'generate_config.json')
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file {config_file} not found")

            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if 'types' in request_data:
                config['QUESTION_TYPES'] = {t: 1.0 / len(request_data['types']) for t in request_data['types']}
            if 'difficulty' in request_data:
                config['DIFFICULTY_LEVELS'] = {request_data['difficulty']: 1.0}

            if request_data.get('is_custom_topic', False):
                custom_topic = request_data.get('topic', '').strip()
                if custom_topic:
                    config['ACTIVE_TOPICS'] = [custom_topic]
            elif 'topic' in request_data and request_data['topic']:
                config['ACTIVE_TOPICS'] = [request_data['topic']]

            config['TARGET_QA_COUNT'] = max(request_data.get('count', 5), 3)
            config['QA_PER_REQUEST'] = min(5, config['TARGET_QA_COUNT'])

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            qa_pairs, _, _ = loop.run_until_complete(process_qa_generation(config))
            loop.close()

            response = {
                "status": "success",
                "count": len(qa_pairs),
                "questions": qa_pairs
            }
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            error_response = {
                "status": "error",
                "message": str(e)
            }
            self.wfile.write(json.dumps(error_response).encode())

    def handle_transform_post(self):
        self.logger.info("\nReceived transform request...")
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self.handle_transform_request(request_data))
            loop.close()

            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            error_response = {
                "status": "error",
                "message": str(e)
            }
            self.wfile.write(json.dumps(error_response).encode())

    def handle_audit_request(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data)

            audit_mode = request_data.get('mode', 'full')
            qa_pairs = load_latest_qa_pairs_from_dir()

            auditor = ModelAuditor(mode=audit_mode)
            auditor.start_monitoring()
            auditor.content_safety_check(qa_pairs)
            report = auditor.generate_report(qa_pairs)

            # Save reports
            with open("report.json", "w") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            # Generate markdown report
            markdown_content = self.generate_markdown_report(report)
            with open("report.md", "w", encoding="utf-8") as f:
                f.write(markdown_content)

            self.wfile.write(json.dumps(report).encode())
        except Exception as e:
            error_response = {
                "status": "error",
                "message": str(e)
            }
            self.wfile.write(json.dumps(error_response).encode())

    def generate_markdown_report(self, report_data):
        markdown = []
        markdown.append("# Model Test Report\n")
        markdown.append(f"*Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        audit_result = report_data.get('audit_result', {})
        success_rate = (audit_result.get('success_rate', 0) * 100)
        is_successful = audit_result.get('is_successful', False)

        markdown.append("## Audit Summary\n")
        markdown.append(f"- **Mode**: {report_data.get('metadata', {}).get('audit_mode', 'full').upper()}\n")
        markdown.append(f"- **Success Rate**: {success_rate:.1f}%\n")
        markdown.append(f"- **Result**: {'✅ PASSED (>90%)' if is_successful else '❌ FAILED'}\n\n")

        sections = [
            ('metadata', 'Metadata'),
            ('performance', 'Performance Metrics'),
            ('resources', 'Resource Usage'),
            ('compliance', 'Compliance Checks'),
            ('safety_issues', 'Safety Issues')
        ]

        for section_key, section_title in sections:
            if section_key in report_data:
                content = report_data[section_key]
                markdown.append(f"## {section_title}\n")

                if isinstance(content, dict):
                    for key, value in content.items():
                        if value is not None:
                            markdown.append(f"- **{key}**: {value}\n")
                elif isinstance(content, list):
                    for item in content:
                        markdown.append(f"- {item}\n")
                markdown.append("\n")

        if 'qa_pairs' in report_data and report_data['qa_pairs']:
            markdown.append("## Test Cases\n")
            for qa in report_data['qa_pairs']:
                status = "✅ PASS" if qa.get('success') else "❌ FAIL"
                markdown.append(
                    f"### Input\n{qa.get('question', 'N/A')}\n\n"
                    f"### Output\n{qa.get('answer', 'N/A')}\n\n"
                    f"**Result**: {status}\n"
                )
                if 'issue' in qa:
                    markdown.append(f"**Issue**: {qa['issue']}\n")
                markdown.append("---\n")

        return "".join(markdown)


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8000), APIHandler)
    print("Server started on http://0.0.0.0:8000")
    print("Available endpoints:")
    print("POST /generate - Generate questions")
    print("POST /transform - Transform questions")
    print("POST /audit - Run model audit")
    print("GET /get_report - Check report status")
    print("GET /download_report - Download markdown report")
    print("GET /download_json_report - Download JSON report")
    server.serve_forever()