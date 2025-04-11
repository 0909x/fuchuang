# app.py - 展示与报告生成
from fastapi import FastAPI, Response, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import json
import os
from audit import ModelAuditor, monitored_generation, AuditMode,load_latest_qa_pairs_from_dir
import uvicorn
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from pydantic import BaseModel
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
class AuditStatus(BaseModel):
    is_ready: bool = False

audit_status = AuditStatus()

@app.post("/set_audit_ready")
async def set_audit_ready():
    audit_status.is_ready = True
    return {"status": "ready"}

@app.get("/check_audit_ready")
async def check_audit_ready():
    return audit_status

@app.post("/reset_audit")
async def reset_audit():
    audit_status.is_ready = False
    return {"status": "reset"}


def json_to_markdown(json_data, markdown_path):
    """将JSON报告转换为Markdown格式"""
    markdown_content = []
    
    # 添加标题
    markdown_content.append("# 大模型测试报告\n")
    
    # 审核模式信息
    mode = json_data.get('metadata', {}).get('audit_mode', 'full').upper()
    markdown_content.append(f"## 审核模式: {mode}\n")
    
    # 审核结果
    audit_result = json_data.get('audit_result', {})
    success_rate = (audit_result.get('success_rate') or 0) * 100
    result_status = "通过 (>90%)" if audit_result.get('is_successful') else "不通过"
    markdown_content.append(
        f"**审核成功率**: {success_rate:.1f}%  \n"
        f"**判定结果**: {result_status}\n"
    )
    
    # 添加各部分内容
    sections = ['metadata', 'performance', 'resources', 'safety_issues']
    for section in sections:
        if section in json_data:
            content = json_data[section]
            markdown_content.append(f"## {section.upper()}\n")
            
            if isinstance(content, dict):
                for key, value in content.items():
                    if value is not None:
                        markdown_content.append(f"- **{key}**: {value}\n")
            markdown_content.append("\n")
    
    # 处理QA对
    if 'qa_pairs' in json_data and json_data['qa_pairs']:
        markdown_content.append("## 测试用例详情\n")
        for qa in json_data['qa_pairs']:
            status = "✅ 通过" if qa.get('success') else "❌ 不通过"
            markdown_content.append(
                f"### 输入  \n{qa.get('question', '无')}\n\n"
                f"### 输出  \n{qa.get('answer', '无')}\n\n"
                f"**结果**: {status}\n"
            )
            if 'issue' in qa:
                markdown_content.append(f"**问题**: {qa['issue']}\n")
            markdown_content.append("---\n")
    
    # 写入文件
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write("".join(markdown_content))


# 修改dashboard函数，添加输入输出展示区域
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
    <html>
        <head>
            <title>模型测试报告</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                .hidden { display: none; }
                .visible { display: block; }
                #status { margin-left: 10px; color: green; }
                #qa-container { margin-top: 20px; }
                .qa-item { border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; }
                .success { background-color: #e6ffe6; }
                .failure { background-color: #ffe6e6; }
                .audit-result { 
                    padding: 10px; 
                    margin: 10px 0; 
                    font-weight: bold; 
                    text-align: center;
                }
                .passed { background-color: #4CAF50; color: white; }
                .failed { background-color: #f44336; color: white; }
            </style>
        </head>
        <body>
            <h1>大模型测试报告</h1>
            <div>
                <select id="auditMode">
                    <option value="basic">基础模式</option>
                    <option value="full" selected>完整模式</option>
                    <option value="performance">性能模式</option>
                </select>
                <button onclick="runTest()">运行测试</button>
                <span id="status" class="hidden">测试完成！</span>
                <a href="/download_report" id="downloadLink" class="hidden">下载完整报告(PDF)</a>
            </div>
            
            <div id="audit-result" class="audit-result hidden"></div>
            
            <div id="charts">
                <canvas id="perfChart" width="400" height="200"></canvas>
                <canvas id="safetyChart" width="400" height="200"></canvas>
            </div>
            
            <div id="qa-container" class="hidden">
                <h3>测试用例详情</h3>
                <div id="qa-list"></div>
            </div>
            
            <script>
                async function runTest() {
                    const mode = document.getElementById('auditMode').value;
                    const button = document.querySelector('button');
                    const status = document.getElementById('status');
                    const downloadLink = document.getElementById('downloadLink');
                    const qaContainer = document.getElementById('qa-container');
                    const auditResult = document.getElementById('audit-result');
                    
                    // 重置显示
                    button.disabled = true;
                    button.textContent = '测试运行中...';
                    status.classList.add('hidden');
                    qaContainer.classList.add('hidden');
                    auditResult.classList.add('hidden');
                    
                    try {
                        const response = await fetch(`/run_test?mode=${mode}`);
                        if (!response.ok) {
                            throw new Error(await response.text());
                        }
                        
                        const data = await response.json();
                        status.textContent = '测试完成！';
                        status.classList.remove('hidden');
                        downloadLink.classList.remove('hidden');
                        
                        // 显示审核结果
                        showAuditResult(data.audit_result);
                        
                        // 显示QA对
                        showQAPairs(data.qa_pairs || []);
                        
                        // 更新图表数据
                        updateCharts(data);
                    } catch (error) {
                        status.textContent = '测试失败: ' + error.message;
                        status.style.color = 'red';
                        status.classList.remove('hidden');
                    } finally {
                        button.disabled = false;
                        button.textContent = '运行测试';
                    }
                }
                
                function showAuditResult(result) {
                    const auditResult = document.getElementById('audit-result');
                    const rate = (result.success_rate ?? 0) * 100;
                    auditResult.textContent = `审核成功率: ${rate.toFixed(1)}% | ` +
                                             `判定结果: ${result.is_successful ? '通过 (>90%)' : '不通过'}`;
                    auditResult.className = result.is_successful ? 'audit-result passed' : 'audit-result failed';
                    auditResult.classList.remove('hidden');
                }
                
                function showQAPairs(qaPairs) {
                    const qaContainer = document.getElementById('qa-container');
                    const qaList = document.getElementById('qa-list');
                    
                    qaList.innerHTML = '';
                    qaPairs.forEach(qa => {
                        const qaItem = document.createElement('div');
                        qaItem.className = 'qa-item ' + (qa.success ? 'success' : 'failure');
                        qaItem.innerHTML = `
                            <p><strong>输入:</strong> ${qa.question || '无'}</p>
                            <p><strong>输出:</strong> ${qa.answer || '无'}</p>
                            <p><strong>审核结果:</strong> ${qa.success ? '通过' : '不通过'}</p>
                            ${qa.issue ? `<p><strong>问题:</strong> ${qa.issue}</p>` : ''}
                        `;
                        qaList.appendChild(qaItem);
                    });
                    
                    qaContainer.classList.remove('hidden');
                }
                
                function updateCharts(data) {
                    // 这里添加图表更新逻辑
                    console.log('更新图表数据:', data);
                }
            </script>
        </body>
    </html>
    """

@app.get("/run_test")
async def run_test(mode: AuditMode = 'full'):
    try:
        auditor = ModelAuditor(mode=mode)
        try:
            qa_pairs = load_latest_qa_pairs_from_dir()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=400,
                detail=f"QA数据目录不存在: {os.path.abspath('generated_qa')}"
            )
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"QA文件格式错误: {str(e)}"
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"QA数据结构错误: {str(e)}"
            )
            
        if not qa_pairs:
            raise HTTPException(
                status_code=400, 
                detail=f"找到的QA文件为空，请检查数据生成逻辑"
            )
        auditor.start_monitoring()
        auditor.content_safety_check(qa_pairs)
        report = auditor.generate_report(qa_pairs)
        report['audit_result'] = {
            'success_rate': report['performance'].get('success_rate', 0),
            'is_successful': report['compliance']['content_safety']
}

        # 为每个QA对添加详细状态
        for qa in qa_pairs:
            qa['success'] = not any(
                issue['question'] == qa['question'] 
                for issue in report.get('safety_issues', [])
            )
            if not qa['success']:
                qa['issue'] = next(
                    (issue['issue'] for issue in report.get('safety_issues', [])
                    if issue['question'] == qa['question']),
                    "未知问题"
                )
        
        # 确保报告包含完整QA数据
        report['qa_pairs'] = qa_pairs
  
        # 保存报告
        with open("report.json", "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        markdown_path = "report.md"
        json_to_markdown(report, markdown_path)
        
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/download_report")
async def download_report():
    markdown_path = "report.md"
    if not os.path.exists(markdown_path):
        raise HTTPException(status_code=404, detail="报告未生成，请先运行测试")
    return FileResponse(
        markdown_path,
        filename="model_test_report.md",
        media_type="text/markdown"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)