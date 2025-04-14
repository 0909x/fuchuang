# app.py - 大模型测试与报告生成系统
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import json
import os
from audit import ModelAuditor, AuditMode, load_latest_qa_pairs_from_dir
import uvicorn
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class AuditStatus(BaseModel):
    is_ready: bool = False


audit_status = AuditStatus()


# ======================
# 审计状态管理端点
# ======================
@app.post("/set_audit_ready")
async def set_audit_ready():
    """设置审计准备状态为就绪"""
    audit_status.is_ready = True
    return {"status": "ready"}


@app.get("/check_audit_ready")
async def check_audit_ready():
    """检查当前审计准备状态"""
    return audit_status


@app.post("/reset_audit")
async def reset_audit():
    """重置审计准备状态"""
    audit_status.is_ready = False
    return {"status": "reset"}


# ======================
# 报告生成核心功能
# ======================
def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """
    生成标准化的Markdown格式测试报告
    参数:
        report_data: 包含所有测试数据的字典
    返回:
        格式化后的Markdown字符串
    """
    markdown = []

    # 报告标题和元信息
    markdown.append("# 大模型测试报告\n")
    markdown.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

    # 审计摘要
    audit_result = report_data.get('audit_result', {})
    success_rate = (audit_result.get('success_rate', 0) * 100)
    is_successful = audit_result.get('is_successful', False)

    markdown.append("## 审计摘要\n")
    markdown.append(f"- **审计模式**: {report_data.get('metadata', {}).get('audit_mode', 'full').upper()}\n")
    markdown.append(f"- **测试用例总数**: {len(report_data.get('qa_pairs', []))}\n")
    markdown.append(f"- **成功率**: {success_rate:.1f}%\n")
    markdown.append(f"- **最终结果**: {'✅ 通过 (>90%)' if is_successful else '❌ 未通过'}\n\n")

    # 详细报告部分
    sections = [
        ('metadata', '元数据'),
        ('performance', '性能指标'),
        ('resources', '资源使用'),
        ('compliance', '合规性检查'),
        ('safety_issues', '安全问题')
    ]

    for section_key, section_title in sections:
        if section_key in report_data:
            content = report_data[section_key]
            if not content:
                continue

            markdown.append(f"## {section_title}\n")

            if isinstance(content, dict):
                for key, value in content.items():
                    if value is not None:
                        markdown.append(f"- **{key}**: {value}\n")
            elif isinstance(content, list):
                for item in content:
                    markdown.append(f"- {item}\n")
            markdown.append("\n")

    # 测试用例详情
    if 'qa_pairs' in report_data and report_data['qa_pairs']:
        markdown.append("## 测试用例详情\n")

        # 统计通过率
        passed = sum(1 for qa in report_data['qa_pairs'] if qa.get('success'))
        total = len(report_data['qa_pairs'])
        markdown.append(f"**通过率**: {passed}/{total} ({passed / total * 100:.1f}%)\n\n")

        for qa in report_data['qa_pairs']:
            status = "✅ 通过" if qa.get('success') else "❌ 未通过"
            markdown.append(
                f"### 输入\n{qa.get('question', '无')}\n\n"
                f"### 模型输出\n{qa.get('answer', '无')}\n\n"
                f"**测试结果**: {status}\n"
            )
            if 'issue' in qa:
                markdown.append(f"**问题描述**: {qa['issue']}\n")
            if 'evaluation_metrics' in qa:
                markdown.append("**评估指标**:\n")
                for metric, value in qa['evaluation_metrics'].items():
                    markdown.append(f"- {metric}: {value}\n")
            markdown.append("---\n")

    return "".join(markdown)


# ======================
# 报告下载端点
# ======================
@app.get("/get_report")
async def get_report():
    """
    获取报告下载链接
    返回:
        JSON和Markdown报告的下载URL
    """
    if not os.path.exists("report.json"):
        raise HTTPException(status_code=404, detail="报告未生成，请先运行测试")

    # 确保Markdown报告存在
    if not os.path.exists("report.md"):
        with open("report.json", "r") as f:
            report_data = json.load(f)
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(generate_markdown_report(report_data))

    return {
        "markdown_url": "/download_report",
        "json_url": "/download_json_report"
    }


@app.get("/download_json_report")
async def download_json_report():
    """下载JSON格式的测试报告"""
    if not os.path.exists("report.json"):
        raise HTTPException(status_code=404, detail="JSON报告未找到")

    return FileResponse(
        "report.json",
        filename="model_audit_report.json",
        media_type="application/json"
    )


@app.get("/download_report")
async def download_report():
    """下载Markdown格式的测试报告"""
    if not os.path.exists("report.md"):
        if os.path.exists("report.json"):
            with open("report.json", "r") as f:
                report_data = json.load(f)
            with open("report.md", "w", encoding="utf-8") as f:
                f.write(generate_markdown_report(report_data))
        else:
            raise HTTPException(status_code=404, detail="报告未生成")

    return FileResponse(
        "report.md",
        filename="model_audit_report.md",
        media_type="text/markdown"
    )


# ======================
# 测试执行端点
# ======================
@app.get("/run_test")
async def run_test(mode: AuditMode = 'full'):
    """
    执行模型测试
    参数:
        mode: 测试模式 (basic/full/performance)
    返回:
        完整的测试报告数据
    """
    try:
        # 初始化审计器
        auditor = ModelAuditor(mode=mode)

        # 加载测试用例
        qa_pairs = load_latest_qa_pairs_from_dir()
        if not qa_pairs:
            raise HTTPException(
                status_code=400,
                detail="未找到有效测试用例，请检查数据生成逻辑"
            )

        # 执行测试
        auditor.start_monitoring()
        auditor.content_safety_check(qa_pairs)
        report = auditor.generate_report(qa_pairs)

        # 增强报告数据
        report['audit_result'] = {
            'success_rate': report['performance'].get('success_rate', 0),
            'is_successful': report['compliance']['content_safety']
        }

        # 标记测试用例状态
        safety_issues = report.get('safety_issues', [])
        for qa in qa_pairs:
            qa['success'] = not any(
                issue['question'] == qa['question']
                for issue in safety_issues
            )
            if not qa['success']:
                qa['issue'] = next(
                    (issue['issue'] for issue in safety_issues
                     if issue['question'] == qa['question']),
                    "未知问题"
                )

        report['qa_pairs'] = qa_pairs

        # 保存报告
        with open("report.json", "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 生成Markdown报告
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(generate_markdown_report(report))

        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"测试执行失败: {str(e)}")


# ======================
# 仪表盘页面
# ======================
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """主仪表盘页面"""
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>大模型测试平台</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {
                font-family: 'Microsoft YaHei', sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .hidden { display: none; }
            .status-box {
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                font-weight: bold;
            }
            .success { background-color: #d4edda; color: #155724; }
            .failure { background-color: #f8d7da; color: #721c24; }
            .qa-item {
                margin: 15px 0;
                padding: 15px;
                border-radius: 5px;
                background-color: #f8f9fa;
                border-left: 5px solid;
            }
            .qa-pass { border-color: #28a745; }
            .qa-fail { border-color: #dc3545; }
            button {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover { background-color: #0069d9; }
            button:disabled { background-color: #6c757d; }
            select {
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #ced4da;
            }
            .report-section {
                margin-top: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>大模型测试平台</h1>

        <div>
            <select id="auditMode">
                <option value="basic">基础模式</option>
                <option value="full" selected>完整模式</option>
                <option value="performance">性能模式</option>
            </select>
            <button onclick="runTest()">执行测试</button>
            <div id="status" class="status-box hidden"></div>
        </div>

        <div id="report-links" class="hidden">
            <h3>报告下载</h3>
            <a href="/download_report" class="button">下载Markdown报告</a>
            <a href="/download_json_report" class="button">下载JSON报告</a>
        </div>

        <div id="report-container" class="report-section hidden">
            <h2>测试结果概览</h2>
            <div id="summary"></div>
            <h3>详细测试用例</h3>
            <div id="qa-list"></div>
        </div>

        <script>
            async function runTest() {
                const mode = document.getElementById('auditMode').value;
                const button = document.querySelector('button');
                const status = document.getElementById('status');
                const reportLinks = document.getElementById('report-links');
                const reportContainer = document.getElementById('report-container');

                // 重置UI状态
                button.disabled = true;
                status.textContent = "测试执行中...";
                status.className = "status-box";
                status.classList.remove("hidden");
                reportLinks.classList.add("hidden");
                reportContainer.classList.add("hidden");

                try {
                    const response = await fetch(`/run_test?mode=${mode}`);
                    if (!response.ok) {
                        throw new Error(await response.text());
                    }

                    const data = await response.json();
                    status.textContent = "测试执行完成！";
                    status.classList.add("success");
                    reportLinks.classList.remove("hidden");

                    // 显示测试结果
                    displayReport(data);
                } catch (error) {
                    status.textContent = "测试失败: " + error.message;
                    status.classList.add("failure");
                } finally {
                    button.disabled = false;
                }
            }

            function displayReport(data) {
                const reportContainer = document.getElementById('report-container');
                const summaryDiv = document.getElementById('summary');
                const qaListDiv = document.getElementById('qa-list');

                // 显示摘要信息
                const successRate = (data.audit_result?.success_rate || 0) * 100;
                const isSuccess = data.audit_result?.is_successful;

                summaryDiv.innerHTML = `
                    <p><strong>测试模式</strong>: ${data.metadata?.audit_mode || 'full'}</p>
                    <p><strong>成功率</strong>: ${successRate.toFixed(1)}%</p>
                    <p><strong>最终结果</strong>: ${isSuccess ? '✅ 通过' : '❌ 未通过'}</p>
                `;

                // 显示测试用例
                if (data.qa_pairs && data.qa_pairs.length > 0) {
                    qaListDiv.innerHTML = '';
                    data.qa_pairs.forEach(qa => {
                        const qaItem = document.createElement('div');
                        qaItem.className = `qa-item ${qa.success ? 'qa-pass' : 'qa-fail'}`;
                        qaItem.innerHTML = `
                            <p><strong>输入:</strong> ${qa.question || '无'}</p>
                            <p><strong>输出:</strong> ${qa.answer || '无'}</p>
                            <p><strong>结果:</strong> ${qa.success ? '✅ 通过' : '❌ 未通过'}</p>
                            ${qa.issue ? `<p><strong>问题:</strong> ${qa.issue}</p>` : ''}
                        `;
                        qaListDiv.appendChild(qaItem);
                    });
                }

                reportContainer.classList.remove("hidden");
            }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)