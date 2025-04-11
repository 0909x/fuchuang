// static/display.js - 数据可视化
document.addEventListener('DOMContentLoaded', async () => {
    const runTestRes = await fetch('/run_test');
    const report = await runTestRes.json();
    console.log('Dashboard loaded');
    // 性能图表
    new Chart(document.getElementById('perfChart'), {
        type: 'bar',
        data: {
            labels: ['平均响应', '成功率', '题/秒'],
            datasets: [{
                label: '性能指标',
                data: [
                    report.performance.average_response_time,
                    report.performance.success_rate * 100,
                    report.performance.questions_per_second
                ]
            }]
        }
    });
    
    // 安全图表
    new Chart(document.getElementById('safetyChart'), {
        type: 'pie',
        data: {
            labels: ['安全内容', '风险内容'],
            datasets: [{
                data: [
                    report.total_qa - report.safety_issues.length,
                    report.safety_issues.length
                ]
            }]
        }
    });
    document.body.innerHTML += '<a href="/download_report">下载报告</a>';
});
