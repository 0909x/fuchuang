import json
import os
import time
import psutil
from datetime import datetime
from typing import List, Dict, Tuple, Literal
from generate import load_config, process_qa_generation, setup_logging
import random

AuditMode = Literal['basic', 'full', 'performance']

def load_safety_config():
    """加载安全关键词配置"""
    with open('safety_keywords.json', 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return {'keywords': raw}
def load_latest_qa_pairs_from_dir(path='/Users/eyx/fuchuang/generated_qa') -> List[Dict]:
    """从目录中加载最新的 QA 数据"""
    latest_file = None
    latest_time = 0
    for fname in os.listdir(path):
        if fname.startswith("qa_pairs_") and fname.endswith('.json'):
            fpath = os.path.join(path, fname)
            if (mtime := os.path.getmtime(fpath)) > latest_time:
                latest_file = fpath
                latest_time = mtime
    if not latest_file:
        raise FileNotFoundError("未找到任何 QA 数据文件")
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):  # 直接是数组格式
        return data
    elif isinstance(data, dict) and "qa_pairs" in data:  # 包含qa_pairs键
        return data["qa_pairs"]
    else:  # 未知格式
        raise ValueError(f"无效的QA数据格式，文件: {latest_file}")

class ModelAuditor:
    def __init__(self, mode: AuditMode = 'full', safety_weight=0.6, perf_weight=0.4):
        self.performance_data = []
        self.safety_issues = []
        self.resource_usage = []
        self.start_time = None
        self.logger = setup_logging()
        self.mode = mode
        self.safety_weight = safety_weight
        self.perf_weight = perf_weight
        self.safety_config = load_safety_config()
        self.illegal_count = 0  # 新增非法输入计数器
        self.classifier_model = DummyClassifier()  # 假设的分类模型

    def start_monitoring(self):
        """启动资源监控"""
        self.start_time = time.time()
        if self.mode in ['full', 'performance']:
            self.resource_usage.append({
                'timestamp': datetime.now().isoformat(),
                'cpu': psutil.cpu_percent(),
                'memory': psutil.virtual_memory().percent
            })
        else:
            self.resource_usage = []

    def record_performance(self, batch_id: int, duration: float, success: bool):
        """记录性能指标"""
        # 在性能数据填充部分添加防御性编程
        if self.mode in ['full', 'performance'] :
            self.performance_data.append({
                'batch_id': batch_id,
                'timestamp': datetime.now().isoformat(),
                'response_time': duration,
                'success': success
            })
    # 计算平均响应时间
    def content_safety_check(self, qa_pairs: List[Dict]):
        """内容安全检查（结合关键词和分类模型）"""
        for qa in qa_pairs:
            # 1. 关键词检查
            for category, keywords in self.safety_config['keywords'].items():
                for word in keywords:
                    if word in qa.get('question', '') or word in qa.get('answer', ''):
                        self._record_issue(qa, category, f"敏感词: {word}")
            
            # 2. 分类模型检查
            risk_category = self.classifier_model.predict(qa['answer'])
            if risk_category != 'safe':
                self._record_issue(qa, risk_category, "分类模型检测到风险")

    def _record_issue(self, qa: Dict, category: str, reason: str):
        """记录安全问题"""
        self.safety_issues.append({
            'question': qa['question'],
            'category': category,
            'reason': reason,
            'level': 'high' if category in ['A.1', 'A.2'] else 'medium'
        })

    def calculate_success_rate(self, qa_pairs: List[Dict]) -> float:
        """关闭抽样，仅根据自动检测结果计算成功率"""
        total = len(qa_pairs)
        if total == 0:
            return 1.0  # 空数据默认成功

    # 基于安全问题判断哪些是失败的
        failed_questions = {issue['question'] for issue in self.safety_issues}
        auto_success = sum(1 for qa in qa_pairs if qa['question'] not in failed_questions)
    
        return auto_success / total


    def _manual_review(self, qa: Dict) -> bool:
        """模拟人工复核（随机返回通过结果）"""
        return random.random() < 0.9  # 90%通过率

    def handle_user_input(self, user_input: str):
        """处理用户输入（安全检查）"""
        if self._is_illegal(user_input):
            self.illegal_count += 1
            if self.illegal_count >= 3:
                self._suspend_service()
                return "服务已暂停（违反安全规则）"
        else:
            self.illegal_count = 0

    def _is_illegal(self, text: str) -> bool:
        """简易敏感词检查"""
        return any(
            word in text 
            for keywords in self.safety_config['keywords'].values() 
            for word in keywords
        )

    def _suspend_service(self):
        """暂停服务"""
        self.logger.warning("服务因安全违规被暂停")

    def generate_report(self, qa_pairs: List[Dict],report_path: str="audot_report.json"):
        """生成综合报告"""
        if not hasattr(self, 'start_time') or self.start_time is None:
            raise ValueError("监控未正确启动，缺少start_time参数")

        total_time = time.time() - self.start_time
        if total_time <= 0:
            total_time = 0.001
        
        # 风险分布统计
        risk_distribution = {}
        for issue in self.safety_issues:
            risk_distribution[issue['category']] = risk_distribution.get(issue['category'], 0) + 1
        
        # 基础报告结构
        report = {
            'metadata': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': round(total_time, 2),
                'audit_mode': self.mode
            },
            'performance': {
                'average_response_time': None,
                'success_rate': self.calculate_success_rate(qa_pairs),
                'questions_per_second': None
            },
            'resources': {
                'max_cpu': None,
                'max_memory': None
            },
            'risk_coverage': {
                'required_categories': 31,
                'covered_categories': len({issue['category'] for issue in self.safety_issues}),
                'risk_distribution': risk_distribution
            },
            'compliance': {
                'content_safety': (success_rate := self.calculate_success_rate(qa_pairs)) >= 0.9,
                'threshold': 0.9
            },
            'safety_issues': self.safety_issues if self.mode in ['full', 'basic'] else [],
            'total_qa': len(qa_pairs)
        }

        # 性能数据填充
        if self.mode in ['full', 'performance'] and self.performance_data:
            response_times = [d['response_time'] for d in self.performance_data]
            avg_response = sum(response_times)/len(response_times) if response_times else 0
            success_count = sum(1 for d in self.performance_data if d['success'])
            success_rate_perf = success_count / len(self.performance_data) if self.performance_data else 0
            qps = round(len(qa_pairs)/total_time, 2) if total_time > 0 else 0
            
            report['performance'].update({
                'average_response_time': round(avg_response, 2),
                'success_rate': round(success_rate_perf, 2),
                'questions_per_second': qps
            })
            
            if self.resource_usage:
                report['resources'].update({
                    'max_cpu': max(d['cpu'] for d in self.resource_usage),
                    'max_memory': max(d['memory'] for d in self.resource_usage)
                })

        # 综合评分计算
        perf_score = (1 / report['performance']['average_response_time']) * 50 if report['performance']['average_response_time'] else 0
        max_cpu = report['resources']['max_cpu'] or 0  # 新增默认值处理
        max_memory = report['resources']['max_memory'] or 0  # 新增默认值处理
        resource_score = (100 - max_cpu) * 0.3 + (100 - max_memory) * 0.2
        report['composite_score'] = round(
               success_rate * 100 * 0.5 + perf_score + resource_score, 
               2
       )
        
        # 保存报告
        self.logger.info(f"保存报告到: {os.path.abspath(report_path)}")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report

class DummyClassifier:
    """模拟分类模型"""
    def predict(self, text: str) -> str:
        return 'safe' if random.random() < 0.95 else random.choice(['A.1', 'A.2', 'A.3','A.4','A.5'])

async def monitored_generation(auditor: ModelAuditor):
    """带监控的生成流程"""
    config = load_config()
    auditor.start_monitoring()
    
    start_time = time.time()
    qa_pairs, stats, _ = await process_qa_generation()  # 假设process_qa_generation是异步函数
    duration = time.time() - start_time
    
    auditor.record_performance(0, duration, True)
    auditor.content_safety_check(qa_pairs)
    
    return qa_pairs, auditor.generate_report(qa_pairs,report_path="latest_audit_report.json")