# change.py
import json
import logging
import hashlib
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel
from openai import AsyncOpenAI

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('transformation')


# === 基础数据模型 ===
class Question(BaseModel):
    id: str
    content: Dict
    history: List[Dict] = []
    relations: Dict[str, str] = {}
    batch_id: int
    metadata: Dict = {}


class TransformationLog(BaseModel):
    timestamp: datetime
    operation: str
    params: Dict
    original_id: str
    new_id: Optional[str]
    status: str = "completed"


# === 核心变形引擎 ===
class TransformationEngine:
    def __init__(self, config: Dict):
        self.client = AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        self.model = config.get("model", "deepseek-chat")
        self.max_retries = config.get("MAX_RETRIES", 3)
        self.retry_delay = config.get("RETRY_DELAY", 5)

        # 扩展的提示模板
        self.prompt_templates = {
            "paraphrase": """请用不同的表述方式改写以下题目，保持核心考点不变：
原题：{question}
要求：
1. 替换至少3个专业术语的同义词
2. 调整句式结构但保持题目长度
3. 保持选项数量不变（如果是选择题）
4. 确保生成的题目与原题难度相同
5. 返回完整的JSON格式题目，包含所有必要字段
返回包含'question'和'type'字段的完整JSON""",

            "restructure": """将题目从{source_type}转换为{target_type}格式：
原题：{question}
当前题型：{source_type}
目标题型：{target_type}
要求：
1. 保持考察知识点不变
2. {conversion_rule}
3. 保留原题难度级别
4. 返回完整的JSON格式题目，包含所有必要字段
返回包含'question'和'type'字段的完整JSON""",

            "difficulty": """将题目{direction}化：
原题：{question}
当前难度：{current_level}
目标难度：{target_level}
要求：
1. 调整选项/问题表述
2. {difficulty_rule}
3. 保持专业术语准确性
4. 返回完整的JSON格式题目，包含所有必要字段
返回包含'question'和'type'字段的完整JSON""",

            "format": """将题目转换为{target_format}格式：
{question}
当前格式：{current_format}
目标格式：{target_format}
要求：
1. 保持内容完整性
2. 符合{target_format}规范
3. 保留关键元数据
4. 确保格式转换后的题目仍可读
返回包含'question'和'type'字段的完整JSON""",

            "translate": """将题目从{source_lang}翻译为{target_lang}：
{question}
要求：
1. 专业术语使用公认译法
2. 保留原格式标记
3. 符合目标语言的学术规范
4. 确保翻译后的题目仍可读
返回包含'question'和'type'字段的完整JSON，并确保包含'json'字样"""
        }

        # 扩展的转换规则
        self.rules = {
            # 题型转换规则
            "multiple_choice->short_answer": "生成3-5个关键得分点，考察相同的知识点",
            "short_answer->multiple_choice": "生成4个合理选项，包含1个正确答案和3个干扰项",
            "fill_blank->multiple_choice": "基于空缺部分生成4个合理选项",
            "true_false->multiple_choice": "将判断题转换为选择题，保持考察点不变",

            # 难度调整规则
            "easy": "增加提示信息或简化专业术语",
            "medium": "保持适当挑战性但不过于复杂",
            "hard": "增加干扰选项或引入复杂概念",

            # 格式转换说明
            "markdown": "使用Markdown语法格式化题目",
            "latex": "使用LaTeX语法格式化题目",
            "plain": "转换为纯文本格式"
        }

    async def transform(self, question: Dict, operation: str, **params) -> Dict:
        """执行题目变形"""
        for attempt in range(self.max_retries):
            try:
                prompt = self._build_prompt(question, operation, params)
                response = await self._call_api(prompt)
                transformed = self._parse_response(response, operation)

                if transformed:
                    return transformed

            except Exception as e:
                logger.error(f"变形失败(尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                continue

        logger.error(f"达到最大重试次数，放弃变形操作: {operation}")
        return None

    def _build_prompt(self, question: Dict, operation: str, params: Dict) -> str:
        # 确保必要字段存在
        question["source_type"] = question.get("type", "unknown")
        question.setdefault("source_type", question.get("type", "unknown"))
        question.setdefault("current_format", "json")
        question.setdefault("question", "")

        base_params = {
            "question": question["question"],  # 只传问题文本
            "current_level": question.get("difficulty", "unknown"),
            "current_format": question["current_format"],
            "current_type": question["source_type"],  # 使用source_type
            "source_lang": "zh" if operation == "translate" and params.get("target_lang") == "en" else "en"
        }

        if operation == "restructure":
            conversion_key = f"{question.get('type', 'unknown')}->{params.get('target_type', 'unknown')}"
            params["conversion_rule"] = self.rules.get(conversion_key, "")

        if operation == "difficulty":
            params["difficulty_rule"] = self.rules.get(params.get("direction", ""), "")
            params["target_level"] = params.get("direction", "")

        return self.prompt_templates[operation].format(**base_params, **params)

    async def _call_api(self, prompt: str) -> str:
        """调用API"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str, operation: str) -> Dict:
        """解析响应"""
        try:
            data = json.loads(response)
            # 校验必要字段
            required_fields = ["question", "type"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # 添加变形元数据
            data["transformations"] = data.get("transformations", []) + [{
                "operation": operation,
                "timestamp": datetime.now().isoformat()
            }]

            return data
        except json.JSONDecodeError as e:
            logger.error(f"无效的JSON响应: {str(e)}")
            logger.debug(f"原始响应内容: {response}")
            return None
        except Exception as e:
            logger.error(f"响应解析失败: {str(e)}")
            return None


# === 数据管理系统 ===
class DataManager:
    def __init__(self):
        self.qa_dir = Path("generated_qa")
        self.transformed_dir = Path("transformed_qa")
        self.transformed_dir.mkdir(parents=True, exist_ok=True)

    def _generate_id(self, content: Dict) -> str:
        """生成唯一ID"""
        content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content_str.encode()).hexdigest()[:12]

    def load_questions(self, batch_id: int) -> List[Question]:
        """加载指定批次的题目（修复版）"""
        try:
            # 找到所有候选文件
            candidates = list(self.qa_dir.glob("qa_pairs_*.json"))
            if not candidates:
                raise FileNotFoundError("没有找到任何题目文件")

            # 按修改时间排序
            candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            print(f"找到 {len(candidates)} 个候选文件，最新文件: {candidates[0].name}")

            # 尝试加载文件
            for filepath in candidates:
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        print(f"从 {filepath.name} 加载到 {len(data)} 条记录")

                        if not data:
                            print("警告: 文件内容为空，尝试下一个文件")
                            continue

                        # 检查批次ID是否匹配
                        if batch_id != -1 and data[0].get("batch_id") != batch_id:
                            print(f"批次ID不匹配: 期望{batch_id}，实际{data[0].get('batch_id')}")
                            continue

                        return [
                            Question(
                                id=self._generate_id(q),
                                content=q,
                                batch_id=q.get("batch_id", batch_id),
                                metadata={"source_file": filepath.name}
                            ) for q in data
                        ]

                except json.JSONDecodeError as e:
                    print(f"文件 {filepath.name} JSON解析失败: {str(e)}")
                    continue

            raise ValueError("没有找到有效的题目数据")

        except Exception as e:
            print(f"加载题目失败: {str(e)}")
            raise

    def save_transformed(self, questions: List[Question], source_batch: int) -> Path:
        """保存变形后的题目"""
        if not questions:
            raise ValueError("没有有效的题目可保存")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"transformed_{timestamp}_from_{source_batch}.json"
        filepath = self.transformed_dir / filename

        output = []
        for q in questions:
            if not q:
                continue

            output.append({
                "id": q.id,
                "content": q.content,
                "history": q.history,
                "relations": q.relations,
                "batch_id": q.batch_id,
                "metadata": {
                    **q.metadata,
                    "transformed_at": timestamp,
                    "source_batch": source_batch
                }
            })

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            logger.info(f"成功保存 {len(output)} 题到 {filepath}")
            return filepath

        except (IOError, json.JSONEncodeError) as e:
            logger.error(f"保存失败: {str(e)}")
            raise


# === 任务处理系统 ===
class TransformationSystem:
    def __init__(self, config_path: str = "generate_config.json"):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

            self.engine = TransformationEngine(self.config)
            self.data_manager = DataManager()
            self.logs: List[TransformationLog] = []

        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise

    async def transform_batch(
            self,
            batch_id: int,
            operations: List[Dict],
            concurrency: int = 5
    ) -> Tuple[List[Question], List[TransformationLog]]:
        """处理整个批次的变形"""
        try:
            original_questions = self.data_manager.load_questions(batch_id)
            if not original_questions:
                raise ValueError(f"批次{batch_id}没有可用的题目")

            semaphore = asyncio.Semaphore(concurrency)
            tasks = []

            logger.info(f"开始处理批次 {batch_id} 的 {len(original_questions)} 题")

            # 为每个题目创建变形任务
            for question in original_questions:
                for op in operations:
                    task = self._process_question(question, op, semaphore)
                    tasks.append(task)

            # 并行执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"任务处理异常: {str(result)}")
                    continue
                if result:
                    valid_results.append(result)

            # 保存结果
            saved_path = self.data_manager.save_transformed(valid_results, batch_id)

            logger.info(f"批次 {batch_id} 处理完成，生成 {len(valid_results)} 个变形题目")

            return valid_results, self.logs

        except Exception as e:
            logger.error(f"处理批次 {batch_id} 失败: {str(e)}")
            raise

    async def _process_question(
            self,
            question: Question,
            op: Dict,
            semaphore: asyncio.Semaphore
    ) -> Optional[Question]:
        """处理单个题目的变形（修复版）"""
        async with semaphore:
            try:
                start_time = datetime.now()

                # 修复参数传递方式
                operation_type = op["type"]
                transformed = await self.engine.transform(
                    question=question.content,
                    operation=operation_type,
                    **op.get("params", {})
                )

                if not transformed:
                    raise ValueError("API返回空结果")

                new_question = self._create_new_version(question, transformed, op)
                duration = (datetime.now() - start_time).total_seconds()

                logger.info(
                    f"成功处理题目 {question.id[:8]} 操作: {operation_type} "
                    f"耗时: {duration:.2f}s"
                )
                return new_question

            except Exception as e:
                self.logs.append(TransformationLog(
                    timestamp=datetime.now(),
                    operation=op["type"],
                    params=op.get("params", {}),
                    original_id=question.id,
                    new_id=None,
                    status=f"failed: {str(e)}"
                ))
                logger.error(f"处理题目 {question.id[:8]} 失败: {str(e)}")
                return None

    def _create_new_version(
            self,
            original: Question,
            content: Dict,
            operation: Dict
    ) -> Question:
        """创建新版本记录"""
        new_id = self.data_manager._generate_id(content)

        # 保留原题历史并添加新记录
        new_history = original.history + [{
            "operation": operation["type"],
            "params": operation.get("params", {}),
            "timestamp": datetime.now().isoformat(),
            "source_id": original.id
            }]

        # 建立关系映射
        relations = {
            **original.relations,
            "origin": original.relations.get("origin", original.id),
            "last_transformation": operation["type"]
        }

        new_question = Question(
            id=new_id,
            content=content,
            history=new_history,
            relations=relations,
            batch_id=original.batch_id,
            metadata={
                **original.metadata,
                "transformed_at": datetime.now().isoformat()
            }
        )

        # 记录日志
        self.logs.append(TransformationLog(
            timestamp=datetime.now(),
            operation=operation["type"],
            params=operation.get("params", {}),
            original_id=original.id,
            new_id=new_id,
            status="completed"
        ))

        return new_question