# test_transform_existing_fixed.py
import asyncio
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from change import TransformationSystem, Question

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transformation_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('transformation_test')


class ExistingQuestionTester:
    def __init__(self):
        self.system = TransformationSystem()
        self.qa_dir = Path("generated_qa")

    def load_latest_questions(self, max_questions=5):
        """加载最新生成的题目文件"""
        qa_files = sorted(
            [f for f in self.qa_dir.glob("qa_pairs_*.json") if "statistics" not in f.name],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        if not qa_files:
            raise FileNotFoundError("未找到任何题目文件，请先运行generate.py生成题目")

        latest_file = qa_files[0]
        logger.info(f"正在加载最新题目文件: {latest_file.name}")

        with open(latest_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        # 转换为Question对象（只取前max_questions题）
        return [
            Question(
                id=hashlib.md5(json.dumps(q).encode()).hexdigest()[:8],
                content=q,
                batch_id=q.get("batch_id", 0),
                metadata={"source_file": latest_file.name}
            ) for q in questions[:max_questions]
        ]

    async def test_transformation(self, questions, operation_name, operation_type, params_builder=None):
        """通用变形测试方法"""
        logger.info(f"\n{'=' * 30}")
        logger.info(f"开始测试: {operation_name}")
        logger.info(f"操作类型: {operation_type}")

        transformed_questions = []
        for q in questions:
            try:
                # 构建参数（如果有参数构建器）
                params = params_builder(q) if params_builder else {}

                result = await self.system.engine.transform(
                    question=q.content,
                    operation=operation_type,
                    **params
                )

                if not result:
                    raise ValueError("API返回空结果")

                # 验证必要字段
                for field in ["question", "type"]:
                    if field not in result:
                        raise ValueError(f"变形结果缺少必需字段: {field}")

                # 创建新版本记录
                new_q = Question(
                    id=hashlib.md5(json.dumps(result).encode()).hexdigest()[:8],
                    content=result,
                    batch_id=q.batch_id,
                    history=q.history + [{
                        "operation": operation_type,
                        "params": params,
                        "timestamp": datetime.now().isoformat()
                    }],
                    relations={
                        "origin": q.id,
                        "transformation": operation_type
                    },
                    metadata={
                        **q.metadata,
                        "tested_at": datetime.now().isoformat()
                    }
                )
                transformed_questions.append(new_q)

                # 记录对比信息
                logger.info(f"\n原题(ID:{q.id[:8]}): {q.content['question'][:50]}...")
                logger.info(f"变形后: {result['question'][:50]}...")

            except Exception as e:
                logger.error(f"题目 {q.id[:8]} 变形失败: {str(e)}")
                continue

        # 保存本批次变形结果
        if transformed_questions:
            save_path = self.system.data_manager.save_transformed(
                transformed_questions,
                source_batch=0  # 测试批次标记为0
            )
            logger.info(f"成功保存 {len(transformed_questions)} 个变形题目到: {save_path}")

        return transformed_questions


async def main():
    tester = ExistingQuestionTester()

    try:
        # 加载最新生成的题目（最多3题）
        questions = tester.load_latest_questions(max_questions=3)
        logger.info(f"成功加载 {len(questions)} 道题目进行测试")

        # 定义测试用例
        test_cases = [
            {
                "name": "同义替换",
                "type": "paraphrase",
                "params_builder": None
            },
            {
                "name": "题型转换",
                "type": "restructure",
                "params_builder": lambda q: {
                    "source_type": q.content["type"],  # 显式传递source_type
                    "target_type": "short_answer" if q.content["type"] == "multiple_choice" else "multiple_choice"
                }
            },
            {
                "name": "增加难度",
                "type": "difficulty",
                "params_builder": lambda _: {"direction": "hard"}
            },
            {
                "name": "转Markdown格式",
                "type": "format",
                "params_builder": lambda _: {"target_format": "markdown"}
            },
            {
                "name": "中译英",
                "type": "translate",
                "params_builder": lambda _: {"target_lang": "en"}
            }
        ]

        # 执行所有测试用例
        for case in test_cases:
            await tester.test_transformation(
                questions=questions,
                operation_name=case["name"],
                operation_type=case["type"],
                params_builder=case["params_builder"]
            )

    except Exception as e:
        logger.error(f"测试过程发生错误: {str(e)}", exc_info=True)
    finally:
        logger.info("测试完成")


if __name__ == "__main__":
    logger.info("=== 开始题目变形测试 ===")
    asyncio.run(main())