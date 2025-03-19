# -*- coding:utf-8 -*-
import os
import json
import time
import logging
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from openai import OpenAI, AsyncOpenAI
import random
import sys
import argparse


# 加载配置文件
def load_config() -> Dict:
    """从test_config.json加载配置，不存在则创建配置"""
    config_path = os.path.join(os.path.dirname(__file__), "test_config.json")

    try:
        # 尝试加载已有的配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        # 如果配置文件不存在，创建空配置文件
        empty_config = {
            "api_key": "",  # API密钥
            "base_url": "",  # API基础URL
            "model": "",  # 用于回答问题的模型名称
            "MAX_CONCURRENT_CALLS": 5,  # 最大并发API调用数
            "BATCH_SIZE": 10,  # 批处理大小
            "FAIL_WAIT_TIME": 5,  # 失败等待时间(秒)
            "MAX_RETRIES": 3,  # 最大重试次数
            "RETRY_DELAY": 5,  # 重试延迟(秒)
            "QA_FILE_PATH": "",  # 问答对文件路径(留空，将由用户填写)
            "OUTPUT_DIR": "model_answers",  # 输出结果目录
            "LOG_DIR": "logs"  # 日志目录
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(empty_config, f, indent=4, ensure_ascii=False)
        print(f"已创建空的配置文件{config_path}，请填写必要的配置后再运行")
        return empty_config
    except json.JSONDecodeError as e:
        print(f"配置文件格式错误: {e}")
        raise


# 验证配置是否有效
def validate_config(config):
    """验证配置是否有效"""
    required_fields = ["api_key", "base_url", "model", "QA_FILE_PATH"]
    missing_fields = [field for field in required_fields if not config.get(field)]

    if missing_fields:
        return f"配置缺少必要字段: {', '.join(missing_fields)}"

    # 检查问答对文件是否存在
    if not os.path.exists(config["QA_FILE_PATH"]):
        return f"问答对文件不存在: {config['QA_FILE_PATH']}"

    return None


# 设置日志记录
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('model_tester')

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    log_file = os.path.join(log_dir, f"model_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_qa_pairs(file_path: str) -> List[Dict]:
    """加载问答对数据"""
    logger = logging.getLogger('model_tester')
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        logger.info(f"已加载 {len(qa_pairs)} 个问答对")
        return qa_pairs
    except Exception as e:
        logger.error(f"读取问答对文件失败 {file_path}: {str(e)}")
        sys.exit(1)


def prepare_questions(qa_pairs: List[Dict]) -> List[Dict]:
    """从问答对中提取问题和类型，准备提交给模型"""
    questions = []

    for i, qa in enumerate(qa_pairs):
        # 提取问题和问题类型
        question_text = qa.get("question", "")
        question_type = qa.get("type", "unknown")

        # 创建问题数据
        question_data = {
            "id": i,
            "question": question_text,
            "type": question_type,
            "original_qa": qa  # 保存完整的原始问答对
        }

        # 对于选择题，特别提取选项
        if question_type == "multiple_choice" and "options" in qa:
            question_data["options"] = qa["options"]

        questions.append(question_data)

    return questions


def format_multiple_choice_question(question: Dict) -> str:
    """格式化选择题，包含问题和选项"""
    # 获取问题文本
    question_text = question["question"]

    # 如果没有选项，直接返回问题文本
    if "options" not in question:
        return question_text

    # 格式化选项
    options = question["options"]
    options_text = ""

    # 处理不同的选项格式
    if isinstance(options, dict):
        # 如果选项是字典形式 {"A": "选项A", "B": "选项B", ...}
        for key, value in options.items():
            options_text += f"\n{key}. {value}"
    elif isinstance(options, list):
        # 如果选项是列表形式 ["选项A", "选项B", ...]
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D...
            options_text += f"\n{letter}. {option}"

    # 组合问题和选项
    return f"{question_text}{options_text}"


def get_prompt_for_answering(question: Dict) -> List[Dict]:
    """为回答问题准备提示信息，根据问题类型设计不同的提示"""
    # 获取问题类型
    question_type = question.get("type", "unknown")

    # 构建适合不同题型的系统提示
    system_prompts = {
        "multiple_choice": """你正在参加一个测试，这是一道选择题。请仔细阅读问题和选项，并只给出你认为正确的选项字母（A、B、C或D）作为回答。不要解释你的选择，只需提供选项字母。""",

        "fill_blank": """你正在参加一个测试，这是一道填空题。请仔细阅读问题，并给出填入空白处的确切答案。请简洁回答，只提供填空内容，不要重复问题或添加解释。""",

        "true_false": """你正在参加一个测试，这是一道判断题。请仔细阅读陈述，并判断其正确性。只需回答"true"（如果你认为陈述是正确的）或"false"（如果你认为陈述是错误的）。不要提供任何解释。""",

        "short_answer": """你正在参加一个测试，这是一道简答题。请仔细阅读问题，并给出简洁但全面的回答。不要过度详细，专注于问题要求的关键点。"""
    }

    # 获取对应类型的系统提示，如果类型未知则使用通用提示
    system_prompt = system_prompts.get(
        question_type,
        """你正在参加一个测试。请仔细阅读问题，并给出准确的答案。只需提供答案本身，不要添加额外解释。"""
    )

    # 准备用户提示（问题内容）
    if question_type == "multiple_choice":
        # 对于选择题，需要格式化问题和选项
        user_prompt = format_multiple_choice_question(question)
    else:
        # 其他题型直接使用问题内容
        user_prompt = question["question"]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


async def answer_question(question: Dict, semaphore: asyncio.Semaphore, logger: logging.Logger) -> Dict:
    """使用指定模型回答问题"""
    config = load_config()
    messages = get_prompt_for_answering(question)
    question_id = question["id"]
    question_type = question.get("type", "unknown")

    async with semaphore:
        for attempt in range(config["MAX_RETRIES"]):
            try:
                client = AsyncOpenAI(
                    api_key=config['api_key'],
                    base_url=config['base_url']
                )

                response = await client.chat.completions.create(
                    model=config["model"],
                    messages=messages,
                    temperature=0.1  # 低温度以获得确定性回答
                )

                model_answer = response.choices[0].message.content

                logger.debug(f"问题 {question_id} ({question_type}): 模型回答: {model_answer}")

                # 创建答案数据
                result = {
                    "question": question["question"],
                    "type": question_type,
                    "model_answer": model_answer
                }

                # 对于选择题，保存选项信息
                if question_type == "multiple_choice" and "options" in question:
                    result["options"] = question["options"]

                # 保存原始问答对中的正确答案
                if "original_qa" in question:
                    result["correct_answer"] = question["original_qa"].get("answer", "")

                return result

            except Exception as e:
                logger.error(f"问题 {question_id} API调用失败 (尝试 {attempt + 1}/{config['MAX_RETRIES']}): {str(e)}")
                if attempt < config["MAX_RETRIES"] - 1:
                    wait_time = config["RETRY_DELAY"] * (2 ** attempt)  # 指数退避
                    logger.info(f"问题 {question_id} 等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"问题 {question_id} 达到最大重试次数，放弃请求")
                    return {
                        "question": question["question"],
                        "type": question_type,
                        "model_answer": "API调用失败",
                        "error": str(e)
                    }


async def process_batch(questions: List[Dict], semaphore: asyncio.Semaphore, logger: logging.Logger) -> List[Dict]:
    """处理一批问题的回答"""
    tasks = []
    for question in questions:
        tasks.append(answer_question(question, semaphore, logger))

    return await asyncio.gather(*tasks)


def save_model_answers(model_answers: List[Dict], output_dir: str, qa_file_path: str, model_name: str):
    """保存模型的回答结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 获取问答对文件名（不含路径和扩展名）
    qa_filename = os.path.splitext(os.path.basename(qa_file_path))[0]

    # 创建输出文件名
    output_filename = f"answer_by_{model_name}_to_{qa_filename}.json"
    output_path = os.path.join(output_dir, output_filename)

    # 保存为JSON数组格式
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(model_answers, f, ensure_ascii=False, indent=2)

    return output_path


def save_intermediate_results(responses: List[Dict], output_dir: str, qa_file_path: str, model_name: str,
                              batch_num: int):
    """保存中间结果到临时文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 获取问答对文件名（不含路径和扩展名）
    qa_filename = os.path.splitext(os.path.basename(qa_file_path))[0]

    # 创建输出文件名
    temp_filename = f"answer_by_{model_name}_to_{qa_filename}_temp_batch{batch_num}.json"
    temp_path = os.path.join(output_dir, temp_filename)

    # 保存为JSON数组格式
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    return temp_path


async def main_async():
    """主异步函数"""
    # 加载配置
    config = load_config()
    validation_error = validate_config(config)
    if validation_error:
        print(f"错误: {validation_error}")
        print("请在test_config.json中填写所有必要的配置项后再运行")
        sys.exit(1)

    # 设置日志记录
    logger = setup_logging(config.get("LOG_DIR", "logs"))
    logger.info("开始模型测试任务")

    # 获取模型名称（用于文件命名）
    model_name = config["model"].replace("/", "_").replace(":", "_")

    # 加载问答对
    qa_pairs = load_qa_pairs(config["QA_FILE_PATH"])

    # 准备问题
    questions = prepare_questions(qa_pairs)

    # 按题型统计问题数量
    question_types = {}
    for q in questions:
        q_type = q.get("type", "unknown")
        question_types[q_type] = question_types.get(q_type, 0) + 1

    logger.info(f"已准备 {len(questions)} 个问题，按题型分布: {question_types}")

    # 创建输出目录
    output_dir = config.get("OUTPUT_DIR", "model_answers")
    os.makedirs(output_dir, exist_ok=True)

    # 创建并发控制信号量
    semaphore = asyncio.Semaphore(config.get("MAX_CONCURRENT_CALLS", 5))

    # 分批处理问题
    batch_size = config.get("BATCH_SIZE", 10)
    all_answers = []

    # 计算总批次
    total_batches = (len(questions) + batch_size - 1) // batch_size
    logger.info(f"将分 {total_batches} 批处理 {len(questions)} 个问题")

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        batch_num = i // batch_size + 1

        logger.info(f"处理批次 {batch_num}/{total_batches}, 包含 {len(batch)} 个问题")

        # 处理当前批次
        answers = await process_batch(batch, semaphore, logger)
        all_answers.extend(answers)

        # 定期保存中间结果
        if batch_num % 5 == 0 or batch_num == total_batches:
            temp_file = save_intermediate_results(
                all_answers,
                output_dir,
                config["QA_FILE_PATH"],
                model_name,
                batch_num
            )
            logger.info(f"已保存中间结果到 {temp_file}")

        # 显示进度
        success_count = sum(1 for a in answers if "error" not in a)
        logger.info(f"批次 {batch_num} 完成: {success_count}/{len(batch)} 成功")

        # 批次之间稍微暂停，避免API限流
        if i + batch_size < len(questions):
            await asyncio.sleep(1)

    # 保存最终结果
    output_file = save_model_answers(all_answers, output_dir, config["QA_FILE_PATH"], model_name)
    logger.info(f"所有模型回答已保存到 {output_file}")

    # 计算成功率
    total = len(all_answers)
    errors = sum(1 for a in all_answers if "error" in a)
    success_rate = ((total - errors) / total) * 100 if total > 0 else 0

    # 打印总结
    logger.info(f"\n=== 模型测试完成 ===")
    logger.info(f"- 总问题数: {total}")
    logger.info(f"- 成功回答数: {total - errors}")
    logger.info(f"- 失败回答数: {errors}")
    logger.info(f"- 成功率: {success_rate:.2f}%")
    logger.info(f"- 回答结果保存到: {output_file}")

    return {
        "output_file": output_file,
        "total_questions": total,
        "successful_responses": total - errors,
        "failed_responses": errors,
        "success_rate": success_rate
    }


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试模型回答问题的能力")
    parser.add_argument("--qa-file", help="问答对文件路径")
    parser.add_argument("--model", help="要测试的模型名称")
    parser.add_argument("--output-dir", help="输出目录")
    parser.add_argument("--concurrent", type=int, help="最大并发API调用数")
    parser.add_argument("--batch-size", type=int, help="批处理大小")

    return parser.parse_args()


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()

        # 加载配置
        config = load_config()

        # 根据命令行参数更新配置
        if args.qa_file:
            config["QA_FILE_PATH"] = args.qa_file
            print(f"使用问答对文件: {args.qa_file}")

        if args.model:
            config["model"] = args.model
            print(f"使用模型: {args.model}")

        if args.output_dir:
            config["OUTPUT_DIR"] = args.output_dir
            print(f"使用输出目录: {args.output_dir}")

        if args.concurrent:
            config["MAX_CONCURRENT_CALLS"] = args.concurrent
            print(f"最大并发调用数: {args.concurrent}")

        if args.batch_size:
            config["BATCH_SIZE"] = args.batch_size
            print(f"批处理大小: {args.batch_size}")

        # 保存更新后的配置
        with open(os.path.join(os.path.dirname(__file__), "test_config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        # 执行测试
        result = asyncio.run(main_async())

        print(f"\n模型测试完成")
        print(f"- 总问题数: {result['total_questions']}")
        print(f"- 成功率: {result['success_rate']:.2f}%")
        print(f"- 回答结果保存到: {result['output_file']}")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()