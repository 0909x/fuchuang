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


# 加载配置文件
def load_config() -> Dict:
    """从generate_config.json加载配置，不存在则创建配置"""
    config_path = os.path.join(os.path.dirname(__file__), "generate_config.json")

    try:
        # 尝试加载已有的配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        # 如果配置文件不存在，创建空配置文件
        empty_config = {
            "api_key": "",
            "base_url": "",
            "model": "",
            "MAX_CONCURRENT_CALLS": 5,
            "BATCH_SIZE": 8,
            "FAIL_WAIT_TIME": 5,
            "MAX_RETRIES": 3,
            "RETRY_DELAY": 5,
            "TARGET_QA_COUNT": 500,  # 目标问答对数量
            "QA_PER_REQUEST": 5,  # 每次API调用生成的问答对数量
            "OUTPUT_DIR": "generated_qa",
            "LOG_DIR": "logs",
            "QUESTION_TYPES": {  # 各种题型的生成比例
                "multiple_choice": 0.4,  # 选择题
                "fill_blank": 0.2,  # 填空题
                "true_false": 0.2,  # 判断题
                "short_answer": 0.2  # 简答题
            },
            "TOPICS": [  # 可选的主题列表
                "计算机科学",
                "人工智能",
                "机器学习",
                "数据科学",
                "编程语言",
                "软件工程",
                "网络安全",
                "云计算",
                "大数据",
                "区块链",
                "物联网",
                "Web开发",
                "移动开发",
                "操作系统",
                "数据库系统"
            ],
            "DIFFICULTY_LEVELS": {  # 难度分布
                "easy": 0.3,  # 简单题目
                "medium": 0.5,  # 中等难度
                "hard": 0.2  # 困难题目
            }
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
    required_fields = ["api_key", "base_url", "model"]
    missing_fields = [field for field in required_fields if not config.get(field)]

    if missing_fields:
        return f"配置缺少必要字段: {', '.join(missing_fields)}"

    return None


# 设置日志记录
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('qa_generator')

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    log_file = os.path.join(log_dir, f"qa_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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


def get_random_question_type(config):
    """根据配置的比例随机选择题型"""
    question_types = config.get("QUESTION_TYPES", {
        "multiple_choice": 0.4,
        "fill_blank": 0.2,
        "true_false": 0.2,
        "short_answer": 0.2
    })

    types = list(question_types.keys())
    weights = list(question_types.values())

    # 随机选择一个题型
    return random.choices(types, weights=weights, k=1)[0]


def get_random_topic(config):
    """从配置的主题列表中随机选择一个主题"""
    topics = config.get("TOPICS", ["计算机科学", "人工智能", "数据科学"])
    return random.choice(topics)


def get_random_difficulty(config):
    """根据配置的比例随机选择难度"""
    difficulty_levels = config.get("DIFFICULTY_LEVELS", {
        "easy": 0.3,
        "medium": 0.5,
        "hard": 0.2
    })

    levels = list(difficulty_levels.keys())
    weights = list(difficulty_levels.values())

    # 随机选择一个难度
    return random.choices(levels, weights=weights, k=1)[0]


def get_prompt_for_qa_generation(existing_qa_pairs=None, question_type=None, topic=None, difficulty=None) -> List[Dict]:
    """
    生成不同类型问答对的提示。

    Args:
        existing_qa_pairs: 已有的问答对，用于避免重复
        question_type: 问题类型 (multiple_choice, fill_blank, true_false, short_answer)
        topic: 主题
        difficulty: 难度级别 (easy, medium, hard)

    Returns:
        包含系统提示和用户提示的消息列表
    """
    system_prompt = """你是一个专业的教育内容生成助手，擅长创建各种类型的高质量教育测试题目。你需要根据要求生成指定类型、主题和难度的考试题目。"""

    # 准备已有问题的示例，避免重复
    existing_qa_text = ""
    if existing_qa_pairs:
        sample_size = min(3, len(existing_qa_pairs))
        samples = random.sample(existing_qa_pairs, sample_size)
        existing_qa_text = "\n已生成的问题示例（请避免重复）:\n"
        for i, qa in enumerate(samples):
            existing_qa_text += f"问题 {i + 1}: {qa.get('question', '')}\n"

    # 根据题型准备不同的提示
    type_prompts = {
        "multiple_choice": """
【选择题】需要包含:
1. 清晰的问题描述
2. 4个选项(A、B、C、D)
3. 正确答案
4. 答案解析，解释为什么正确选项是正确的，其他选项为什么不正确

格式示例:
{
    "type": "multiple_choice",
    "question": "选择题：什么是Python中的列表推导式？",
    "options": {
        "A": "一种创建函数的简写方式",
        "B": "一种使用简洁语法从可迭代对象创建列表的方法",
        "C": "Python的一种内置数据结构",
        "D": "一种用于遍历列表的循环结构"
    },
    "answer": "B",
    "explanation": "列表推导式(List Comprehension)是Python中一种简洁创建列表的方法，它使用简单的表达式从可迭代对象生成列表，而不需要使用传统的循环和append()方法。选项A描述的是lambda函数，选项C描述的是列表本身而非列表推导式，选项D描述的是for循环。"
}
""",

        "fill_blank": """
【填空题】需要包含:
1. 包含空缺部分的句子或段落，使用"____"表示空缺处
2. 正确答案
3. 答案解析，解释为什么这是正确答案

格式示例:
{
    "type": "fill_blank",
    "question": "填空题：在Python中，____ 函数用于将字符串转换为整数。",
    "answer": "int",
    "explanation": "int()是Python的内置函数，用于将字符串、浮点数等转换为整数类型。例如int('123')会返回整数123。"
}
""",

        "true_false": """
【判断题】需要包含:
1. 一个陈述句
2. 正确答案(true或false)
3. 答案解析，解释为什么这个陈述是对的或错的

格式示例:
{
    "type": "true_false",
    "question": "判断题：Python中的字典是无序集合。",
    "answer": false,
    "explanation": "这个陈述是错误的。在Python 3.7及以后的版本中，字典是有序集合，会保留元素插入的顺序。在此之前的版本中，字典确实是无序的。"
}
""",

        "short_answer": """
【简答题】需要包含:
1. 一个开放性问题
2. 正确答案(简要但全面的回答)
3. 答案要点(列出回答中应包含的关键点)

格式示例:
{
    "type": "short_answer",
    "question": "简答题：解释什么是RESTful API以及它的主要特点。",
    "answer": "RESTful API是一种基于REST(Representational State Transfer)架构风格的应用程序接口。它使用HTTP请求来执行CRUD(创建、读取、更新、删除)操作。RESTful API的主要特点包括：无状态性、统一接口、可缓存性、分层系统和按需代码。它通常使用JSON或XML格式传输数据，并使用标准HTTP方法如GET、POST、PUT和DELETE来操作资源。",
    "key_points": [
        "基于REST架构风格",
        "使用HTTP请求执行CRUD操作",
        "具有无状态性",
        "提供统一接口",
        "支持缓存",
        "采用分层系统架构",
        "使用JSON或XML传输数据",
        "使用标准HTTP方法操作资源"
    ]
}
"""
    }

    # 获取题型特定的提示
    type_text = type_prompts.get(question_type, type_prompts["multiple_choice"])

    # 添加难度和主题信息
    topic_text = f"主题: {topic}\n" if topic else ""
    difficulty_text = f"难度: {difficulty}\n" if difficulty else ""

    user_prompt = f"""请生成5个高质量的{question_type_names.get(question_type, "选择题")}。

{topic_text}{difficulty_text}
{type_text}
{existing_qa_text}

请直接返回JSON数组，不要用代码块包围，格式如下：
[
    {{
        "type": "{question_type}",
        "question": "问题内容",
        ...其他字段根据题型不同而不同...
    }},
    ...共5个题目
]

请确保:
1. 生成的问题在主题和难度上符合要求
2. 问题具有教育价值且事实准确
3. 问题多样化，覆盖主题的不同方面
4. 不与已有问题重复
5. 严格按照示例中的JSON格式输出
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


# 定义中文的题型名称映射
question_type_names = {
    "multiple_choice": "选择题",
    "fill_blank": "填空题",
    "true_false": "判断题",
    "short_answer": "简答题"
}


async def generate_qa_pairs(semaphore: asyncio.Semaphore,
                            existing_qa_pairs: List[Dict] = None,
                            batch_id: int = 0) -> List[Dict]:
    """生成问答对的异步函数"""
    logger = logging.getLogger('qa_generator')
    config = load_config()

    # 随机选择题型、主题和难度
    question_type = get_random_question_type(config)
    topic = get_random_topic(config)
    difficulty = get_random_difficulty(config)

    logger.info(f"批次 {batch_id}: 生成{question_type_names.get(question_type, '未知')}，主题:{topic}，难度:{difficulty}")

    messages = get_prompt_for_qa_generation(existing_qa_pairs, question_type, topic, difficulty)

    async with semaphore:
        try:
            client = AsyncOpenAI(
                api_key=config['api_key'],
                base_url=config['base_url']
            )

            for attempt in range(config["MAX_RETRIES"]):
                try:
                    completion = await client.chat.completions.create(
                        model=config["model"],
                        messages=messages,
                        temperature=0.7
                    )

                    try:
                        content = completion.choices[0].message.content
                        # 检查返回内容是否被代码块包围，如果是则提取其中的JSON
                        if content.startswith("```json") and content.endswith("```"):
                            content = content[7:-3].strip()

                        qa_pairs = json.loads(content)
                        logger.info(
                            f"批次 {batch_id}: 成功生成 {len(qa_pairs)} 个{question_type_names.get(question_type, '未知')}")

                        # 为每个问答对添加元数据
                        for qa in qa_pairs:
                            # if "type" not in qa:
                            #     qa["type"] = question_type
                            qa["topic"] = topic
                            qa["difficulty"] = difficulty
                            # qa["generation_time"] = datetime.now().isoformat()
                            # qa["batch_id"] = batch_id

                        return qa_pairs
                    except json.JSONDecodeError:
                        logger.error(f"批次 {batch_id}: 解析问答对JSON失败")
                        logger.debug(f"模型返回内容: {completion.choices[0].message.content}")
                        if attempt < config["MAX_RETRIES"] - 1:
                            logger.info(f"批次 {batch_id}: 重试中 ({attempt + 1}/{config['MAX_RETRIES']})")
                            await asyncio.sleep(config["RETRY_DELAY"])
                        else:
                            return []

                except Exception as e:
                    logger.error(f"批次 {batch_id}: API调用出错: {str(e)}")
                    if attempt < config["MAX_RETRIES"] - 1:
                        wait_time = config["RETRY_DELAY"] * (2 ** attempt)  # 指数退避策略
                        logger.info(
                            f"批次 {batch_id}: 等待 {wait_time} 秒后重试 ({attempt + 1}/{config['MAX_RETRIES']})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"批次 {batch_id}: 达到最大重试次数，放弃请求")
                        return []

        except Exception as e:
            logger.error(f"批次 {batch_id}: 生成问答对时出错: {str(e)}")
            return []


def save_qa_pairs(qa_pairs: List[Dict], output_dir: str, filename: str = None):
    """保存问答对到指定目录"""
    os.makedirs(output_dir, exist_ok=True)

    # 如果未指定文件名，使用时间戳创建新文件
    if filename is None:
        filename = f"qa_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    file_path = os.path.join(output_dir, filename)

    # 检查是否存在现有文件并合并
    existing_pairs = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_pairs = json.load(f)
        except Exception as e:
            print(f"读取现有问答对文件失败: {str(e)}")

    # 合并问答对
    all_pairs = existing_pairs + qa_pairs

    # 保存合并后的问答对
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    return len(all_pairs)


def generate_statistics(qa_pairs: List[Dict]):
    """生成问答对的统计信息"""
    stats = {
        "total": len(qa_pairs),
        "by_type": {},
        "by_topic": {},
        "by_difficulty": {}
    }

    # 按类型统计
    for qa in qa_pairs:
        qa_type = qa.get("type", "unknown")
        stats["by_type"][qa_type] = stats["by_type"].get(qa_type, 0) + 1

        topic = qa.get("topic", "unknown")
        stats["by_topic"][topic] = stats["by_topic"].get(topic, 0) + 1

        difficulty = qa.get("difficulty", "unknown")
        stats["by_difficulty"][difficulty] = stats["by_difficulty"].get(difficulty, 0) + 1

    return stats


async def process_qa_generation():
    """处理问答对生成的主函数"""
    # 加载配置
    config = load_config()
    validation_error = validate_config(config)
    if validation_error:
        print(f"错误: {validation_error}")
        print("请在generate_config.json中填写所有必要的配置项后再运行")
        sys.exit(1)

    # 设置日志记录
    logger = setup_logging(config.get("LOG_DIR", "logs"))
    logger.info(f"开始生成问答对，目标数量: {config['TARGET_QA_COUNT']}")

    # 创建输出目录
    output_dir = config.get("OUTPUT_DIR", "generated_qa")
    os.makedirs(output_dir, exist_ok=True)

    # 创建新的问答对文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    qa_filename = f"qa_pairs_{timestamp}.json"
    qa_file_path = os.path.join(output_dir, qa_filename)

    # 初始化空的问答对列表
    all_qa_pairs = []
    save_qa_pairs(all_qa_pairs, output_dir, qa_filename)

    logger.info(f"将创建新的问答对文件: {qa_file_path}")

    # 计算需要生成的问答对数量
    target_count = config["TARGET_QA_COUNT"]
    remaining_count = target_count

    # 创建并发控制信号量
    semaphore = asyncio.Semaphore(config.get("MAX_CONCURRENT_CALLS", 5))

    # 计算需要的批次数
    qa_per_request = config.get("QA_PER_REQUEST", 5)
    required_batches = (remaining_count + qa_per_request - 1) // qa_per_request  # 向上取整

    logger.info(f"需要生成 {remaining_count} 个问答对，计划分 {required_batches} 个批次")

    # 批量处理，每批次生成一组问答对
    current_qa_count = 0
    batch_size = config.get("BATCH_SIZE", 8)

    batch_id = 0
    while current_qa_count < target_count:
        # 确定当前批次的大小
        current_batch_size = min(batch_size, (target_count - current_qa_count + qa_per_request - 1) // qa_per_request)

        logger.info(
            f"处理批次 {batch_id + 1} 到 {batch_id + current_batch_size} (总进度: {current_qa_count}/{target_count})")

        # 创建当前批次的任务
        tasks = []
        for i in range(current_batch_size):
            task = generate_qa_pairs(
                semaphore,
                all_qa_pairs,
                batch_id + i + 1
            )
            tasks.append(task)

        # 并行执行当前批次的任务
        batch_results = await asyncio.gather(*tasks)

        # 处理批次结果
        new_pairs = []
        for result in batch_results:
            if result:
                new_pairs.extend(result)

        # 更新计数并保存结果
        current_qa_count += len(new_pairs)
        batch_id += current_batch_size

        # 如果生成了新的问答对，保存到文件
        if new_pairs:
            all_qa_pairs.extend(new_pairs)
            total_saved = save_qa_pairs(all_qa_pairs, output_dir, qa_filename)
            logger.info(f"已保存 {len(new_pairs)} 个新问答对，当前总计: {total_saved}/{target_count}")

            # 定期打印统计信息
            if len(all_qa_pairs) % 50 == 0 or len(all_qa_pairs) >= target_count:
                stats = generate_statistics(all_qa_pairs)
                logger.info(
                    f"当前统计: 总计{stats['total']}题，按类型: {stats['by_type']}，按难度: {stats['by_difficulty']}")
        else:
            # 如果当前批次没有生成任何问答对，等待一段时间后继续
            logger.warning("当前批次未能生成问答对，等待后继续...")
            await asyncio.sleep(config.get("FAIL_WAIT_TIME", 5))

        # 检查是否已达到目标数量
        if current_qa_count >= target_count:
            logger.info(f"已达到目标数量 ({current_qa_count}/{target_count})，停止生成")
            break

    # 生成最终统计信息
    final_stats = generate_statistics(all_qa_pairs)
    logger.info(f"问答对生成完成，总计: {len(all_qa_pairs)} 个问答对")
    logger.info(f"最终统计: 按类型: {final_stats['by_type']}")
    logger.info(f"按主题: {final_stats['by_topic']}")
    logger.info(f"按难度: {final_stats['by_difficulty']}")

    # 保存统计信息
    stats_file = os.path.join(output_dir, f"statistics_{timestamp}.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    return all_qa_pairs, final_stats


async def main_async():
    """主异步函数"""
    try:
        qa_pairs, stats = await process_qa_generation()
        print(f"\n成功生成 {len(qa_pairs)} 个问答对")
        print(f"问题类型分布: {stats['by_type']}")
        print(f"难度分布: {stats['by_difficulty']}")
    except Exception as e:
        print(f"生成过程中发生错误: {str(e)}")


def main():
    """主函数"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()