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
            "TARGET_QA_COUNT": 10,  # 目标问答对数量
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
            "ACTIVE_TOPICS": [],  # 当前活动的主题列表（由命令行参数设置）
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


def update_config_topics(topics: List[str]) -> None:
    """更新配置文件中的活动主题列表"""
    config_path = os.path.join(os.path.dirname(__file__), "generate_config.json")

    try:
        # 加载现有配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 更新活动主题
        config["ACTIVE_TOPICS"] = topics

        # 保存更新后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        print(f"已更新配置文件中的活动主题为: {', '.join(topics)}")

    except Exception as e:
        print(f"更新配置文件时出错: {str(e)}")
        sys.exit(1)


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


def get_topic_from_config(config):
    """从配置的活动主题列表中选择一个主题，如果没有活动主题则从所有主题中选择"""
    # 首先检查是否有活动主题列表
    active_topics = config.get("ACTIVE_TOPICS", [])

    if active_topics:
        # 如果有活动主题，从中随机选择一个
        return random.choice(active_topics)
    else:
        # 否则从所有主题中随机选择
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
1. 生成的问题在主题"{topic}"相关的领域内，确保题目内容严格围绕这个主题
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


async def generate_qa_pairs(semaphore=None, existing_qa_pairs=None, batch_id=0, config=None):
    """
    生成问答对的异步函数

    参数:
        semaphore (asyncio.Semaphore): 并发控制信号量
        existing_qa_pairs (List[Dict]): 已存在的问答对列表(用于去重)
        batch_id (int): 当前批次ID
        config (Dict): 生成配置

    返回:
        List[Dict]: 生成的问答对列表
    """
    logger = logging.getLogger('qa_generator')

    # 加载配置(如果未传入)
    config = config or load_config()

    try:
        # 随机选择题型、主题和难度
        question_type = get_random_question_type(config)
        topic = get_topic_from_config(config)
        difficulty = get_random_difficulty(config)

        logger.info(
            f"批次 {batch_id}: 生成{question_type_names.get(question_type, '未知')}，主题:{topic}，难度:{difficulty}")

        # 准备生成提示
        messages = get_prompt_for_qa_generation(
            existing_qa_pairs=existing_qa_pairs,
            question_type=question_type,
            topic=topic,
            difficulty=difficulty
        )

        # 如果没有传入信号量，创建一个虚拟信号量
        semaphore = semaphore or asyncio.Semaphore(1)

        async with semaphore:
            for attempt in range(config.get("MAX_RETRIES", 3)):
                try:
                    # 创建API客户端
                    client = AsyncOpenAI(
                        api_key=config['api_key'],
                        base_url=config['base_url']
                    )

                    # 调用API生成题目
                    completion = await client.chat.completions.create(
                        model=config["model"],
                        messages=messages,
                        temperature=0.7
                    )

                    # 处理返回内容
                    content = completion.choices[0].message.content

                    # 检查是否被代码块包围
                    if content.startswith("```json") and content.endswith("```"):
                        content = content[7:-3].strip()

                    # 解析JSON
                    qa_pairs = json.loads(content)

                    # 添加元数据
                    for qa in qa_pairs:
                        qa["type"] = qa.get("type", question_type)
                        qa["topic"] = topic
                        qa["difficulty"] = difficulty
                        qa["generation_time"] = datetime.now().isoformat()
                        qa["batch_id"] = batch_id

                    logger.info(
                        f"批次 {batch_id}: 成功生成 {len(qa_pairs)} 个{question_type_names.get(question_type, '未知')}")
                    return qa_pairs

                except json.JSONDecodeError as e:
                    logger.error(f"批次 {batch_id}: 解析JSON失败 (尝试 {attempt + 1}/{config['MAX_RETRIES']})")
                    logger.debug(f"原始返回内容: {content}")
                    if attempt < config["MAX_RETRIES"] - 1:
                        await asyncio.sleep(config["RETRY_DELAY"])
                    continue

                except Exception as e:
                    logger.error(f"批次 {batch_id}: API调用出错: {str(e)} (尝试 {attempt + 1}/{config['MAX_RETRIES']})")
                    if attempt < config["MAX_RETRIES"] - 1:
                        wait_time = config["RETRY_DELAY"] * (2 ** attempt)  # 指数退避
                        await asyncio.sleep(wait_time)
                    continue

            logger.error(f"批次 {batch_id}: 达到最大重试次数，放弃请求")
            return []

    except Exception as e:
        logger.error(f"批次 {batch_id}: 生成问答对时发生未捕获错误: {str(e)}")
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


async def process_qa_generation(config=None):
    """
    处理问答对生成的主异步函数

    参数:
        config (Dict): 生成配置(可选)

    返回:
        Tuple[List[Dict], Dict, str]:
            - 生成的问答对列表
            - 统计信息字典
            - 输出文件路径
    """
    # 加载配置(如果未传入)
    config = config or load_config()

    # 验证配置
    validation_error = validate_config(config)
    if validation_error:
        logger.error(f"配置验证失败: {validation_error}")
        raise ValueError(f"无效配置: {validation_error}")

    # 初始化日志
    logger = setup_logging(config.get("LOG_DIR", "logs"))

    # 获取活动主题
    active_topics = config.get("ACTIVE_TOPICS", [])
    if active_topics:
        logger.info(f"使用指定主题: {', '.join(active_topics)}")
    else:
        logger.info("未指定特定主题，将从全部主题中随机选择")

    target_count = config["TARGET_QA_COUNT"]
    logger.info(f"开始生成问答对，目标数量: {target_count}")

    # 准备输出目录
    output_dir = config.get("OUTPUT_DIR", "generated_qa")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    qa_filename = f"qa_pairs_{timestamp}.json"
    qa_filepath = os.path.join(output_dir, qa_filename)

    # 初始化数据存储
    all_qa_pairs = []
    save_qa_pairs(all_qa_pairs, output_dir, qa_filename)
    logger.info(f"输出文件: {qa_filepath}")

    # 并发控制
    semaphore = asyncio.Semaphore(config.get("MAX_CONCURRENT_CALLS", 5))
    qa_per_request = config.get("QA_PER_REQUEST", 5)
    required_batches = (target_count + qa_per_request - 1) // qa_per_request

    logger.info(f"计划分 {required_batches} 个批次生成，每批 {qa_per_request} 题")

    # 批量生成
    current_count = 0
    batch_size = config.get("BATCH_SIZE", 8)
    batch_id = 0

    while current_count < target_count:
        # 计算当前批次大小
        current_batch_size = min(
            batch_size,
            (target_count - current_count + qa_per_request - 1) // qa_per_request
        )

        logger.info(
            f"处理批次 {batch_id + 1}-{batch_id + current_batch_size} "
            f"(进度: {current_count}/{target_count})"
        )

        # 创建并执行批次任务
        tasks = [
            generate_qa_pairs(
                semaphore=semaphore,
                existing_qa_pairs=all_qa_pairs,
                batch_id=batch_id + i + 1,
                config=config
            )
            for i in range(current_batch_size)
        ]

        batch_results = await asyncio.gather(*tasks)

        # 处理结果
        new_pairs = []
        for result in batch_results:
            if result:
                new_pairs.extend(result)

        # 更新状态
        current_count += len(new_pairs)
        batch_id += current_batch_size

        if new_pairs:
            all_qa_pairs.extend(new_pairs)
            total_saved = save_qa_pairs(all_qa_pairs, output_dir, qa_filename)

            logger.info(
                f"已保存 {len(new_pairs)} 题，总计: {total_saved}/{target_count} "
                f"(成功率: {total_saved / (batch_id * qa_per_request):.1%})"
            )

            # 定期打印统计信息
            if len(all_qa_pairs) % 50 == 0 or len(all_qa_pairs) >= target_count:
                stats = generate_statistics(all_qa_pairs)
                logger.info(
                    f"当前统计: 总计{stats['total']}题 | "
                    f"题型: {stats['by_type']} | "
                    f"难度: {stats['by_difficulty']}"
                )
        else:
            logger.warning("当前批次未生成有效题目，等待后重试...")
            await asyncio.sleep(config.get("FAIL_WAIT_TIME", 5))

        # 完成检查
        if current_count >= target_count:
            logger.info(f"完成生成，实际生成 {len(all_qa_pairs)} 题")
            break

    # 生成最终统计
    final_stats = generate_statistics(all_qa_pairs)
    stats_filename = f"statistics_{timestamp}.json"
    stats_filepath = os.path.join(output_dir, stats_filename)

    with open(stats_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    logger.info(f"统计信息已保存: {stats_filepath}")
    logger.info("=== 生成过程完成 ===")

    return all_qa_pairs, final_stats, qa_filepath



def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成各种类型的教育测试题目')
    parser.add_argument('topics', nargs='*', help='要生成题目的主题，例如"人工智能 大数据技术"')
    parser.add_argument('-c', '--count', type=int, help='要生成的题目数量，默认使用配置文件中的设置')
    parser.add_argument('-o', '--output-dir', help='输出目录，默认使用配置文件中的设置')
    parser.add_argument('-i', '--info', action='store_true', help='显示当前配置信息')

    return parser.parse_args()


async def main_async():
    """主异步函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()

        # 加载配置
        config = load_config()

        # 显示配置信息并退出
        if args.info:
            print("\n当前配置信息:")
            print(f"API模型: {config.get('model', '未设置')}")
            print(f"目标题目数量: {config.get('TARGET_QA_COUNT', 500)}")
            print(f"并发调用数: {config.get('MAX_CONCURRENT_CALLS', 5)}")
            print(f"每批大小: {config.get('BATCH_SIZE', 8)}")
            print(f"输出目录: {config.get('OUTPUT_DIR', 'generated_qa')}")
            print("\n题型分布:")
            for qtype, weight in config.get('QUESTION_TYPES', {}).items():
                print(f"  {question_type_names.get(qtype, qtype)}: {weight * 100:.1f}%")
            print("\n难度分布:")
            for level, weight in config.get('DIFFICULTY_LEVELS', {}).items():
                print(f"  {level}: {weight * 100:.1f}%")
            print("\n可用主题:")
            for topic in config.get('TOPICS', []):
                print(f"  - {topic}")
            if config.get('ACTIVE_TOPICS'):
                print("\n当前活动主题:")
                for topic in config.get('ACTIVE_TOPICS', []):
                    print(f"  - {topic}")
            sys.exit(0)

        # 更新配置中的题目数量
        if args.count:
            config["TARGET_QA_COUNT"] = args.count
            with open(os.path.join(os.path.dirname(__file__), "generate_config.json"), 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"已将目标题目数量设置为 {args.count}")

        # 更新配置中的输出目录
        if args.output_dir:
            config["OUTPUT_DIR"] = args.output_dir
            with open(os.path.join(os.path.dirname(__file__), "generate_config.json"), 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"已将输出目录设置为 {args.output_dir}")

        # 处理主题参数
        if args.topics:
            # 更新配置中的活动主题
            update_config_topics(args.topics)

        # 运行生成过程
        qa_pairs, stats, output_file = await process_qa_generation()
        print(f"\n成功生成 {len(qa_pairs)} 个问答对")
        print(f"问题类型分布: {stats['by_type']}")
        print(f"难度分布: {stats['by_difficulty']}")
        print(f"输出文件: {output_file}")
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