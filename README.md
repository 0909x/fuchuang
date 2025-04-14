# fuchuang

### 问答对生成模块

generate.py支持用户命令行输入多主题，使用前请在generate_config.json中补充api调用信息。

```cmd
cd generation
python generate.py topic1 topic2 ... -c num -o output_dir -i
```

* -c指定生成题目数量
* -o指定问答对输出目录
* -i查看当前配置信息

### 模型测试模块

test.py支持测试单个模型，使用前请在test_config.json中补充api调用信息。

```cmd
cd generation
python test.py --qa-file path/to/your/qa_pairs.json --model model_name --output-dir path/to/output --concurrent 10 --batch-size 20 
```

* qa_pairs.json的路径需手动指定或在配置中填充！
* 本模块每次调用仅支持测试一款模型，若需灵活测试多款模型，建议选择api集成平台，使用model参数直接调用不同模型，否则必须修改配置信息。
* –output-dir指定输出目录，–concurrent指定最大并发api数量，–batch-size指定一次调用处理的问题数，这些可以直接使用默认值。
* 输出的json中每个元素有model_answer与correct_answer属性，分别为测试模型的作答与正确答案，后续分析时直接提取即可。

### 链接部分
修改了generate、page、sever
已实现自定义标签、数量、题目类型
未实现题目类型多选、默认标签调用题库
