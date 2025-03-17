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