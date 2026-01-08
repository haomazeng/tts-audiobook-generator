# TTS Audiobook Generator

基于阿里云百炼 CosyVoice 的文本转语音工具，支持批量将 PDF 和 Markdown 文件转换为 MP3 音频。

## 功能特性

- 支持 PDF 和 Markdown 文件
- 批量转换整个目录
- 断点续传 - 中断后可继续
- 并发处理 - 加速转换
- 进度显示 - 实时查看进度
- 专业播音风格声音

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

复制配置模板：

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`，填入你的阿里云 API 密钥：

```yaml
aliyun:
  api_key: "your_api_key_here"
```

或使用环境变量：

```bash
export ALIYUN_API_KEY="your_api_key_here"
```

### 3. 运行转换

转换单个文件：

```bash
python tts_tool.py --input document.pdf --output ./audiobooks
```

批量转换目录：

```bash
python tts_tool.py --input ./documents --output ./audiobooks
```

只转换 PDF：

```bash
python tts_tool.py --input ./documents --format pdf
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input`, `-i` | 输入文件或目录（必需） | - |
| `--output`, `-o` | 输出目录 | `./audiobooks` |
| `--config`, `-c` | 配置文件路径 | `config.yaml` |
| `--format` | 文件格式 (pdf/md/all) | `all` |
| `--voice` | 声音类型 | `zhixiaobai` |
| `--no-resume` | 不使用断点续传 | false |
| `--verbose`, `-v` | 详细日志 | false |

## 声音类型

- `zhixiaobai` - 知小百（专业播音，推荐）
- `longwan` - 龙万（男声）
- `zhichu` - 知楚（女声）
- `aiqi` - 爱奇（儿童）
- `zhichu_v2` - 知楚 v2（增强版）

## 断点续传

转换过程中的进度会自动保存。如果中断，重新运行相同的命令会从断点继续：

```bash
# 第一次运行（中断）
python tts_tool.py --input document.pdf

# 继续运行（从断点继续）
python tts_tool.py --input document.pdf
```

如需重新开始，使用 `--no-resume`：

```bash
python tts_tool.py --input document.pdf --no-resume
```

## 开发

运行测试：

```bash
pytest tests/ -v
```

## 项目结构

```
tts-audiobook-generator/
├── src/
│   ├── __init__.py
│   ├── audio_processor.py    # 音频处理模块
│   ├── cli.py                # CLI 界面
│   ├── config.py             # 配置管理
│   ├── converter.py          # 主转换引擎
│   ├── progress.py           # 进度跟踪
│   ├── text_extractor.py     # 文本提取
│   └── tts_client.py         # TTS API 客户端
├── tests/                    # 测试文件
├── config.yaml.example       # 配置模板
├── requirements.txt          # 依赖列表
└── tts_tool.py               # 入口脚本
```

## License

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
