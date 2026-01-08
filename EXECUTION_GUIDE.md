# 执行指南

## 在新会话中开始执行

### 1. 打开新终端会话

```bash
cd "/Users/hao/Library/Mobile Documents/iCloud~md~obsidian/Documents/ios/领域/核心文献图表-生成式人工智能网页赋能跨学科实践/tts-audiobook-generator"
```

### 2. 启动 Claude Code 并使用 executing-plans 技能

在 Claude Code 中输入：

```
使用 superpowers:executing-plans 来执行 docs/plans/2025-01-08-tts-audiobook-generator.md 中的实现计划
```

### 3. 执行流程

- executing-plans 技能会逐任务执行计划
- 每个任务会：写测试 -> 运行 -> 实现 -> 验证 -> 提交
- 完成后自动推送到 GitHub

### 4. 关键检查点

检查点1（任务1-3完成后）: 基础模块就绪
检查点2（任务4-6完成后）: 核心功能就绪  
检查点3（任务7-10完成后）: 完整功能就绪

### 5. 配置API密钥

执行完成后，创建配置文件：

```bash
cp config.yaml.example config.yaml
# 编辑 config.yaml 填入你的阿里云 API 密钥
```

## 仓库地址

https://github.com/haomazeng/tts-audiobook-generator
