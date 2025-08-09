# Audio-Spectrum-Visualizer - 音频频谱可视化生成器

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

将音频文件转换为带有动态频谱可视化的视频文件。支持多种音频格式，可自定义频谱参数和视频分辨率。

![软件界面截图](https://github.com/LoyeJun/Audio-Spectrum-Visualizer/blob/main/PixPin_2025-08-09_10-40-18.png)


## 功能特点

- 🎵 支持多种音频格式：WAV, FLAC, MP3, OGG, M4A
- ⚙️ 可自定义参数：
  - 目标帧率 (10-120 FPS)
  - 频谱条数 (8-1024 bins)
  - 视频分辨率 (预设或自定义)
  - 频谱平滑效果
- 📊 实时显示处理进度和性能信息：
  - 已用时间/剩余时间
  - 实时帧率
  - 导出倍速
- 📂 自动处理文件名中的非法字符
- ⚡ 自动下载 FFmpeg (系统未安装时)

## 安装与使用

### 依赖安装

```bash
pip install -r requirements.txt
```

### 运行程序

```bash
python Audio-to-Video.py
```

### 使用步骤

1. 点击"选择音频文件"按钮，选择要处理的音频文件
2. 调整编码参数（帧率、频谱条数、分辨率等）
3. 设置输出文件名
4. 点击"开始导出"按钮
5. 等待处理完成，生成的视频将保存在当前目录

## 技术细节

- **音频处理**：使用 `soundfile` 库读取音频数据
- **频谱计算**：基于 FFT（快速傅里叶变换）算法
- **视频编码**：使用 FFmpeg 进行高效视频编码
- **GUI框架**：基于 PyQt6 构建用户界面

## 常见问题

### 处理速度慢怎么办？
- 降低目标帧率
- 减少频谱条数
- 选择较低的分辨率

### 如何提高频谱质量？
- 增加频谱条数
- 启用频谱平滑选项
- 使用高质量源音频文件

## 贡献指南

欢迎贡献代码！请遵循以下步骤：
1. Fork 本仓库
2. 创建新分支 (`git checkout -b feature/your-feature`)
3. 提交修改 (`git commit -am 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建 Pull Request

## 许可证
本项目采用 [MIT 许可证](LICENSE) - 详情请见 LICENSE 文件。

## 文件结构建议
```
AudioSpectrumVisualizer/
├── Audio-to-Video.py       # 主程序
├── README.md               # 项目文档
├── requirements.txt        # Python依赖列表
├── LICENSE                 # 许可证文件
└── screenshot.png          # 软件界面截图
```
