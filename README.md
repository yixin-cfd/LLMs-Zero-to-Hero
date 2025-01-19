# LLMs-Zero-to-Hero
开个新坑，从无名小卒到大模型（LLM）大英雄~ 欢迎关注[B站后续更新](https://space.bilibili.com/12420432)！！！

## 目录
- 大模型基础，介绍大模型训练的流程
    - Dense Model （完成从零手写：代码在 /src/video/build_gpt.ipynb 中）
    - MOE Model
    - ...
- 完全从零到一训练 LLM (Pre-Training)
- 完全从零到一微调 LLM (Supervised Fine-Tuning, SFT)
- 完全从零到一微调 LLM (Direct Preference Optimization, DPO)
- 完全从零到一微调 LLM (Reinforcement Learning from Human Feedback, RLHF)
- 用于写 Python 代码的 Code-LLM
- 大模型的部署
    - 推理优化，量化等
- ...

## 代码仓库结构
```
├── chapter01   # 不同章节的学习笔记，最终会形成一本书籍
│   ├── README.md
│   ├── ...
├── chapter02
│   ├── README.md
│   ├── train.py
│   ├── ...
├── src/
│   ├── hero/  # 最终自研实现的大模型等会放到这个地方；
│   ├── chapter01/  # 这里会存放 chapter01 的代码；
│   ├── chapter02/  # 这里会存放 chapter02 的代码；
│   ├── video/  # 录制视频的时候用到的代码；
├── README.md
```


陆续会更新，欢迎关注！！！
- 方式 1：可以加我 wx: bbruceyuan 来群里催更～ 
- 方式 2：关注我的博客：[chaofa用代码打点酱油](https://www.bbruceyuan.com/)
- 方式 3： 关注我的公众号: [chafa用代码打点酱油](https://mp.weixin.qq.com/s/WxLbKvW4_9g0ajQ0wGRruQ)