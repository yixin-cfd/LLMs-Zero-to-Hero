# LLMs-Zero-to-Hero
开个新坑，从无名小卒到大模型（LLM）大英雄~ 欢迎关注[B站后续更新](https://space.bilibili.com/12420432)！！！

## 特点
- 完全从零手写，边写边讲知识点，致敬 Andrej Karpathy
- 体系化，具有完整的实践路线
- 配套视频讲解，[B站视频](https://www.bilibili.com/video/BV1qWwke5E3K)

## 目录
- 大模型基础，介绍大模型训练的流程
    - Dense Model （完成从零手写 Build a nanoGPT from Scratch：代码在 /src/video/build_gpt.ipynb 中，对应 [B站视频](https://www.bilibili.com/video/BV1qWwke5E3K)）
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
- 方式 1：可以加我 wx: bbruceyuan 来群里催更或者**反馈问题**～ 
- 方式 2：关注我的博客：[chaofa用代码打点酱油](https://www.bbruceyuan.com/)
- 方式 3： 关注我的公众号: [chafa用代码打点酱油](https://mp.weixin.qq.com/s/WxLbKvW4_9g0ajQ0wGRruQ)



> 最后欢迎大家使用 [AIStackDC](https://aistackdc.com/phone-register?invite_code=D872A9) 算力平台，主打一个便宜方便，如果你需要的话可以使用我的邀请链接: [https://aistackdc.com/phone-register?invite_code=D872A9](https://aistackdc.com/phone-register?invite_code=D872A9)