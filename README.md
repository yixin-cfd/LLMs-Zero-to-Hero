<p align="center">
<h1 align="center">《LLMs-Zero-to-Hero》</h1>
<h4 align="right">从大模型无名小卒到LLM大师</h4>
</p>

开个新坑，从无名小卒到大模型（LLM）大英雄~ 欢迎关注[B站后续更新](https://space.bilibili.com/12420432)！！！

## 特点
- 完全从零手写，边写边讲知识点，致敬 Andrej Karpathy
- 体系化，具有完整的实践路线
- 配套视频讲解，[B站视频](https://www.bilibili.com/video/BV1qWwke5E3K)
- 配套镜像 GPU，用于模型的训练，有演示和展示 Demo
- 最小使用 3090，4090 即可训练~

> 大家可以用我的 [AIStackDC 注册链接](https://aistackdc.com/phone-register?invite_code=D872A9)获得 GPU 优惠券。

## 目录
- 大模型基础，介绍大模型训练的流程
    - [Dense Model](https://github.com/bbruceyuan/LLMs-Zero-to-Hero/blob/master/src/video/build_gpt.ipynb) （[B站视频](https://www.bilibili.com/video/BV1qWwke5E3K)）
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

> 如果本套教程对你有难度，可以看看 [Hands-On Large Language Models CN(ZH) -- 动手学大模型](https://github.com/bbruceyuan/Hands-On-Large-Language-Models-CN)

## 已更新内容目录
| 章节 | 文章解读 | 中文 Notebook<br/>复制后可直接运行| 视频讲解 <br/> (可点击)|
|---|---|------|------|
| 完全从零手写一个nanoGPT | todo | [![中文可运行 Notebook](https://img.shields.io/badge/notebook-代码-pink)](https://github.com/bbruceyuan/LLMs-Zero-to-Hero/blob/master/src/video/build_gpt.ipynb) | [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1qWwke5E3K)](https://www.bilibili.com/video/BV1qWwke5E3K/)<br />[![Youtube](https://img.shields.io/youtube/views/2g5-aHYWiio)](https://www.youtube.com/watch?v=2g5-aHYWiio) |
| 激活函数优化| [LLM activate function激活函数的进化之路，从 ReLU，GELU 到 swishGLU](https://bruceyuan.com/llms-zero-to-hero/activate-function-from-relu-gelu-to-swishglu.html) | todo | todo | 
| LLM MOE 的进化之路 | [LLM MOE的进化之路，从普通简化 MOE，到 spare_moe，再到 deepseek 使用的 share_xpert_spare_moe](https://bruceyuan.com/llms-zero-to-hero/the-way-of-moe-model-evolution.html) | [![中文可运行 Notebook](https://img.shields.io/badge/notebook-代码-pink)](https://github.com/bbruceyuan/LLMs-Zero-to-Hero/blob/master/src/video/build_moe_model.ipynb) | todo | 


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



> 最后欢迎大家使用 [AIStackDC](https://aistackdc.com/phone-register?invite_code=D872A9) 算力平台，主打一个便宜方便（有专门的客服支持），如果你需要的话可以使用我的邀请链接: [https://aistackdc.com/phone-register?invite_code=D872A9](https://aistackdc.com/phone-register?invite_code=D872A9)