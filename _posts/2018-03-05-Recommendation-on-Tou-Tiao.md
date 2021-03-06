---
layout: post
title: 今日头条推荐技术
categories: recommendation
author: CC
tags: recommendation

---

# 今日头条推荐技术

> 如果用形式化的方式去描述，那推荐系统实际上是拟合一个用户对内容的满意程度函数：$y=F(x_i,x_u,x_c)$。显然，这个函数需要输入三个维度的变量:
>
> - 第一个维度是内容特征：对于头条这样一个综合平台来说，内容包含：图文、视频、UGC小视频、问答、微头条。每种内容都有自己的特征，通常会采用不同的技术对不同的内容提取特征
> - 第二个维度是用户特征：包括兴趣标签、职业、年龄、性别、还有其他模型刻画出的隐式的用户兴趣等
> - 第三个维度是环境特征：这是移动互联网时代推荐的特点，用户随时随地移动，在工作、通勤、旅游等不同的场景，信息偏好有所偏移。
>
> 结合三方面的维度，模型会给出一个预估。

*本文摘录自曹欢欢《今日头条算法原理》*



## 1. 系统概览

### 1.1 典型的推荐算法

- 协同过滤
- Logistic Regression
- Factorization Machine
- GBDT
- DNN+LR

一般来说，一个优秀的工业级推荐系统需要非常灵活的算法实验平台，可以支持多种推荐算法组合，包括模型结构调整。



### 1.2 典型的推荐特征

有四类特征在推荐模型中扮演着十分重要的角色

- 相关性特征：即评估内容的属性域用户是否匹配。
  - 显式的：关键词匹配、分类匹配、来源匹配、主题匹配
  - 隐式的：FM模型中的匹配、用户向量和内容向量相似度或距离
- 环境特征：地理位置、时间
- 热度特征：全局热度、分类热度、主题热度、关键词热度。这类特征在解决冷启动问题上十分有效
- 协同特征：可以部分程度上帮助解决所谓的算法越推越窄的问题。协同特征可考虑用户历史行为，也可以考虑相似用户的兴趣偏好，从而可以扩展模型的探索能力。



## 模型的训练

头条采用的是**实时训练**，主要基于strom集群实时处理样本数据，包括点击、展现、收藏、分享等动作类型。因为实时训练具有省资源、反馈快等优点。用户的行为信息可以被模型快速捕捉并反馈至下一刷的推荐效果。

> 具体地，线上服务器记录实时特征，导入到Kafka文件队列中，然后进一步导入storm集群消费kafka数据，客户端回传推荐的label构造训练样本，随后根据最新的样本进行在线训练更新模型参数，最终线上模型得到更新。整个过程的延时主要在用户的动作反馈延时，因为文章推荐给用户，用户不一定马上就看，不考虑这部分时间，整个系统几乎是实时的。



### 1.3 生成推荐内容——召回策略设计

因为头条目前的容量非常大，推荐系统不可能多所有内容进行预估。因此，设计了一些召回策略，即每次推荐时，从海量的内容中筛选出千级别的内容库。

头条的召回策略主要采用的是倒排思路，即离线维护一个倒排，这个倒排的key可以是分类、topic、实体、来源等，排序考虑热度、新鲜度、动作等。线上召回可以迅速从倒排中根据用户兴趣标签对内容进行截断，高效的从很大的内容库中筛选比较靠谱的一小部分内容。

![](https://raw.githubusercontent.com/clhchtcjj/Pit-for-Typora/master/toutiao-recall.png)



## 2 内容分析

内容分析囊括文本分析、图片分析和视频分析。这篇报告主要介绍的是文本分析，因为头条一开始主要做资讯的。

文本分析在推荐系统的应用有：

- 用户兴趣建模：给喜欢阅读“互联网”文章的用户打上“互联网”标签
- 帮助内容推荐：将“互联网”内容推荐给有“互联网”标签的用户
- 生成频道内容：“德甲”的内容进“德甲频道”

**举一个文本分析的实例：**

![](https://raw.githubusercontent.com/clhchtcjj/Pit-for-Typora/master/toutiao-nlp.png)

从上图可以看到，一篇文章包含的特征有：分类、关键词、topic、实体词等文本特征。

当然，还有一些特征也十分重要，如文本相似度、时空特征、质量特征。

*问题：隐语义特征已经很好的帮助推荐，为什么还需要人工标注标签？*

*答：有一些产品需要解释推荐内容，如频道、兴趣表达等重要产品功能需要有一个明确定义、容易理解的文本标签体系。*



### 2.1 文本层次化分类

![](https://raw.githubusercontent.com/clhchtcjj/Pit-for-Typora/master/toutiao-classifier.png)

头条采用的是典型的**层次化文本分类算法**，相比较于单独的分类器，利用层次化文本分类可以很好的解决数据倾斜问题。上述架构，每个元分类器可以是异构的。有些分类SVM效果好，有些要结合CNN，有些要结合RNN再处理一下。



## 3 用户标签

### 3.1 标签概览

头条的用户标签大致上可以分为三类：

- 兴趣标签
  - 感兴趣的类别和主题
  - 感兴趣的关键词
  - 感兴趣的来源
  - 感兴趣的用户聚类
  - 各种垂直兴趣特征（车型，体育球队，感兴趣股票）
- 身份特性
  - 年龄：基于模型预估出，分析的内容包括：机型、阅读时间分布
  - 性别：第三方账号活动
  - 常驻地点：在位置信息的基础上，通过传统的聚类的方法拿到常驻点，常驻点结合其他信息可以推测用户的工作地点、出差地点和旅游地点。
- 行为特征
  - 晚上才看视频

最简单的用户标签是用浏览过的内容标签，但是并不是所有内容标签都直接作为用户标签的，需要做一次过滤和处理：

- 过滤噪声：通过停留时间，过滤标题党
- 热点惩罚：对用户在一些热门的文章上做降权处理
- 时间衰减：随着用户动作的增加，老的特征权重会随时间衰减，新动作贡献的特征权重会更大
- 惩罚表现：如果一篇推荐给用户的文章没有被点击，相关的特征（类别、关键词、来源）权重会被惩罚
- 考虑全局背景：考虑给定特征的人均点击比例，是不是相关内容推送比较多，以及相关的关闭和dislike信号等



### 3.1 用户标签流式计算框架

![](https://raw.githubusercontent.com/clhchtcjj/Pit-for-Typora/master/toutiao-userprofile.png)

头条在计算用户标签时采用的是strom集群流式计算系统。与之前的基于hadoop的批量处理，流式计算只要用户有动作就更新标签，CPU代价小，降低了计算资源的开销。

也可以将流式计算与批量计算混合使用，如对于用户性别、年龄、常驻地点的信息可以不实时的重复计算。



## 4 评估分析

**A/B Test+人工评估**

线上实验平台只能通过数据指标变化推测用户体验的变化，但数据指标和用户体验存在差异，很多指标不能完全量化。很多改进仍然要通过人工分析，重大改进需要人工评估二次确认。 



## 5 安全机制

- 风险内容识别技术：个人觉得关键在于样本库的构造，图片、文本
- 泛低质内容识别技术：对评论做情感分析，结合用其他的负反馈信息（举报、不感兴趣、踩）等，来解决很多语义上的低质问题。
- 人工复审



