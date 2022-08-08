# bert-ndcg for LP(Link Prediction)，链接预测

## 概述
这里做个尝试，利用bert+ndcg去做链接预测的尝试。大体来説，就是将”头实体“、”关系“以及”尾实体“拼接在一起，它们的输出会通过排序学习中的ndcg进行优化。
这里利用umls的数据进行测试，umls数据已被处理成排序学习的数据格式，见data/umls/ptrain.csv;ptest.csv;pdev.csv。

处理思路大致是将关系当作query，将头实体、关系以及尾实体当作feature。正样本的label为1，反之label为0。

 ## 数据说明

数据处理见data/process.py

训练数据示例如下，其中各列为label、query、head、relation以及tail，格式与ltr的格式类似，不同是feature这块是以字符串形式给出。

```
1,1,acquired abnormality,location of,experimental model of disease
0,1,acquired abnormality,location of,fungus
1,2,anatomical abnormality,manifestation of,physiologic function
0,2,anatomical abnormality,manifestation of,gene or genome
1,3,alga,is a,entity
0,3,gene or genome,is a,entity
```

## 训练和预测见（src/test_krl.py）

## 项目结构
- data
    - umls
- examples
    - test_lp.py #训练及预测
- model
    - pretrained_model #存放预训练模型和相关配置文件
- src # 主代码


## 参考
- https://github.com/yao8839836/kg-bert