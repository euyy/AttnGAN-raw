<!--
 * @Description: 
 * @version: 
 * @Author: Yue Yang
 * @Date: 2022-05-13 07:04:07
 * @LastEditors: Yue Yang
 * @LastEditTime: 2022-06-06 22:58:13
-->
# version 1.0

主要是把文本编码器换成了Transformer encoder。

数据集中取到的句子在前面增加了一个 `[CLS]` 标记，用来获取整个句子的信息。

text encoder的输出中，第一个单词位置对应的向量就是 *sentence embedding* ，剩下的是 *words embedding* 。

数据集的具体细节见下。

## 2022/05/09

首先，在数据集中字典的创建时，给每个句子的开头加了 `[cls]` 标记，这样在使用文本编码器的时候，就可以直接使用第一个维度的向量作为句子向量，而不需要使用 *max_pooling* 或者 *mean_pooling* 来获取句子向量。

因为Transformer Encoder在进行单词的 *embedding* 时，设置了 $pad\_idx = 0$ ，并且该值并不会求梯度，也就是说值为 $0$ 的单词，其嵌入向量就是零向量，且值不会发生改变，而其他非零向量的值是会发生改变的。

在每个 *batch* 的句子对齐时，会用 $0$ 作为 *pad* 填充，这就与 `[cls]` 标记重复了。

后面求mask的时候，是使用值是否是 $pad=0$ 来求的，这也会导致 `[cls]` 标记被 *mask* ，因此， *dataset* 中 *idx* 从 $1$ 开始，且 $1$ 为 `[cls]` 标记，建立字典时从 $idx=2$ 开始

## 2022/05/10

预训练text encoder和image encoder

```json
python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml -- gpu 0

{
    "name": "pretrain AttnGAN with birds",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/pretrain_DAMSM.py",
    "console": "integratedTerminal",
    "args": [
        "--cfg",
        "cfg/DAMSM/bird.yml",
        "--gpu",
        "0"
    ]
}
```

模型保存在 `../output/birds_DAMSM_2022_05_10_08_53_39` 中，

## 2022/05/11

训练生成器 G 和判别器 D 。选取预训练模型放在 `../DAMSMencoders/bird` 中，有文本编码器 text encoder 和图像编码器 image encoder ，选择的是 $epoch = 550$  的模型。

```json
python main.py --cfg cfg/bird_attn2.yml --gpu 0

{
    "name": "train AttnGAN with birds",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/main.py",
    "console": "integratedTerminal",
    "args": [
        "--cfg",
        "cfg/bird_attn2.yml",
        "--gpu",
        "0"
    ]
}
```

## 2022/06/06
把 image decoder 部分写完了，没有使用残差结构，等待调试看是否有其他问题