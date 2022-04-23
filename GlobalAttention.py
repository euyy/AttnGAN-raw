"""
Global attention takes a matrix and a query metrix.
Based on each query vector q, it computes a parameterized convex combination of the matrix
based.
H_1 H_2 H_3 ... H_n
  q   q   q       q
    |  |   |       |
      \ |   |      /
              .....
          \   |  /
                  a
Constructs a unit mapping.
$$(H_1 + H_n, q) => (a)$$
Where H is of `batch x n x dim` and q is of `batch x dim`.

References:
https://github.com/OpenNMT/OpenNMT-py/tree/fc23dfef1ba2f258858b2765d24565266526dc76/onmt/modules
http://www.aclweb.org/anthology/D15-1166
"""

import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes):
    '''
    name: 
    test: test font
    msg: 
    param {*}
    return {*}
    '''    
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


def func_attention(query, context, gamma1): 
    '''
    description: 单词向量在图像特征的引导下,生成新的单词特征向量
        14个样本,每个样本最多12个单词,每个单词是256维的向量
        图像特征通道数256,每个通道是17*17大小
        返回在图像特征的引导下生成的新的单词特征向量
        返回: context * contextT * query 
            (256,17*17) * (17*17,256) * (256,cap_len) => (256,cap_len)
    param {*} query(word): tensor(14,256,12)
    param {*} context(img_feat): tensor(14,256,17,17)
    param {*} gamma1 4.0,超参,用于放缩attn
    return {*} weightedContext: tensor(14, 256, 12),
    return {*} attn: tensor(14,12,17,17), 每个单词对每个像素点的attn值
    '''
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    '''
        14个样本,每个样本最多12个单词,每个单词是256维的向量
        图像特征通道数256,每个通道是17*17大小
        返回值加权的内容,就是单词在像素点的256维向量的引导下,生成新的单词的向量
        query(word): tensor(14,256,12)
        context(img_feat): tensor(14,256,17,17),单词和文本
    '''
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL) # ! tensor(14,256,17*17=289)
    contextT = torch.transpose(context, 1, 2).contiguous() # ! tensor(14,289,256)

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper # ! tensor(14,289,12)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL) # ! tensor(14*289,12)
    attn = nn.Softmax(dim=1)(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL) # ! tensor(14,289,12)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous() # ! tensor(14,12,289)
    attn = attn.view(batch_size*queryL, sourceL) # ! tensor(14*12,289)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax(dim=1)(attn)
    attn = attn.view(batch_size, queryL, sourceL) # ! tensor(14,12,289)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous() # ! tensor(14,289,12)

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT) # ! tensor(14,256,12)

    return weightedContext, attn.view(batch_size, -1, ih, iw) # ! tensor(14, 256,12), tensor(14,12,17,17)


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax(dim=1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn
