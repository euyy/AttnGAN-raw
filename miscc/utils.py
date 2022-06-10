import os
import sys
import errno
import numpy as np
from torch.nn import init

import torch
import torch.nn as nn

from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import skimage.transform

from miscc.config import cfg


# For visualization ################################################
COLOR_DIC = {0:[128,64,128],  1:[244, 35,232],
             2:[70, 70, 70],  3:[102,102,156],
             4:[190,153,153], 5:[153,153,153],
             6:[250,170, 30], 7:[220, 220, 0],
             8:[107,142, 35], 9:[152,251,152],
             10:[70,130,180], 11:[220,20, 60],
             12:[255, 0, 0],  13:[0, 0, 142],
             14:[119,11, 32], 15:[0, 60,100],
             16:[0, 80, 100], 17:[0, 0, 230],
             18:[0,  0, 70],  19:[0, 0,  0]}
FONT_MAX = 50


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    # get a font
    # fnt = None  # ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list


def build_super_images(real_imgs, captions, ixtoword,
                       attn_maps, att_sze, lr_imgs=None,
                       batch_size=cfg.TRAIN.BATCH_SIZE,
                       max_word_num=cfg.TEXT.WORDS_NUM):
    '''
    description: 生成最多8个样本的效果图,如果有上一个分辨率的图像,会一起生成
        输入高分辨率图像和底分辨率图像,以及对应的文本,输出加有单词注意力的图像
        输出图像集合和对应的句子
        # ! att_maps的大小和lr_imgs的大小是一样的
        每行的第一个是生成的图像, 第二个是加了所有特征图最大值的特征图的图像, 第一行的是特征图, 第二行是加了特征图的生成图像
    param {*} real_imgs: tensor(bs,3,128/256,128/256)
    param {*} captions: tensor(bs,12)
    param {*} ixtoword: dict
    param {*} attn_maps: tensor(bs,12,64/128,64/128)
    param {*} att_sze: 64/128
    param {*} lr_imgs: tensor(bs,3,64/128,64/128),上一个分辨率的图像
    param {*} batch_size
    param {*} max_word_num: 12
    return {*} img_set: 图像集合
    return {*} sentences: 句子
    '''
    nvis = min(8,batch_size)
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]
    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2) # ! 128/256
    # ! 设置图像的大小,每行文本加图像为FONT_MAX,每行有最多max_word_num+2列,列宽为vis_size+2
    text_convas = \
        np.ones([batch_size * FONT_MAX,
                 (max_word_num + 2) * (vis_size + 2), 3],
                dtype=np.uint8) # ! array(bs*50,(12+2)*(128/256+2),3) 
    # ! 设置不同列的文本颜色
    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]


    real_imgs = \
        nn.functional.interpolate(real_imgs,size=(vis_size, vis_size),
                                  mode='bilinear', align_corners=False) # ! tensor(bs,3,128/256,128/256)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1)) # ! tensor(bs,128/256,128/256,3)
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
    if lr_imgs is not None:
        lr_imgs = \
            nn.functional.interpolate(lr_imgs,size=(vis_size, vis_size),
                                  mode='bilinear', align_corners=False) # ! tensor(bs,3,128/256,128/256)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1)) # ! tensor(bs,128/256,128/256,3)

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)
    '''
    # ! 把文本写到画布上
    text_convas: 画布
    captions: tensor(bs,12)
    ixtoword: dict
    vis_size: 128/256
    return:
        text_map: 写了文本的画布
        sentences: 写的文本内容
    '''
    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num): # ! 第i个样本
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze) # ! tensor(1,cap_len,64/126,64/128),第i个图像的注意力图
        # --> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True) # ! tensor(1,1,64/128,64/128), 所有词对图像注意力的最大值,等价于句子对于图像的注意力了
        attn = torch.cat([attn_max[0], attn], 1) # ! tensor(1,cap_len+1,64/128,64/128), 每个图像共cap_len+1个加注意力的图像，除了每个单词外的，第一个是全部单词的注意力最大值的注意力图像
        #
        attn = attn.view(-1, 1, att_sze, att_sze) # ! tensor(cap_len+1,1,64/128,64/128)
        attn = attn.repeat(1, 3, 1, 1).data.numpy() # ! tensor(cap_len+1,3,64/128,64/128), 给rgb通道都加相同的注意力值
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1)) # ! tensor(cap_len+1,64/128,64/128,3), 转成图像的矩阵类型，即通道数在最后
        num_attn = attn.shape[0] # ! cap_len+1, 每个图像有多少个加注意力的图
        #
        img = real_imgs[i] # ! 第i个图像（高分辨率）
        if lr_imgs is None:
            lrI = img
        else:
            lrI = lr_imgs[i] # ! 第i个图像（低分辨率）
        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]
        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            one_map = attn[j] # ! 第j个单词的注意力图
            if (vis_size // att_sze) > 1: # ! 把注意力图放大成高分辨率图像的大小
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze,
                                                     multichannel=True)
            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
        # ! 把注意力图放缩到[0,1]，再乘以255放缩到[0,255]
        # ! 然后和图像进行合并
        
        for j in range(seq_len + 1):
            if j < num_attn:
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal) 
                one_map *= 255
                #
                PIL_im = Image.fromarray(np.uint8(img))
                PIL_att = Image.fromarray(np.uint8(one_map))
                merged = \
                    Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new('L', (vis_size, vis_size), (210))
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)
                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad
            row.append(one_map)
            row.append(middle_pad)
            #
            row_merge.append(merged)
            row_merge.append(middle_pad)
        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
        # ! 第i个图像的注意力图生成完成

    # ! 将所有图像合成一个大的图像, 并返回
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


def build_super_images2(real_imgs, captions, cap_lens, ixtoword,
                        attn_maps, att_sze, vis_size=256, topK=5):
    batch_size = real_imgs.size(0)
    max_word_num = np.max(cap_lens)
    text_convas = np.ones([batch_size * FONT_MAX,
                           max_word_num * (vis_size + 2), 3],
                           dtype=np.uint8)

    real_imgs = \
        nn.functional.interpolate(real_imgs,size=(vis_size, vis_size),
                                    mode='bilinear', align_corners=False)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    img_set = []
    num = len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size, off1=0)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = cap_lens[i]
        thresh = 2./float(num_attn)
        #
        img = real_imgs[i]
        row = []
        row_merge = []
        row_txt = []
        row_beforeNorm = []
        conf_score = []
        for j in range(num_attn):
            one_map = attn[j]
            mask0 = one_map > (2. * thresh)
            conf_score.append(np.sum(one_map * mask0))
            mask = one_map > thresh
            one_map = one_map * mask
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze,
                                                     multichannel=True)
            minV = one_map.min()
            maxV = one_map.max()
            one_map = (one_map - minV) / (maxV - minV)
            row_beforeNorm.append(one_map)
        sorted_indices = np.argsort(conf_score)[::-1]

        for j in range(num_attn):
            one_map = row_beforeNorm[j]
            one_map *= 255
            #
            PIL_im = Image.fromarray(np.uint8(img))
            PIL_att = Image.fromarray(np.uint8(one_map))
            merged = \
                Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
            mask = Image.new('L', (vis_size, vis_size), (180))  # (210)
            merged.paste(PIL_im, (0, 0))
            merged.paste(PIL_att, (0, 0), mask)
            merged = np.array(merged)[:, :, :3]

            row.append(np.concatenate([one_map, middle_pad], 1))
            #
            row_merge.append(np.concatenate([merged, middle_pad], 1))
            #
            txt = text_map[i * FONT_MAX:(i + 1) * FONT_MAX,
                           j * (vis_size + 2):(j + 1) * (vis_size + 2), :]
            row_txt.append(txt)
        # reorder
        row_new = []
        row_merge_new = []
        txt_new = []
        for j in range(num_attn):
            idx = sorted_indices[j]
            row_new.append(row[idx])
            row_merge_new.append(row_merge[idx])
            txt_new.append(row_txt[idx])
        row = np.concatenate(row_new[:topK], 1)
        row_merge = np.concatenate(row_merge_new[:topK], 1)
        txt = np.concatenate(txt_new[:topK], 1)
        if txt.shape[1] != row.shape[1]:
            print('Warnings: txt', txt.shape, 'row', row.shape,
                  'row_merge_new', row_merge_new.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


####################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
