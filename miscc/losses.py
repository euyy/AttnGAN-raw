import torch
import torch.nn as nn

import numpy as np
from miscc.config import cfg

from GlobalAttention import func_attention


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """
    Returns cosine similarity between x1 and x2, computed along dim.
    return: x1*x2 / ( |x1| * |x2| )
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids,
              batch_size, eps=1e-8):
    '''
    description: 
    param {*} cnn_code(sent_emb): tensor(14,256)
    param {*} rnn_code(global_img_emb): tensor(14,256)
    param {*} labels: tensor(14), (1,2,...,bs-1)
    param {*} class_ids: array(14)
    param {*} batch_size: 14
    param {*} eps
    return {*}
    '''
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda() # ! tensor(14,14)

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0) # ! tensor(1,14,256)
        rnn_code = rnn_code.unsqueeze(0) # ! tensor(1,14,256)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True) # ! tensor(1,14,1)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True) # ! tensor(1,14,1)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2)) # ! tensor(1,14,14)
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2)) # ! tensor(1,14,14)
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3 # ! tensor(1,14,14), cnn * rnn / (|cnn| * |rnn|)

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()  # ! tensor(14,14)
    if class_ids is not None:
        scores0.data.masked_fill_(masks == 1, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels) # ! tensor(1)
        loss1 = nn.CrossEntropyLoss()(scores1, labels) # ! tensor(1)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size):
    '''
    description: 
    param {*} img_features: tensor(14,256,17,17)
    param {*} words_emb: tensor(14,256,12)
    param {*} labels: tensor(14)
    param {*} cap_lens: tensor(14)
    param {*} class_ids: array(14)
    param {*} batch_size: 14
    return {*} loss0: tensor(1)
    return {*} loss1: tensor(1)
    return {*} att_maps: tensor(bs, sent_len, 17, 17)
    '''
    
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """

    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist() # ! list[14]
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8) # ! array(14)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous() # ! tensor(1,256,cap_len)
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1) # ! tensor(14,256,cap_len)
        # batch x nef x 17*17
        context = img_features  # ! tensor(14,256,17,17)
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        '''
        weiContext: tensor(14,256,cap_len), 图像引导后的文本特征
        attn: tensor(14,cap_len,17,17)
        第i个样本的单词特征向量和所有样本的图像特征进行注意力操作
        不同的样本句子长度cap_len可能不一样
        '''
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous() # ! tensor(14,cap_len,256),第i个样本单词特征复制成bs个样本
        weiContext = weiContext.transpose(1, 2).contiguous() # ! tensor(14,cap_len,256),第i个样本单词在所有样本图像特征的注意力引导下,生成的新的单词特征
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1) # ! tensor(14*cap_len,256)
        weiContext = weiContext.view(batch_size * words_num, -1) # ! tensor(14*cap_len,256)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext) # ! tensor(14*cap_len) ,计算了新生成的单词特征与原有的单词特征之间的相似性
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num) # ! tensor(14, cap_len) 

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True) # ! tensor(14，1) 
        row_sim = torch.log(row_sim) # ! tensor(14，1) 

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1) # ! tensor(14,14)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda() # ! tensor(14,14)

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks == 1, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels) # ! tensor(1),logSoftmax + nllloss, nllloss是根据labels选择每个样本的值,取负值,求和,再求均值
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


# ##################Loss for G and Ds##############################
def discriminator_loss(netD, real_imgs, fake_imgs, conditions,
                       real_labels, fake_labels):
    '''
    imgs[i]: tensor(14,3,img_sz, img_sz)
    fake_imgs[i]: tensor(14,3,img_sz, img_sz)
    conditions: sent_emb, tensor(14,256)
    real_labels: tensor(14),全1
    fake_labels: tensor(14),全0
    '''
    # Forward
    real_features = netD(real_imgs) # ! tensor(14,768,4,4)
    fake_features = netD(fake_imgs.detach()) # ! tensor(14,768,4,4)
    # loss
    #
    cond_real_logits = netD.COND_DNET(real_features, conditions) # ! tensor(14)
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.COND_DNET(fake_features, conditions) # ! tensor(14)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size]) # ! tensor(13)
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features) # ! tensor(14)
        fake_logits = netD.UNCOND_DNET(fake_features) # ! tensor(14)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.
    return errD


def generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                   words_embs, sent_emb, match_labels,
                   cap_lens, class_ids):
    '''
    fake_imgs: list[tensor(14,3,64,64),tensor(14,3,128,128),tensor(14,3,256,256)]
    real_labels: tensor(14), 全1
    words_embs: tensor(14,256,12)
    sent_emb: tensor(14,256)
    match_labels: tensor(14), 0到bs-1,(0,1,...,bs-1) 
    cap_lens: tensor(14)
    class_ids: array(14)
    '''
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](fake_imgs[i]) # ! tensor(14,768,4,4)
        cond_logits = netsD[i].COND_DNET(features, sent_emb) # ! tensor(14)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is  not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss
        # err_img = errG_total.data[0]
        logs += 'g_loss%d: %.2f ' % (i, g_loss.item())

        # Ranking loss
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            # ! region_features: tensor(14,256,17,17)
            # ! cnn_code: tensor(14,256)
            # ! _: list[14*tensor(1,cap_len,17,17)]
            region_features, cnn_code = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                                             match_labels, cap_lens,
                                             class_ids, batch_size)
            w_loss = (w_loss0 + w_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA
            # err_words = err_words + w_loss.data[0]

            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                                         match_labels, class_ids, batch_size)
            s_loss = (s_loss0 + s_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA
            # err_sent = err_sent + s_loss.data[0]

            errG_total += w_loss + s_loss
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss.item(), s_loss.item())
    return errG_total, logs


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # * 0.5 * sum(exp(sigma^2) - ( 1 + sigma^2 ) + mu^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
