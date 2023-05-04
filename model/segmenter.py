import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from model.mae import mae_vit_base_patch16

from .layers import FPN, Projector, TransformerDecoder, MaskIoUProjector

#😉 okay. This is the CRIS class/ 
class CRIS(nn.Module): 
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
<<<<<<< HEAD
        try:
            clip_state_dict = torch.jit.load(
                cfg.clip_pretrain, map_location="cpu").eval().state_dict()
        except:
            clip_state_dict = torch.load(cfg.clip_pretrain, map_location="cpu")
        self.backbone = build_model(clip_state_dict, cfg.word_len).float()
=======
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval() #😉 it loads a CLIP model. 
        self.backbone = build_model(clip_model.state_dict(),
                                    cfg.word_len).float() #😉 builds the model and calls it backbone.
>>>>>>> cross valiation and some refactoring
        # Multi-Modal FPN
        self.neck_with_text_state = cfg.neck_with_text_state #😉 What is this? Apparently it is a multimodal FPN. The original paper also had a multimodl Neck.  The neck is used to combine the visual and text information. But it was not an FPN orignally.  
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out) #😉  The original paper uses something for fusing weights at different levels. It is similar to the FPN code And apparently it is also an FPN. I checked the code of the paper
        # Decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate) #😉 Decoder as put in the paper.
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3) #😉 projector

        # Mask IoU Projector
        self.pred_mask_iou = cfg.pred_mask_iou #😉 This is false.
        self.mask_iou_loss_type = cfg.mask_iou_loss_type
        self.mask_iou_loss_weight = cfg.mask_iou_loss_weight
        if self.pred_mask_iou:
            self.mask_iou_proj = MaskIoUProjector(cfg.word_dim, cfg.vis_dim,
                                                  cfg.vis_dim)
            if self.mask_iou_loss_type.lower() == 'mse':
                self.mask_iou_loss = nn.MSELoss()
            elif self.mask_iou_loss_type.lower() == 'bce':
                self.mask_iou_loss = nn.BCEWithLogitsLoss()
            else:
                assert False, 'Not support mask_iou_loss_type: {}'.format(
                    self.mask_iou_loss_type)

        # MoE
        self.use_moe_select_best_sent = cfg.use_moe_select_best_sent
        self.max_sent_num = cfg.max_sent_num
        self.use_moe_consistency_loss = cfg.use_moe_consistency_loss
        self.moe_consistency_loss_weight = cfg.moe_consistency_loss_weight
        if self.use_moe_select_best_sent:
            self.sent_selector = MaskIoUProjector(cfg.word_dim, cfg.vis_dim,
                                                  cfg.vis_dim)
            if self.use_moe_consistency_loss:
                self.moe_consistency_loss = nn.MSELoss()

        # MAE
        self.use_mae_gen_target_area = cfg.use_mae_gen_target_area #😉 mae generate target area. I guess this is the flag to activate MAE. Set to True.
        self.mae_pretrain = cfg.mae_pretrain #😉 mae pretrained, kept in the VIT place.  
        self.mae_input_shape = cfg.mae_input_shape #😉 214
        self.mae_mask_ratio = cfg.mae_mask_ratio #😉 0.75. I think this is 
        self.reconstruct_full_img = cfg.reconstruct_full_img #😉 This is set to false.
        self.mae_hard_example_mining_type = cfg.mae_hard_example_mining_type #😉 This is set to v1. It is supposed to ensure that we only choose points that we did not correctly predict to be used for prediction.
        self.mae_shared_encoder = cfg.mae_shared_encoder #whether to image encoders. 
        if self.mae_shared_encoder:
<<<<<<< HEAD
            assert "visual.proj" in clip_state_dict, 'CLIP model must use vit based visual encoder.'
        if self.use_mae_gen_target_area:
            assert self.mae_pretrain is not None
            assert self.mae_pretrain != ''
            self.mae = mae_vit_base_patch16(img_size=self.mae_input_shape[0])
            mae_state_dict = self.mae.state_dict()
            pre_state_dict = torch.load(self.mae_pretrain, map_location="cpu")
            # pos_embed is fixed sin-cos embedding and does not need to be loaded from the pre-trained model.
            pre_state_dict['model'].pop('pos_embed')
            mae_state_dict.update(pre_state_dict['model'])
            self.mae.load_state_dict(mae_state_dict)
            print('load MAE pretrain model from {} successfully.'.format(
                self.mae_pretrain))
=======
            assert "visual.proj" in clip_model.state_dict(
            ), 'CLIP model must use vit based visual encoder.'
        if self.use_mae_gen_target_area: #😉 If we are using masked autoencoers, 
            self.mae = MaskedAutoencoderViT(patch_size=16,
                                            embed_dim=768,
                                            depth=12,
                                            num_heads=12,
                                            decoder_embed_dim=512,
                                            decoder_depth=8,
                                            decoder_num_heads=16,
                                            mlp_ratio=4) #😉 Get the VIT MAE, 
            if self.mae_pretrain is not None and self.mae_pretrain != '': #😉 load MAE from pretrain.
                mae_state_dict = self.mae.state_dict()
                pre_state_dict = torch.load(self.mae_pretrain,
                                            map_location="cpu")
                mae_state_dict.update(pre_state_dict['model'])
                self.mae.load_state_dict(mae_state_dict)
                print('load MAE pretrain model from {} successfully.'.format(
                    self.mae_pretrain))
>>>>>>> cross valiation and some refactoring

    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool() #😉 Not sure what this does.

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        batch_size, _, img_h, img_w = img.shape #😉 dimensions.
        vis = self.backbone.encode_image(img) #😉 encode image with clip

        if self.use_moe_select_best_sent:
            # (b, sent_num, words) -> (b*sent_num, words)
            word = word.reshape((batch_size * self.max_sent_num, -1))
            pad_mask = pad_mask.reshape((batch_size * self.max_sent_num, -1))
            f_word, state = self.backbone.encode_text(word)
            if self.neck_with_text_state:
                fq_list = []
                for i_sent in range(self.max_sent_num):
                    this_fq = self.neck(
                        vis,
                        state[i_sent * batch_size:(i_sent + 1) * batch_size])
                    fq_list.append(this_fq)
                fq = torch.stack(fq_list, dim=0)
            else:
                fq = self.neck(vis)
                fq = fq.repeat((self.max_sent_num, 1, 1, 1))
                ori_shape = fq.shape
                fq = fq.reshape((
                    self.max_sent_num,
                    batch_size,
                ) + ori_shape[1:])
            fq = fq.transpose(0, 1)
            ori_shape = fq.shape
            fq = fq.reshape((ori_shape[0] * ori_shape[1], ) + ori_shape[2:])
            b, c, h, w = fq.size()
            fq = self.decoder(fq, f_word, pad_mask)
            fq = fq.reshape(b, c, h, w)
            pred = self.proj(fq, state)
            score = self.sent_selector(fq, state)
            pred_all = pred.reshape(
                (batch_size, self.max_sent_num, img_h // 4, img_w // 4))
            score_all = score.reshape((batch_size, self.max_sent_num))

            best_idx = torch.argmax(score_all, dim=1)  # b, 7
            best_idx_oh = F.one_hot(best_idx, num_classes=self.max_sent_num)
            pred_mask = torch.ones(
                (batch_size, self.max_sent_num, img_h // 4, img_w // 4),
                device=best_idx.device) * best_idx_oh[:, :, None, None]
            pred = torch.masked_select(pred_all, pred_mask.bool()).reshape(
                (batch_size, 1, img_h // 4, img_w // 4))
            if self.training and self.use_moe_consistency_loss:
                pred_all = pred_all.sigmoid()
                moe_consistency_loss = img.new_zeros(
                    (self.max_sent_num, self.max_sent_num))
                for c_i in range(self.max_sent_num):
                    for c_j in range(self.max_sent_num):
                        if c_i == c_j:
                            continue
                        moe_consistency_loss[c_i,
                                             c_j] = self.moe_consistency_loss(
                                                 pred_all[:, c_i],
                                                 pred_all[:, c_j])
        else:
            word, state = self.backbone.encode_text(word) #😉 Get the text encodings with clip.
            # b, 512, 26, 26 (C4)
            if self.neck_with_text_state: #😉 if we are adding text to the neck, which the original paper did, then we pass network through FPN like neck network.
                fq = self.neck(vis, state)
            else:
                fq = self.neck(vis)
            b, c, h, w = fq.size()
            fq = self.decoder(fq, word, pad_mask) #😉 Decoder
            fq = fq.reshape(b, c, h, w) #😉 Reshape.
            # b, 1, 104, 104
            pred = self.proj(fq, state) #😉 Project to bigger, and that is the prediction.

        results = dict()
        results['pred'] = pred.detach() #😉 detach prediction from graph.
        if self.pred_mask_iou:
            # b,
            mask_iou_pred = self.mask_iou_proj(fq, state) #😉 I am actually not sure what the mask iou predction refers to.
            results['mask_iou_pred'] = mask_iou_pred

        if self.training: #😉 If we are training,
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach() #😉 if mask shape not the same as image shape, interpolate to make mask same shapre.
            results['target'] = mask #😉 mask is then added as results_target. I amguessing maybe this s the final result, not sure.  
            loss = F.binary_cross_entropy_with_logits(pred, mask)

            if self.pred_mask_iou:
                # threshold = 0.35, same as test
                pred_t = (pred.detach().reshape(
                    (b, -1)).sigmoid() > 0.35).to(torch.float)
                mask_t = (mask.detach().reshape((b, -1))).to(torch.float)
                mask_iou_label = (pred_t * mask_t).sum(-1) / (
                    (pred_t + mask_t) > 0).sum(-1)
                mask_iou_loss = self.mask_iou_loss(mask_iou_pred,
                                                   mask_iou_label)
                loss = loss + mask_iou_loss * self.mask_iou_loss_weight

            if self.use_moe_select_best_sent and self.use_moe_consistency_loss:
                loss = loss + moe_consistency_loss.mean(
                ) * self.moe_consistency_loss_weight

<<<<<<< HEAD
            if self.use_mae_gen_target_area:
                mae_img = F.interpolate(img,
                                        self.mae_input_shape,
                                        mode='bilinear').detach()
                if not self.reconstruct_full_img:
                    mae_mask = F.interpolate(mask,
                                             self.mae_input_shape,
                                             mode='nearest').detach()
                    mae_img = mae_img * mae_mask
                if self.mae_hard_example_mining_type is not None:
                    pred_t = (pred.detach().sigmoid() > 0.35).to(torch.int32)
                    mask_t = mask.detach().to(torch.int32)
=======
            if self.use_mae_gen_target_area: #😉 okay, for MAE stuff, 
                mae_img = F.interpolate(img, self.mae_input_shape,
                                        mode='bilinear').detach()  #😉 Reduce image to Me size wihic is 214
                if not self.reconstruct_full_img:
                    mae_mask = F.interpolate(mask, self.mae_input_shape,
                                             mode='nearest').detach() #😉 if we arent reconstructing the whole image, then we also reduce the size of the mask.
                    mae_img = mae_img * mae_mask #😉 The image that is passed to the mae is the mask * the original image reduced in size.
                if self.mae_hard_example_mining_type is not None: #😉 For hard example mining, 
                    pred_t = (pred.detach().sigmoid() > 0.35).to(torch.int32) #😉 take the predictions that are confident, 
                    mask_t = mask.detach().to(torch.int32) #😉 Take teh mask, 
>>>>>>> cross valiation and some refactoring
                    if self.mae_hard_example_mining_type == 'v0':
                        mae_hard_example = torch.logical_xor(pred_t, mask_t) #😉 Things that are not confident, and not part of the mask are hard examples. that is v0
                    elif self.mae_hard_example_mining_type == 'v1':
                        pred_t = torch.where(mask < 0.5,
                                             torch.zeros_like(pred_t), pred_t) #😉 things that are not vert condident, and also not part of the mask are hard examples.
                        mae_hard_example = torch.logical_xor(pred_t, mask_t)
                    mae_hard_example = mae_hard_example.to(torch.float32)
                    mae_hard_example = F.interpolate(mae_hard_example,
                                                     self.mae_input_shape,
                                                     mode='nearest').detach()
                else:
                    mae_hard_example = None
                mae_encoder = self.backbone.visual.transformer if self.mae_shared_encoder else None
                mae_loss, mae_pred, mae_mask = self.mae(
                    mae_img,
                    hard_example=mae_hard_example,
                    encoder=mae_encoder) #😉 Mae image passes. He detaches. 
                loss = loss + mae_loss

            results['loss'] = loss
        else:
            if self.use_mae_gen_target_area:
                mae_img = F.interpolate(img,
                                        self.mae_input_shape,
                                        mode='bilinear').detach()
                # if not self.reconstruct_full_img:
                #     pred_mask_t = (pred.detach().sigmoid() > 0.35).to(
                #         torch.float32)
                #     pred_mask_t = F.interpolate(pred_mask_t, self.mae_input_shape,
                #                                 mode='nearest').detach()
                #     mae_img = mae_img * pred_mask_t
                mae_encoder = self.backbone.visual.transformer if self.mae_shared_encoder else None
                mae_loss, mae_pred, mae_mask = self.mae(
                    mae_img,
                    mask_ratio=self.mae_mask_ratio,
                    encoder=mae_encoder)

                # visualize MAE
                # 1. visualize the pred img
                mae_pred = self.mae.unpatchify(mae_pred)
                mae_pred = torch.einsum('nchw->nhwc', mae_pred).detach()
                # 2. visualize the mask
                mae_mask = mae_mask.detach()
                mae_mask = mae_mask.unsqueeze(-1).repeat(
                    1, 1, self.mae.patch_embed.patch_size[0]**2 *
                    3)  # (N, H*W, p*p*3)
                mae_mask = self.mae.unpatchify(
                    mae_mask)  # 1 is removing, 0 is keeping
                mae_mask = torch.einsum('nchw->nhwc', mae_mask).detach()
                # 3. original img
                mae_img = torch.einsum('nchw->nhwc', mae_img)
                # 4. masked img
                masked_mae_img = mae_img * (1 - mae_mask)
                # mae reconstruction pasted with visible patches
                mae_img_paste = mae_img * (1 - mae_mask) + mae_pred * mae_mask

                results['mae_img'] = mae_img
                results['mased_mae_img'] = masked_mae_img
                results['mae_pred'] = mae_pred
                results['mae_img_paste'] = mae_img_paste

        return results
