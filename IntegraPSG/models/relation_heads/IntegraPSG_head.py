# EPOCH=12
# COUNT = 0
# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import ModuleList, force_fp32
from mmdet.core import build_assigner, build_sampler, multi_apply
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from IntegraPSG.models.frameworks.cnn_factory import creat_cnn
from IntegraPSG.utils import extract_depth_features
import pickle
import dbm
from IntegraPSG.models.commons.bert_wrapper import BertWrapper

def load_text_embed_db_to_memory(db_path):
    embed_dict = {}
    with dbm.open(db_path, 'r') as db:
        for k in db.keys():
            embed_dict[k.decode('utf-8')] = pickle.loads(db[k])
    return embed_dict

@HEADS.register_module()
class CrossHead2(AnchorFreeHead):
    def __init__(
        self,
        object_categories,
        relation_categories,
        num_classes,
        in_channels,
        num_relations,#56
        num_obj_query=100,
        num_rel_query=100,
        mapper: str = "conv_tiny",
        use_mask=True,
        pixel_decoder=None,
        transformer_decoder=None,
        feat_channels=256,
        out_channels=256,
        num_transformer_feat_level=3,
        embed_dims=256,

        output_feature_size=768,
        use_object_vision_only=False,
        use_triplet_vision_text_cross=True,
        pretrained_transformer=None,
        load_pretrained_weights=True,
        use_adapter=False,
        num_transformer_layer='all',
        text_info_db_dir='./pair_llm_db/qwen2.5_triplet_db1_merged',
        text_embed_db_dir='./pair_llm_db/qwen2.5_triplet_db1_test1_embed',
        text_embedding_size=768,
        use_cpu = None,
        prior_matrix_path='./psg_prior_matrix.npy',

        enforce_decoder_input_project=False,
        n_heads=8,
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
        rel_cls_loss=None,
        subobj_cls_loss=None,
        importance_match_loss=None,
        loss_cls=None,
        loss_mask=None,
        loss_dice=None,
        train_cfg=None,
        test_cfg=dict(max_per_img=100),
        init_cfg=None,
        **kwargs,
    ):
        super(AnchorFreeHead, self).__init__(init_cfg)
        # new config
        self.num_classes = num_classes
        self.num_rel_query = num_rel_query
        self.num_relations = num_relations
        self.use_mask = use_mask

        self.text_embedding_size = text_embedding_size
        if use_cpu:
            self.text_info_dict = load_text_embed_db_to_memory(os.path.join(text_info_db_dir, 'kv_triplet.db'))  # 预加载全部文本描述到内存
            pkl_path = os.path.join(text_embed_db_dir, 'kv_triplet.pkl')
            with open(pkl_path, 'rb') as f:
                self.text_embed_dict = pickle.load(f)
        else:
            self.text_info_db = dbm.open(os.path.join(text_info_db_dir, 'kv_triplet.db'), 'r')
            self.text_embed_db = dbm.open(os.path.join('./pair_llm_db/qwen2.5_triplet_db1_merged_embed', 'kv_triplet.db'), 'r')
        
        self.use_object_vision_only = use_object_vision_only
        self.use_triplet_vision_text_cross = use_triplet_vision_text_cross
        self.num_transformer_layer = num_transformer_layer
        if  self.use_triplet_vision_text_cross:
            self.sub_obj_pair_level_model = BertWrapper(pretrained_transformer, load_pretrained_weights,
                                                        num_transformer_layer, use_adapter, add_cross_attention=True)
            # 在 __init__ 里加
        self.object_categories = object_categories
        self.relation_categories = relation_categories

        self.prior_matrix = torch.from_numpy(np.load(prior_matrix_path)).float()
        self.depth_alpha = nn.Parameter(torch.tensor(0.9, dtype=torch.float32))
        self.depth_beta = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        self.update_importance = creat_cnn(mapper)

        # mask2former init
        self.n_heads = n_heads
        self.embed_dims = embed_dims

        assert "num_feats" in positional_encoding
        num_feats = positional_encoding["num_feats"]
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should"
            f" be exactly 2 times of num_feats. Found {self.embed_dims}"
            f" and {num_feats}."
        )

        self.num_classes = num_classes
        self.num_queries = num_obj_query
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        assert (
            pixel_decoder.encoder.transformerlayers.attn_cfgs.num_levels
            == num_transformer_feat_level
        )
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels,
        )
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (
                self.decoder_embed_dims != feat_channels
                or enforce_decoder_input_project
            ):
                self.decoder_input_projs.append(
                    Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1)
                )
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding
        )
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels),
        )
        if self.use_triplet_vision_text_cross:
            self.fc_triplet_vision_text_cross_input = nn.Sequential(
                nn.Linear(feat_channels * 2, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.fc_triplet_text_prompt_input = nn.Sequential(
                nn.Linear(text_embedding_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.fc_triplet_vision_text_cross_output = nn.Sequential(
                nn.Linear(output_feature_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            
            self.triplet_vision_text_cross_pred = nn.ModuleList(
                [nn.Linear(output_feature_size, 1) for _ in range(self.num_relations)])

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.mask_assigner = build_assigner(self.train_cfg.mask_assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get("num_points", 12544)
            self.oversample_ratio = self.train_cfg.get("oversample_ratio", 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                "importance_sample_ratio", 0.75
            )
            self.id_assigner = build_assigner(self.train_cfg.id_assigner)
        self.num_obj_query = num_obj_query
        self.use_mask = use_mask
        self.in_channels = in_channels

        self.class_weight = loss_cls.class_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

        self.rel_cls_loss = build_loss(rel_cls_loss)
        self.subobj_cls_loss = build_loss(subobj_cls_loss)
        self.importance_match_loss = build_loss(importance_match_loss)
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        self.sub_query_update = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
        )

        self.obj_query_update = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
        )

        self.rel_cls_embed = Linear(self.embed_dims, self.num_relations)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        """  for p in self.relation_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p) """

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """load checkpoints."""
        super(AnchorFreeHead, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    def get_topk_sub_obj_rel_text_embedding(
        self,
        object_categories,
        topk_sub_indices,
        topk_obj_indices,
        relation_categories,
        text_emb_size,
        feature,
        use_text_db=True
    ):
        """
        [1, K*R, dim]
        """
        sub_obj_rel_name_list = []
        sub_obj_rel_str_list = []
        sub_obj_rel_text_embedding_list = []
        for k in range(len(topk_sub_indices)):
            sub_idx = int(topk_sub_indices[k])
            obj_idx = int(topk_obj_indices[k])
            sub_name = object_categories[sub_idx]
            obj_name = object_categories[obj_idx]
            if sub_idx ==133 or obj_idx ==133:
                for rel_name in relation_categories:
                    sub_obj_rel_name = sub_name + '#' + obj_name + '#' + rel_name
                    sub_obj_rel_name_list.append(sub_obj_rel_name)
                    sub_obj_rel_str_list.append('')
                    sub_obj_rel_text_embedding_list.append(torch.zeros([text_emb_size]))
            else:
                for rel_name in relation_categories:
                    key = sub_name + '#' + obj_name + '#' + rel_name
                    sub_obj_rel_name = key
                    #if use_text_db and key in self.text_info_db:
                    #    sub_obj_rel_str = pickle.loads(self.text_info_db[key])
                    if use_text_db and key in self.text_info_dict:
                        sub_obj_rel_str = self.text_info_dict[key]  # 从预加载的字典中获取文本描述
                    else:
                        sub_obj_rel_str = ''

                    #if use_text_db and key in self.text_embed_db:
                    #    emb = torch.from_numpy(np.array(pickle.loads(self.text_embed_db[key])))#原始文本编码器的输出向量    期望[]
                    if use_text_db and key in self.text_embed_dict:
                        emb = torch.from_numpy(np.array(self.text_embed_dict[key]))
                    else:
                        emb = torch.zeros([text_emb_size])
                    sub_obj_rel_name_list.append(sub_obj_rel_name)
                    sub_obj_rel_str_list.append(sub_obj_rel_str)
                    sub_obj_rel_text_embedding_list.append(emb)

        emb = torch.stack(sub_obj_rel_text_embedding_list)
        emb = emb.unsqueeze(0)
        text_emb = emb.to(feature.device).to(feature.dtype)
        return text_emb,sub_obj_rel_name_list,sub_obj_rel_str_list

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred, attn_mask_target_size, mode="bilinear", align_corners=False
        )
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = (
            attn_mask.flatten(2)
            .unsqueeze(1)
            .repeat((1, self.n_heads, 1, 1))
            .flatten(0, 1)
        )
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def forward(self, feats, img_metas,depth=None):
        batch_size = len(img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        decoder_positional_encodings_rel = []

        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(2, 0, 1)
            decoder_positional_encoding_rel = self.decoder_positional_encoding(mask)
            decoder_positional_encoding_rel = decoder_positional_encoding_rel.flatten(
                2
            ).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
            decoder_positional_encodings_rel.append(decoder_positional_encoding_rel)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_feat_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        )
        for i, layer in enumerate(self.transformer_decoder.layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                    -2:
                ],
            )
            query_feat_list.append(query_feat)
        query_feats = torch.stack(query_feat_list)
        device = query_feat.device
        depth_feats_list = []
        for b in range(batch_size):
            depth_b = depth[b]
            if depth_b.dim() == 3 and depth_b.shape[0] == 1:
                depth_b = depth_b.squeeze(0)#[H, W]
            assert depth_b.dim() == 2, f"depth shape error: {depth_b.shape}"
            depth_b = depth_b.float()
            depth_b = torch.nan_to_num(depth_b, 0.0)
            # mask_pred: (batch_size, num_queries, h, w)
            mask_pred_b = mask_pred[b]
            if depth_b.shape != mask_pred_b.shape[-2:]:
                try:
                    depth_b = F.interpolate(
                        depth_b.unsqueeze(0).unsqueeze(0),
                        size=mask_pred_b.shape[-2:],
                        mode='area'
                    ).squeeze(0).squeeze(0)
                except RuntimeError as e:
                    print(f"Warning: Interpolation failed: {e}")
                    depth_b = torch.zeros(mask_pred_b.shape[-2:], device=device)
            mask_bin = (torch.sigmoid(mask_pred_b) > 0.5).float()
            try:
                depth_feats = extract_depth_features(depth_b, mask_bin)
                if torch.isnan(depth_feats).any() or torch.isinf(depth_feats).any():
                    depth_feats = torch.zeros(mask_bin.shape[0], 3, device=device)
            except Exception as e:
                print(f"Warning: Error in depth feature extraction: {e}")
                depth_feats = torch.zeros(mask_bin.shape[0], 3, device=device)
            depth_feats_list.append(depth_feats)
        depth_feats = torch.stack(depth_feats_list, dim=1)
        depth_feats = torch.nan_to_num(depth_feats, 0.0)

        depth_feats = depth_feats.permute(1, 0, 2)
        feat_mean = depth_feats.mean(dim=1, keepdim=True)
        feat_std = depth_feats.std(dim=1, keepdim=True) + 1e-6
        normalized_feats = (depth_feats - feat_mean) / feat_std
        B, N, _ = normalized_feats.shape
        mean_feat = normalized_feats[..., 0]
        std_feat = normalized_feats[..., 1]
        min_feat = normalized_feats[..., 2]
        mean_diff = mean_feat.unsqueeze(2) - mean_feat.unsqueeze(1)
        std_sim = -torch.abs(std_feat.unsqueeze(2) - std_feat.unsqueeze(1))
        min_mean_sim = -torch.abs(min_feat.unsqueeze(2) - mean_feat.unsqueeze(1))
        depth_score = (
            0.5 * torch.tanh(mean_diff)
            + 0.3 * torch.sigmoid(std_sim * 2)
            + 0.2 * torch.sigmoid(min_mean_sim * 2)
        )
        depth_importance = torch.tanh(depth_score * 2)
        depth_importance = (depth_importance - depth_importance.mean()) / (depth_importance.std() + 1e-6)
        depth_importance = torch.clamp(depth_importance, -1, 1)
        sub_prior_indices = cls_pred[..., :-1].argmax(dim=-1)
        obj_prior_indices = cls_pred[..., :-1].argmax(dim=-1)
        prior_matrix = self.prior_matrix.to(sub_prior_indices.device)
        prior_importance = prior_matrix[sub_prior_indices.unsqueeze(2), obj_prior_indices.unsqueeze(1)]

        #query_feats = torch.stack(query_feat_list)
        sub_embed = self.sub_query_update(query_feats)
        obj_embed = self.obj_query_update(query_feats)
        sub_embed = F.normalize(sub_embed[-1].transpose(0, 1), p=2, dim=-1, eps=1e-12)
        obj_embed = F.normalize(obj_embed[-1].transpose(0, 1), p=2, dim=-1, eps=1e-12)
        importance = torch.matmul(sub_embed, obj_embed.transpose(1, 2))
        alpha = torch.clamp(self.depth_alpha, 0.8, 1.2)
        beta = torch.clamp(self.depth_beta, 0.15, 0.3)
        importance = alpha * importance + beta * depth_importance
        impossible_mask = (prior_importance == -1)
        importance[impossible_mask] = -1
        importance = torch.clamp(importance, min=-1.0, max=1.0)
        importance = self.update_importance(importance)
        _, updated_importance_idx = torch.topk(
            importance.flatten(-2, -1), k=self.num_rel_query
        )
        sub_pos = torch.div(
            updated_importance_idx, self.num_obj_query, rounding_mode="trunc"
        )
        obj_pos = torch.remainder(updated_importance_idx, self.num_obj_query)
        obj_query_feat = torch.gather(
            query_feat,
            0,
            obj_pos.unsqueeze(-1).repeat(1, 1, self.embed_dims).transpose(0, 1),
        )
        sub_query_feat = torch.gather(
            query_feat,
            0,
            sub_pos.unsqueeze(-1).repeat(1, 1, self.embed_dims).transpose(0, 1),
        )
        sub_cls_preds = torch.gather(
            cls_pred.clone().detach(),
            1,
            sub_pos.unsqueeze(-1).expand(-1, -1, cls_pred.shape[-1]),
        )
        obj_cls_preds = torch.gather(
            cls_pred.clone().detach(),
            1,
            obj_pos.unsqueeze(-1).expand(-1, -1, cls_pred.shape[-1]),
        )

        if self.use_triplet_vision_text_cross:
            pair_feat = torch.cat([sub_query_feat, obj_query_feat], dim=-1)
            pair_feat = pair_feat.permute(1, 0, 2)
            batch_size, num_pairs, _ = pair_feat.shape
            sub_obj_pair_mask_attention = torch.ones([batch_size,num_pairs],device=pair_feat.device,dtype=torch.bool)

            sub_obj_rel_triplet_text_embedding_list = []
            for b in range(batch_size):
                sub_cat_indices = sub_cls_preds[b].argmax(dim=-1)
                obj_cat_indices = obj_cls_preds[b].argmax(dim=-1)
                sub_obj_rel_text_embedding,_,_ = self.get_topk_sub_obj_rel_text_embedding(
                    object_categories=self.object_categories,
                    topk_sub_indices = sub_cat_indices.tolist(),
                    topk_obj_indices = obj_cat_indices.tolist(),
                    relation_categories = self.relation_categories,
                    text_emb_size = self.text_embedding_size,
                    feature = pair_feat[b],
                    use_text_db=True
                )
                sub_obj_rel_triplet_text_embedding_list.append(sub_obj_rel_text_embedding)
            triplet_text_embedding = torch.cat(sub_obj_rel_triplet_text_embedding_list, dim=0)

            position_ids = torch.zeros(
                pair_feat.size()[:-1], dtype=torch.long, device=pair_feat.device)
            token_type_ids = torch.zeros(
                pair_feat.size()[:-1], dtype=torch.long, device=pair_feat.device)
            vision_emb = self.fc_triplet_vision_text_cross_input(pair_feat)

            text_emb = self.fc_triplet_text_prompt_input(triplet_text_embedding)
            batch_size, _, feature_size = text_emb.shape
            text_emb = text_emb.reshape(batch_size, -1, self.num_relations, feature_size)

            rel_preds_list = []
            for rel_idx in range(self.num_relations):
                vision_text_cross_output = self.sub_obj_pair_level_model(
                    inputs_embeds=vision_emb,
                    encoder_hidden_states=text_emb[:,:, rel_idx, :],
                    encoder_attention_mask=sub_obj_pair_mask_attention,
                    position_ids=position_ids,
                    token_type_ids=token_type_ids,
                    return_dict=True,)
                vision_text_cross_embedding = vision_text_cross_output.last_hidden_state

                vision_text_cross_final_embedding = self.fc_triplet_vision_text_cross_output(vision_text_cross_embedding)

                this_triplet_vision_text_cross_pred = self.triplet_vision_text_cross_pred[rel_idx](
                    vision_text_cross_final_embedding).reshape(
                    [-1, 1, self.num_rel_query])
                this_triplet_vision_text_cross_pred = this_triplet_vision_text_cross_pred.permute(0, 2, 1)
                rel_preds_list.append(this_triplet_vision_text_cross_pred)
            rel_preds = torch.cat(rel_preds_list, dim=2)
        else:
            rel_preds = None

        sub_seg = torch.gather(
            mask_pred.clone().detach(),
            1,
            sub_pos[..., None, None].expand(
                -1, -1, mask_pred.shape[-2], mask_pred.shape[-1]
            ),
        )
        obj_seg = torch.gather(
            mask_pred.clone().detach(),
            1,
            obj_pos[..., None, None].expand(
                -1, -1, mask_pred.shape[-2], mask_pred.shape[-1]
            ),
        )

        all_cls_scores = dict(
            sub=sub_cls_preds,  # (b, 100, 134)
            obj=obj_cls_preds,  # (b, 100, 134)
            cls=cls_pred,  # (b, 100, 134)
            rel=rel_preds,  # (b, 100, 56)
            importance=importance,  # (b, 100, 100)
        )
        all_mask_preds = dict(
            mask=mask_pred,  # (b,100,h,w)
            sub_seg=sub_seg,  # (b,100,h,w)
            obj_seg=obj_seg,  # same
        )
        return all_cls_scores, all_mask_preds

    @force_fp32(apply_to=("all_cls_scores", "all_mask_preds"))
    def loss(
        self,
        all_cls_scores,
        all_mask_preds,
        gt_rels_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        assert (
            gt_bboxes_ignore is None
        ), "Only supports for gt_bboxes_ignore setting to None."

        ### object detection and panoptic segmentation
        all_od_cls_scores = [all_cls_scores["cls"]]
        all_od_mask_preds = [all_mask_preds["mask"]]
        all_gt_rels_list = [gt_rels_list]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore]
        all_gt_masks_list = [gt_masks_list]
        img_metas_list = [img_metas]
        all_gt_labels_list = [gt_labels_list]
        ### scene graph generation
        importance = [all_cls_scores["importance"]]
        all_r_cls_scores = [all_cls_scores["rel"]]
        sub_cls_preds = [all_cls_scores["sub"]]
        obj_cls_preds = [all_cls_scores["obj"]]

        (
            r_losses_cls,
            sub_losses_cls,
            obj_losses_cls,
            loss_match,
        ) = multi_apply(
            self.loss_single,
            sub_cls_preds,
            obj_cls_preds,
            importance,
            all_od_cls_scores,
            all_od_mask_preds,
            all_r_cls_scores,
            all_gt_rels_list,
            all_gt_labels_list,
            all_gt_masks_list,
            img_metas_list,
            all_gt_bboxes_ignore_list,
        )

        loss_dict = dict()

        ### loss of scene graph
        loss_dict["loss_r_cls"] = r_losses_cls[-1]
        loss_dict["loss_sub_cls"] = sub_losses_cls[-1]
        loss_dict["loss_obj_cls"] = obj_losses_cls[-1]
        ### loss of importance score
        loss_dict["loss_match"] = loss_match[-1]

        return loss_dict

    def loss_single(
        self,
        sub_cls_preds,
        obj_cls_preds,
        importance,
        od_cls_scores,
        mask_preds,
        r_cls_scores,
        gt_rels_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        ## before get targets
        num_imgs = od_cls_scores.size(0)
        # obj det&seg
        cls_scores_list = [od_cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        # scene graph
        r_cls_scores_list = [r_cls_scores[i] for i in range(num_imgs)]
        subject_scores_list = [sub_cls_preds[i] for i in range(num_imgs)]
        object_scores_list = [obj_cls_preds[i] for i in range(num_imgs)]

        (
            r_labels_list,
            r_label_weights_list,
            gt_subject_id_list,
            gt_object_id_list,
            gt_importance_list,
        ) = self.get_targets(
            subject_scores_list,
            object_scores_list,
            cls_scores_list,
            mask_preds_list,
            r_cls_scores_list,
            gt_rels_list,
            gt_labels_list,
            gt_masks_list,
            img_metas,
            gt_bboxes_ignore_list,
        )

        r_label_weights = torch.cat(r_label_weights_list, 0)
        r_label_weights_mask = r_label_weights > 0

        gt_object_ids = torch.cat(gt_object_id_list, 0)
        loss_obj_cls = self.subobj_cls_loss(
            obj_cls_preds.flatten(0, 1)[r_label_weights_mask],
            gt_object_ids[r_label_weights_mask],
        )

        gt_subject_ids = torch.cat(gt_subject_id_list, 0)
        loss_sub_cls = self.subobj_cls_loss(
            sub_cls_preds.flatten(0, 1)[r_label_weights_mask],
            gt_subject_ids[r_label_weights_mask],
        )

        r_labels = torch.cat(r_labels_list, 0)
        r_cls_scores = r_cls_scores.reshape(-1, self.num_relations)
        # for seesaw loss
        dummy_objectness = torch.zeros((r_label_weights_mask.sum(), 2)).to(
            r_cls_scores.device
        )
        r_loss_cls = self.rel_cls_loss(
            torch.cat([r_cls_scores[r_label_weights_mask], dummy_objectness], dim=1),
            r_labels[r_label_weights_mask],
        )["loss_cls_classes"]
        # r_loss_cls = self.rel_cls_loss(
        #     r_cls_scores[r_label_weights_mask], r_labels[r_label_weights_mask]
        # )
        # importance matrix after update
        gt_importance = torch.stack(gt_importance_list, 0)
        pos_weight = torch.numel(gt_importance) / (gt_importance > 0).sum()

        loss_match = self.importance_match_loss(importance, gt_importance, pos_weight)

        return r_loss_cls, loss_sub_cls, loss_obj_cls, loss_match

    def get_targets(
        self,
        subject_scores_list,
        object_scores_list,
        cls_scores_list,
        mask_preds_list,
        r_cls_scores_list,
        gt_rels_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(r_cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            r_labels_list,
            r_label_weights_list,
            gt_subject_id_list,
            gt_object_id_list,
            gt_importance_list,
        ) = multi_apply(
            self._get_target_single,
            subject_scores_list,
            object_scores_list,
            cls_scores_list,
            mask_preds_list,
            r_cls_scores_list,
            gt_rels_list,
            gt_labels_list,
            gt_masks_list,
            img_metas,
            gt_bboxes_ignore_list,
        )

        return (
            r_labels_list,
            r_label_weights_list,
            gt_subject_id_list,
            gt_object_id_list,
            gt_importance_list,
        )

    def _get_target_single(
        self,
        subject_score,
        object_score,
        cls_score,
        mask_pred,
        r_cls_score,
        gt_rels,
        gt_labels,
        gt_masks,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        ############################### obj seg ####################################
        # sample points
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2), device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(self.num_queries, 1, 1)
        ).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1, 1)
        ).squeeze(1)

        # assign and sample
        assign_result = self.mask_assigner.assign(
            cls_score, mask_points_pred, gt_labels, gt_points_masks, img_metas
        )
        sampling_result = self.sampler.sample(assign_result, mask_pred, gt_masks)
        od_pos_inds = sampling_result.pos_inds
        # od_neg_inds = sampling_result.neg_inds

        ############################### scene graph #################################
        gt_label_assigned_query = torch.ones_like(gt_labels)
        # gt label pos: object query pos
        gt_label_assigned_query[sampling_result.pos_assigned_gt_inds] = od_pos_inds
        gt_rels = gt_rels.T.long()
        gt_rel_labels = gt_rels[2] - 1
        gt_sub_cls = gt_labels[gt_rels[0]]
        gt_obj_cls = gt_labels[gt_rels[1]]
        gt_sub_pos = gt_label_assigned_query[gt_rels[0]]
        gt_obj_pos = gt_label_assigned_query[gt_rels[1]]

        gt_importance = torch.zeros(
            (self.num_obj_query, self.num_obj_query), device=gt_labels.device
        )

        gt_importance[gt_sub_pos[:], gt_obj_pos[:]] += 1

        triplet_assign_result = self.id_assigner.assign(
            subject_score,
            object_score,
            r_cls_score,
            gt_sub_cls,
            gt_obj_cls,
            gt_rel_labels,
            img_metas,
            gt_bboxes_ignore,
        )

        triplet_sampling_result = self.sampler.sample(
            triplet_assign_result,
            torch.ones_like(subject_score),
            torch.ones_like(subject_score),
        )

        pos_inds = triplet_sampling_result.pos_inds
        # match id targets
        gt_subject_ids = torch.full(
            (self.num_rel_query,),
            -1,
            dtype=torch.long,
            device=gt_labels.device,
        )
        gt_subject_ids[pos_inds] = gt_sub_cls[
            triplet_sampling_result.pos_assigned_gt_inds
        ]

        gt_object_ids = torch.full(
            (self.num_rel_query,),
            -1,
            dtype=torch.long,
            device=gt_labels.device,
        )
        gt_object_ids[pos_inds] = gt_obj_cls[
            triplet_sampling_result.pos_assigned_gt_inds
        ]

        r_labels = torch.full(
            (self.num_rel_query,), -1, dtype=torch.long, device=gt_labels.device
        )
        r_labels[pos_inds] = gt_rel_labels[triplet_sampling_result.pos_assigned_gt_inds]
        r_label_weights = gt_labels.new_zeros(self.num_rel_query)
        r_label_weights[pos_inds] = 1.0

        result = (
            r_labels,
            r_label_weights,
            gt_subject_ids,
            gt_object_ids,
            gt_importance,
        )
        return result

    def forward_train(
        self,
        x,
        img_metas,
        gt_rels,
        gt_bboxes,
        gt_labels=None,
        gt_masks=None,
        gt_bboxes_ignore=None,
        proposal_cfg=None,
        depth=None,  #6月16号新增
        **kwargs,
    ):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas,depth=depth)
        if gt_labels is None:
            loss_inputs = outs + (gt_rels, gt_bboxes, gt_masks, img_metas)
        else:
            loss_inputs = outs + (gt_rels, gt_bboxes, gt_labels, gt_masks, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @force_fp32(apply_to=("all_cls_scores", "all_bbox_preds"))
    def get_bboxes(self, cls_scores, mask_preds, img_metas, rescale=False):
        result_list = []
        for img_id in range(len(img_metas)):
            all_cls_score = cls_scores["cls"][img_id, ...]
            all_masks = mask_preds["mask"][img_id, ...]
            s_cls_score = cls_scores["sub"][img_id, ...]
            o_cls_score = cls_scores["obj"][img_id, ...]
            r_cls_score = cls_scores["rel"][img_id, ...]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            s_mask_pred = mask_preds["sub_seg"][img_id, ...]
            o_mask_pred = mask_preds["obj_seg"][img_id, ...]
            triplets = self._get_bboxes_single(
                all_masks,
                all_cls_score,
                s_cls_score,
                o_cls_score,
                r_cls_score,
                s_mask_pred,
                o_mask_pred,
                img_shape,
                scale_factor,
                rescale,
            )
            result_list.append(triplets)

        return result_list

    def _get_bboxes_single(
        self,
        all_masks,
        all_cls_score,
        s_cls_score,
        o_cls_score,
        r_cls_score,
        s_mask_pred,
        o_mask_pred,
        img_shape,
        scale_factor,
        rescale=False,
    ):
        assert len(s_cls_score) == len(o_cls_score)

        mask_size = (
            round(img_shape[0] / scale_factor[1]),
            round(img_shape[1] / scale_factor[0]),
        )

        assert len(s_cls_score) == len(r_cls_score)

        # 0-based label input for objects and self.num_classes as default background cls
        s_logits = F.softmax(s_cls_score, dim=-1)[..., :-1]
        o_logits = F.softmax(o_cls_score, dim=-1)[..., :-1]

        s_labels = s_logits.argmax(-1) + 1
        o_labels = o_logits.argmax(-1) + 1

        r_dists = F.softmax(r_cls_score, dim=-1).reshape(-1, self.num_relations)
        r_dists = torch.cat(
            [torch.zeros(self.num_rel_query, 1, device=r_dists.device), r_dists], dim=-1
        )

        complete_labels = torch.cat((s_labels, o_labels), 0)
        all_logits = F.softmax(all_cls_score, dim=-1)[..., :-1]

        all_scores, all_labels = all_logits.max(-1)
        all_masks = F.interpolate(
            all_masks.unsqueeze(1), size=mask_size, mode="bilinear", align_corners=False
        ).squeeze(1)
        #### for panoptic postprocessing ####
        s_mask_pred = F.interpolate(
            s_mask_pred.unsqueeze(1),
            size=mask_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        s_mask_pred = torch.sigmoid(s_mask_pred) > 0.5
        o_mask_pred = F.interpolate(
            o_mask_pred.unsqueeze(1),
            size=mask_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        o_mask_pred = torch.sigmoid(o_mask_pred) > 0.5
        masks = torch.cat((s_mask_pred, o_mask_pred), 0)

        keep = (all_labels != s_logits.shape[-1] - 1) & (
            all_scores > 0.5
        )  ## the threshold is set to 0.5
        all_labels = all_labels[keep]
        all_masks = all_masks[keep]
        all_scores = all_scores[keep]
        h, w = all_masks.shape[-2:]

        if all_labels.numel() == 0:
            pan_img = torch.ones(mask_size).cpu().to(torch.long)
        else:
            all_masks = all_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(all_labels):
                if label.item() >= 80:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(all_masks, all_scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = all_masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                seg_img = m_id * INSTANCE_OFFSET + all_labels[m_id]
                seg_img = seg_img.view(h, w).cpu().to(torch.long)
                m_id = m_id.view(h, w).cpu()
                area = []
                for i in range(len(all_scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, pan_img = get_ids_area(all_masks, all_scores, dedup=True)
            if all_labels.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(all_labels)],
                        dtype=torch.bool,
                        device=keep.device,
                    )
                    if filtered_small.any().item():
                        all_scores = all_scores[~filtered_small]
                        all_labels = all_labels[~filtered_small]
                        all_masks = all_masks[~filtered_small]
                        area, pan_img = get_ids_area(all_masks, all_scores)
                    else:                           
                        break
        # dummy bboxes
        det_bboxes = torch.zeros(
            (self.num_rel_query * 2, 5), device=complete_labels.device
        )
        # dummy r_scores and r_labels
        r_scores = torch.zeros((self.num_rel_query), device=complete_labels.device)
        r_labels = torch.zeros((self.num_rel_query), device=complete_labels.device)
        rel_pairs = torch.arange(len(det_bboxes), dtype=torch.int).reshape(2, -1).T
        # (200, 5), (200), (100, 2), (200, h, w), (h, w), (100), (100), (100, 57)
        return (
            det_bboxes,
            complete_labels,
            rel_pairs,
            masks,
            pan_img,
            r_scores,
            r_labels,
            r_dists,
        )

    def simple_test_bboxes(self, feats, img_metas, rescale=False,depth=None):
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas,depth=depth)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list
