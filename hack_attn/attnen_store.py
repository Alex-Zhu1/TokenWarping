import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from libs.rgb_flow import match_flow_warp_kv, rgb_flow_warp_kv, rgb_flow_warp_nose

def register_attention_control_v2(model, controller=None):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, flow_mask=None, warp=None):
            batch_size, sequence_length, _ = hidden_states.shape
            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)
            _, _, dim = query.shape
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            if warp:
                query = rearrange(query, "(b f) d c -> b f d c", f=video_length)
                if controller is not None:
                    query_previous = controller.store_query(query[:, -1:, ...])  # 保存warp前的query
                    controller.cur_index += 1   # 用于记录query当前的index
                    if query_previous is not None:
                        query_previous = query_previous.to(query.device)
                query = rgb_flow_warp_nose(query, flow=flow_mask, video_length=video_length, query_previous=query_previous)

                
                key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
                value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
                if controller is not None and controller.batch_id == 0:
                    # anchor token
                    ##################
                    controller.store_first_key(key[:, 0:1, ...] )      # 保存第一batch时候第一帧的key
                    controller.store_first_value(value[:, 0:1, ...])
                    # 两种key_0的方式
                    # key_0 = key[:, [0] * video_length]               
                    key_0 = match_flow_warp_kv(key, flow=flow_mask, video_length=video_length)   # 第二种，后续帧的occ + first frame
                    # value_0 = value[:, [0] * video_length]   
                    value_0 = match_flow_warp_kv(value, flow=flow_mask, video_length=video_length)

                    # warping token
                    ###################
                    key_warp = rgb_flow_warp_kv(key, flow=flow_mask, video_length=video_length)  
                    value_warp = rgb_flow_warp_kv(value, flow=flow_mask, video_length=video_length)
                    controller.store_key(key_warp[:, -1:, ...]) # 保存warp后的key
                    controller.store_value(value_warp[:, -1:, ...])

                elif controller is not None and controller.batch_id > 0:
                    # achor token
                    ###################
                    # key_0 = repeat(key_first, 'b 1 d c -> b f d c', f=video_length)
                    key_0 = match_flow_warp_kv(key, flow=flow_mask, video_length=video_length, 
                                               is_batch=True, first=controller.get_first_key().to(key.device))

                    # value_0 = repeat(value_first, 'b 1 d c -> b f d c', f=video_length)
                    value_0 = match_flow_warp_kv(value, flow=flow_mask, video_length=video_length, 
                                                 is_batch=True, first=controller.get_first_value().to(key.device))

                    # warping token
                    #####################
                    key_warp = rgb_flow_warp_kv(key, flow=flow_mask, video_length=video_length, 
                                                previous=controller.get_batch_pre_key().to(key.device))
                    value_warp = rgb_flow_warp_kv(value, flow=flow_mask, video_length=video_length, 
                                                  previous=controller.get_batch_pre_value().to(key.device))
                    controller.store_key(key_warp[:, -1:, ...])
                    controller.store_value(value_warp[:, -1:, ...])

                controller.cur_indexKey += 1   # 用于记录当前的index

                key = torch.cat([key_0, key_warp], dim=2)
                key = rearrange(key, "b f d c -> (b f) d c")

                value = torch.cat([value_0, value_warp], dim=2)
                value = rearrange(value, "b f d c -> (b f) d c")
            else:
                ValueError('error of flow warp operation in error module')

            query = self.reshape_heads_to_batch_dim(query)
            if self.added_kv_proj_dim is not None:
                raise NotImplementedError

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            # 变成fresco的head
            query = rearrange(query, "(b h) d c -> b d h c", b=batch_size, h=self.heads)
            key = rearrange(key, "(b h) d c -> b d h c", b=batch_size, h=self.heads)
            value = rearrange(value, "(b h) d c -> b d h c", b=batch_size, h=self.heads)
            # # #

            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                    attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

            if self._use_memory_efficient_attention_xformers:
                hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
                hidden_states = hidden_states.to(query.dtype)
            else:
                if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                    hidden_states = self._attention(query, key, value, attention_mask)
                else:
                    hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

            # fresco的multi-head
            hidden_states = rearrange(hidden_states, "b d h c -> (b h) d c", h=self.heads)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            # # #
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)

            return hidden_states

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.cur_step = 0
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'KeyFrameAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if 'down_blocks' in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif 'up_blocks' in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif 'mid_block' in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    print("flow warp counts: {}".format(cross_att_count))
    controller.num_att_layers = cross_att_count