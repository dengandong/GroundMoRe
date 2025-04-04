from typing import Optional, Tuple
from einops import rearrange

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, CLIPConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPVisionEmbeddings, CLIPEncoder, CLIPEncoderLayer, CLIPAttention, CLIPMLP


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, is_adapted=False):
        super().__init__()

        self.is_loaded = False
        self.is_adapted = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        
    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )

        if not self.is_adapted:
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.vision_tower_name, low_cpu_mem_usage=True
            )
            self.vision_tower.requires_grad_(False)
        else:
            self.vision_tower = AdaptedCLIPVisionModel.from_pretrained(
                self.vision_tower_name, low_cpu_mem_usage=True
            )
            # self.vision_tower.requires_grad_(False)   # TODO, only adapter is trainable

            # self.vision_tower.vision_model.embeddings.temporal_embedding.requires_grad = True
            # for name, param in self.vision_tower.vision_model.encoder.layers.named_parameters():
            #     if 'adapter' in name:
            #         param.requires_grad = True  # zero init
            #     else:
            #         param.requires_grad = False

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        torch.cuda.empty_cache()
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class AdaptedCLIPVisionModel(CLIPVisionModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)

        # add config?
        config.num_frames = 5
        self.vision_model = AdaptedCLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()


class AdaptedCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = AdaptedCLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = AdaptedCLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)


class AdaptedCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.num_frames = config.num_frames
        # self.temporal_embedding = nn.Parameter(torch.randn(1, self.num_frames, self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        # add temporal embedding

        return embeddings


class AdaptedCLIPEncoder(CLIPEncoder):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList([AdaptedCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        for layer in self.layers:
            layer.init_adapter()



class AdaptedCLIPEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.num_frames = config.num_frames
        self.mlp_adapter = Adapter(self.embed_dim, skip_connect=False)
        self.spatial_adapter = Adapter(self.embed_dim, skip_connect=True)
        self.temporal_adapter_1 = Adapter(self.embed_dim, skip_connect=True)
        self.temporal_adapter_2 = Adapter(self.embed_dim, skip_connect=False)

    
    def init_adapter(self):
        for name, module in self.named_modules():
            if "adapter" in name:
                for sub_name, sub_module in module.named_modules():
                    if "fc" in sub_name:
                        if isinstance(sub_module, nn.Linear):
                            nn.init.constant_(sub_module.weight, 0)
                            nn.init.constant_(sub_module.bias, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch * t, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        seq_len = hidden_states.size(1)

        # Adapter added code
        hidden_states = rearrange(hidden_states, "(b t) n d -> t (b n) d", t=self.num_frames)
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        # Adapter added code, temporal part
        hidden_states = self.temporal_adapter_1(hidden_states)
        # temporal attention
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.temporal_adapter_2(hidden_states)
        hidden_states = residual + hidden_states

        # Adapter added code, spatial part
        hidden_states = rearrange(hidden_states, "t (b n) d -> n (b t) d", n=seq_len)
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        # spatial attention
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.spatial_adapter(hidden_states)
        hidden_states = residual + hidden_states

        # Adapter added code, mlp part
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states + self.mlp_adapter(hidden_states)

        hidden_states = rearrange(hidden_states, "n (b t) d -> (b t) n d", t=self.num_frames)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x