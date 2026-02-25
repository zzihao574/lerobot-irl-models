import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# RMSNorm -- Better, simpler alternative to LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


def _make_norm(dim: int, norm_type: str, eps: float = 1e-6) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


# SwishGLU -- GLU-style MLP block with SiLU gating.
class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


class Attention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int = 100,
        causal: bool = False,
        bias=False,
        qk_norm: bool = False,
        attention_impl: str = "auto",
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal
        if attention_impl not in {"auto", "manual"}:
            raise ValueError(f"Unsupported attention_impl: {attention_impl}")
        self.attention_impl = attention_impl

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash and causal:
            print(
                "WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
        self.block_size = block_size
        if self.causal:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
                persistent=False,
            )
        else:
            self.bias = None
        self.qk_norm = qk_norm
        # init qk norm if enabled
        if self.qk_norm:
            self.q_norm = RMSNorm(n_embd // self.n_head, eps=1e-6)
            self.k_norm = RMSNorm(n_embd // self.n_head, eps=1e-6)
        else:
            self.q_norm = self.k_norm = nn.Identity()
    def forward(self, x, custom_attn_mask=None):
        B, T, C = x.size()
        if self.causal and T > self.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.block_size} in causal attention"
            )

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        use_sdpa = self.attention_impl == "auto" and self.flash
        if use_sdpa:
            attn_mask = custom_attn_mask
            if attn_mask is None and self.causal:
                # Flash attention still uses internal causal path, but we validate length against block_size above.
                attn_mask = None
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0,
                is_causal=self.causal and custom_attn_mask is None,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if custom_attn_mask is not None:
                att = att.masked_fill(custom_attn_mask == 0, float("-inf"))
            elif self.causal:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(
        self,
        n_embd: int,
        bias: bool,
        mlp_type: str = "swishglu",
        dropout: float = 0,
    ):
        super().__init__()
        layers = []

        if mlp_type == "swishglu":
            layers.append(SwishGLU(n_embd, 4 * n_embd, bias=bias))
        elif mlp_type == "gelu":
            layers.append(nn.Linear(n_embd, 4 * n_embd, bias=bias))
            layers.append(nn.GELU())
        else:
            raise ValueError(f"Unsupported mlp_type: {mlp_type}")

        layers.append(nn.Linear(4 * n_embd, n_embd, bias=bias))
        layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        mlp_pdrop: float,
        block_size: int = 100,
        causal: bool = True,
        bias: bool = False,  # Attention output projection bias (kept for backward compatibility).
        qk_norm: bool = True,
        norm_type: str = "rmsnorm",
        mlp_type: str = "swishglu",
        mlp_bias: bool = False,
        attention_impl: str = "auto",
    ):
        super().__init__()
        self.ln_1 = _make_norm(n_embd, norm_type, eps=1e-6)
        self.attn = Attention(
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            block_size,
            causal,
            bias,
            qk_norm,
            attention_impl=attention_impl,
        )

        self.ln_2 = _make_norm(n_embd, norm_type, eps=1e-6)
        self.mlp = MLP(n_embd, bias=mlp_bias, mlp_type=mlp_type, dropout=mlp_pdrop)

    def forward(self, x, custom_attn_mask=None):
        x = x + self.attn(self.ln_1(x), custom_attn_mask=custom_attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Noise_Dec_only(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        goal_conditioned: bool,
        cond_mask_prob: float,
        embed_dim: int,
        embed_pdrob: float,
        goal_seq_len: int,
        window_size: int,
        linear_output: bool = True,
        use_pos_emb: bool = True,
        n_layers: int = 6,
        n_heads: int = 16,
        attn_pdrop: float = 0.3,
        resid_pdrop: float = 0.0,
        mlp_pdrop: float = 0.0,
        qk_norm: bool = True,
        norm_type: str = "rmsnorm",
        mlp_type: str = "swishglu",
        mlp_bias: bool = False,
        attention_impl: str = "auto",
        bias: bool = False,
    ):
        super().__init__()

        # Goal tokens are optional (disabled in the current no-goal setup).
        self.goal_conditioned = goal_conditioned
        self.cond_mask_prob = cond_mask_prob
        if not goal_conditioned:
            goal_seq_len = 0

        self.window_size = window_size

        # Source-style token counts:
        # block_size = [sigma] + [goal_seq_len] + [2 * window_size]
        self.block_size = goal_seq_len + 2 * self.window_size + 1

        # Position embeddings are defined per goal token and per timestep (state/action share timestep positions).
        self.seq_size = goal_seq_len + self.window_size

        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            n_layers=n_layers,
            block_size=self.block_size,
            causal=True,
            qk_norm=qk_norm,
            bias=bias,
            mlp_pdrop=mlp_pdrop,
            norm_type=norm_type,
            mlp_type=mlp_type,
            mlp_bias=mlp_bias,
            attention_impl=attention_impl,
        )

        # linear embedding for the state
        self.tok_emb = nn.Linear(state_dim, embed_dim)

        # linear embedding for the goal (only needed in goal-conditioned mode)
        if self.goal_conditioned:
            self.goal_emb = nn.Linear(goal_dim, embed_dim)
        else:
            self.goal_emb = None
        # linear embedding for the action

        self.action_emb = nn.Linear(action_dim, embed_dim)

        # Source DiffusionGPT-style sigma token embedding (input is c_noise = log(sigma) / 4).
        self.sigma_emb = nn.Linear(1, embed_dim)
        
        self.use_pos_emb = use_pos_emb

        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_size, embed_dim))

        self.drop = nn.Dropout(embed_pdrob)

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.goal_seq_len = goal_seq_len

        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100), nn.GELU(), nn.Linear(100, self.action_dim)
            )

        self.apply(self._init_weights)
        # logger.info(
        #     "number of parameters: %e", sum(p.numel() for p in self.parameters())
        # )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, Noise_Dec_only):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def mask_cond(self, goals, uncond: bool = False):
        if goals is None:
            return None
        
        if uncond:
            return torch.zeros_like(goals)
        
        if self.training and self.cond_mask_prob > 0:
            b, g ,d = goals.size()
            mask = torch.bernoulli(
                torch.full((b, g, d), self.cond_mask_prob, device=goals.device, dtype=goals.dtype)
            )
            return goals * (1.0 - mask)
        
        return goals

    def forward(self, states, actions, goals, sigma, uncond: bool = False):
        # Symbols:
        #   B=batch, T=sequence length (state/action tokens per modality), G=goal_seq_len, E=embed_dim, A=action_dim
        # Inputs:
        #   states:  [B, T, D_state_cond]
        #   actions: [B, T, A]
        #   goals:   [B, G, D_goal] or None
        #   sigma:   [B]
        b, t, dim = states.size()
        _, t_a, _ = actions.size()

        if t != t_a:
            raise ValueError(
                f"Beso expects action and state to have the same sequence length, "
                f"got states={t}, actions={t_a}"
            )

        c_noise = (sigma.log() / 4).view(-1, 1, 1)
        c_noise = c_noise.to(states.dtype)
        emb_t = self.sigma_emb(c_noise)  # [B, 1, E]
        state_embed = self.tok_emb(states)  # [B, T, E]
        action_embed = self.action_emb(actions)  # [B, T, E]
        
        if self.goal_conditioned:
            if goals is None:
                raise ValueError("goals must be provided when goal_conditioned=True")
            goals = self.mask_cond(goals)

            if uncond:
                goals = self.mask_cond(goals, uncond=True)
            goal_embed = self.goal_emb(goals)  # [B, G, E]

        # Add shared timestep position embeddings (and optional goal positions).
        if self.use_pos_emb:
            if self.goal_conditioned:
                position_embeddings = self.pos_emb[:, :(self.goal_seq_len + t), :]
                # position_embeddings: [1, G+T, E]
                goal_embed = goal_embed + position_embeddings[:, : self.goal_seq_len, :]
                step_pos = position_embeddings[
                    :, self.goal_seq_len : self.goal_seq_len + t, :
                ]  # [1, T, E]
            else:
                position_embeddings = self.pos_emb[:, :t, :]
                step_pos = position_embeddings  # [1, T, E]

            state_embed = state_embed + step_pos
            action_embed = action_embed + step_pos
        
        # Token dropout before transformer blocks.
        if self.goal_conditioned:
            goal_x = self.drop(goal_embed)
        state_x = self.drop(state_embed)
        action_x = self.drop(action_embed)

        # interleave [state_1, action_1, state_2, action_2, ...]
        sa_seq = (
            torch.stack([state_x, action_x], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(b, t * 2, self.embed_dim)
        )  # [B, 2*T, E]
            
        if self.goal_conditioned:
            encoder_input = torch.cat([emb_t, goal_x, sa_seq], dim=1)
            second_half_index = self.goal_seq_len + 1
            # encoder_input: [B, 1+G+2*T, E]
        else:
            encoder_input = torch.cat([emb_t, sa_seq], dim=1)
            second_half_index = 1
            # encoder_input: [B, 1+2*T, E]

        encoder_output = self.encoder(encoder_input)  # [B, 1(+G)+2*T, E]

        x = encoder_output[:, second_half_index :, :]
        x = x.reshape(b, t, 2, self.embed_dim).permute(0, 2, 1, 3)  # [B, 2, T, E]

        action_tokens = x[:, 1]  # [B, T, E]
        pred_actions = self.action_pred(action_tokens)  # [B, T, A]
        return pred_actions
        
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        n_layers: int,
        block_size: int = 100,
        causal: bool = True,
        qk_norm: bool = True,
        bias: bool = False,
        mlp_pdrop: float = 0,
        norm_type: str = "rmsnorm",
        mlp_type: str = "swishglu",
        mlp_bias: bool = False,
        attention_impl: str = "auto",
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    n_heads,
                    attn_pdrop,
                    resid_pdrop,
                    mlp_pdrop,
                    block_size,
                    causal=causal,
                    bias=bias,
                    qk_norm=qk_norm,
                    norm_type=norm_type,
                    mlp_type=mlp_type,
                    mlp_bias=mlp_bias,
                    attention_impl=attention_impl,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = _make_norm(embed_dim, norm_type, eps=1e-6)

    def forward(self, x, custom_attn_mask=None):
        # x: [B, L, E]
        for layer in self.blocks:
            x = layer(x, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        # x: [B, L, E]
        return x
