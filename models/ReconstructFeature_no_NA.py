import time
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from natten import NeighborhoodAttention2D as NeighborhoodAttention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NSABlock(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicViTLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            NSABlock(dim=dim,
                     num_heads=num_heads, kernel_size=kernel_size,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ResViTBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size=7, mlp_ratio=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=nn.LayerNorm):
        super(ResViTBlock, self).__init__()
        self.dim = dim

        self.residual_group = BasicViTLayer(dim=dim, depth=depth, num_heads=num_heads, kernel_size=kernel_size,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                            attn_drop=attn_drop_rate,
                                            drop_path=drop_path_rate, norm_layer=norm_layer)

    def forward(self, x):
        return self.residual_group(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + x


def conv(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class EncoderDecoder_no_NA(nn.Module):

    def __init__(self, in_channels, N=128, M=320):
        super().__init__()

        depths = [2, 2, 6, 2, 2, 2]
        num_heads = [8, 12, 16, 20, 12, 12]
        kernel_size = 7
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm


        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.g_a0 = conv(in_channels, N, kernel_size=5, stride=2)
        self.g_a1 = ResViTBlock(dim=N,
                                depth=depths[0],
                                num_heads=num_heads[0],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                                norm_layer=norm_layer,
                                )
        self.g_a2 = conv(N, N * 3 // 2, kernel_size=3, stride=2)
        self.g_a3 = ResViTBlock(dim=N * 3 // 2,
                                depth=depths[1],
                                num_heads=num_heads[1],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:1]):sum(depths[:2])],
                                norm_layer=norm_layer,
                                )
        self.g_a4 = conv(N * 3 // 2, N * 2, kernel_size=3, stride=2)
        self.g_a5 = ResViTBlock(dim=N * 2,
                                depth=depths[2],
                                num_heads=num_heads[2],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:2]):sum(depths[:3])],
                                norm_layer=norm_layer,
                                )
        self.g_a6 = conv(N * 2, M, kernel_size=3, stride=2)
        self.g_a7 = ResViTBlock(dim=M,
                                depth=depths[3],
                                num_heads=num_heads[3],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:3]):sum(depths[:4])],
                                norm_layer=norm_layer,
                                )

        self.h_a0 = conv(M, N * 3 // 2, kernel_size=3, stride=2)
        self.h_a1 = ResViTBlock(dim=N * 3 // 2,
                                depth=depths[4],
                                num_heads=num_heads[4],
                                kernel_size=kernel_size // 2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:4]):sum(depths[:5])],
                                norm_layer=norm_layer,
                                )
        self.h_a2 = conv(N * 3 // 2, N * 3 // 2, kernel_size=3, stride=2)
        self.h_a3 = ResViTBlock(dim=N * 3 // 2,
                                depth=depths[5],
                                num_heads=num_heads[5],
                                kernel_size=kernel_size // 2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:5]):sum(depths[:6])],
                                norm_layer=norm_layer,
                                )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = ResViTBlock(dim=N * 3 // 2,
                                depth=depths[0],
                                num_heads=num_heads[0],
                                kernel_size=kernel_size // 2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                                norm_layer=norm_layer,
                                )
        self.h_s1 = deconv(N * 3 // 2, N * 3 // 2, kernel_size=3, stride=2)
        self.h_s2 = ResViTBlock(dim=N * 3 // 2,
                                depth=depths[1],
                                num_heads=num_heads[1],
                                kernel_size=kernel_size // 2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:1]):sum(depths[:2])],
                                norm_layer=norm_layer,
                                )
        self.h_s3 = deconv(N * 3 // 2, M, kernel_size=3, stride=2)

        self.g_s0 = ResViTBlock(dim=M,
                                depth=depths[2],
                                num_heads=num_heads[2],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:2]):sum(depths[:3])],
                                norm_layer=norm_layer,
                                )
        self.g_s1 = deconv(M, N * 2, kernel_size=3, stride=2)
        self.g_s2 = ResViTBlock(dim=N * 2,
                                depth=depths[3],
                                num_heads=num_heads[3],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:3]):sum(depths[:4])],
                                norm_layer=norm_layer,
                                )
        self.g_s3 = deconv(N * 2, N * 3 // 2, kernel_size=3, stride=2)
        self.g_s4 = ResViTBlock(dim=N * 3 // 2,
                                depth=depths[4],
                                num_heads=num_heads[4],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:4]):sum(depths[:5])],
                                norm_layer=norm_layer,
                                )
        self.g_s5 = deconv(N * 3 // 2, N, kernel_size=3, stride=2)
        self.g_s6 = ResViTBlock(dim=N,
                                depth=depths[5],
                                num_heads=num_heads[5],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:5]):sum(depths[:6])],
                                norm_layer=norm_layer,
                                )
        self.g_s7 = deconv(N, in_channels, kernel_size=5, stride=2)


    def g_a(self, x):
        x = self.g_a0(x)
        x = self.g_a2(x)
        x = self.g_a4(x)
        x = self.g_a6(x)
        return x

    def g_s(self, x):
        x = self.g_s1(x)
        x = self.g_s3(x)
        x = self.g_s5(x)
        x = self.g_s7(x)
        return x

    def h_a(self, x):
        x = self.h_a0(x)
        x = self.h_a2(x)
        return x

    def h_s(self, x):
        x = self.h_s1(x)
        x = self.h_s3(x)
        return x

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        y_hat = self.h_s(z)
        x_hat = self.g_s(y_hat)

        return x_hat


if __name__ == '__main__':
    data = torch.randn(8, 576, 128, 64).to('cuda:7')
    # fr = Feature_Reconstruct(3456, 1024).to('cuda:0')
    # fr = Feature_Reconstruct_Mine(in_channels=576, out_channels=576, base_width=256).to('cuda:0')
    fr = EncoderDecoder(576).to('cuda:7')
    # fr = NSABlock(576, kernel_size=5, num_heads=8).to('cuda:7')
    time_start = time.time()
    output = fr(data)

    r = torch.randn(8, 576, 128, 64).to('cuda:7')

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(fr.parameters(), lr=1e-5)
    loss = loss_fn(output, r)
    loss.backward()

    print(output.shape)
