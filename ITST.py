###############################################################################
#                                                                             #
# ITST                                                                        #
# D. Schneider                                                                #
#                                                                             #
###############################################################################

###############################################################################
#                                                                             #
# Imports                                                                     #
#                                                                             #
###############################################################################
import lightning.pytorch as pl
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from huggingface_hub import PyTorchModelHubMixin
from typing import Optional


###############################################################################
#                                                                             #
# Positional Encoding Submodule                                               #
#                                                                             #
###############################################################################
class PositionalEncoding(torch.nn.Module):
    def __init__(
        self,
        length: int,
        depth: int
    ) -> None:
        super().__init__()

        depth = depth/2

        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth

        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates

        pe = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)

        self.register_buffer('pe', torch.Tensor(pe))

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.pe
        return x


###############################################################################
#                                                                             #
# Parameter Extraction Submodule                                              #
#                                                                             #
###############################################################################
class ParamExtraction(torch.nn.Module):
    def __init__(
        self
    ) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        t_min = torch.unsqueeze(torch.min(x, dim=1).values, dim=1)
        t_max = torch.unsqueeze(torch.max(x, dim=1).values, dim=1)
        t_mean = torch.unsqueeze(torch.mean(x, dim=1), dim=1)
        t_std = torch.unsqueeze(torch.std(x, dim=1), dim=1)
        ret = torch.cat([x[:, -2:, :], t_min, t_max, t_mean, t_std], dim=1)
        return ret


###############################################################################
#                                                                             #
# Encoder Submodule                                                           #
#                                                                             #
###############################################################################
class Encoder(torch.nn.Module):
    def __init__(
        self,
        dimv: int,
        dimatt: int,
        n_heads: int,
        drop: float
    ) -> None:
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.attn = torch.nn.MultiheadAttention(
            dimatt,
            n_heads,
            drop,
            batch_first=True
        )
        self.ln2 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.ffn1 = torch.nn.Linear(dimv, dimv)
        self.ffn2 = torch.nn.Linear(dimv, dimv)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        a = self.ln1(x)
        a, _ = self.attn(a, a, a, attn_mask=mask)
        x = self.ln2(a + x)
        a = self.ffn2(torch.nn.ELU()(self.ffn1(x)))
        return x + a


###############################################################################
#                                                                             #
# Decoder Submodule                                                           #
#                                                                             #
###############################################################################
class Decoder(torch.nn.Module):
    def __init__(
        self,
        dimv: int,
        dimatt: int,
        n_heads: int,
        drop: float
    ) -> None:
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.attn1 = torch.nn.MultiheadAttention(
            dimatt,
            n_heads,
            drop,
            batch_first=True
        )
        self.ln2 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.attn2 = torch.nn.MultiheadAttention(
            dimatt,
            n_heads,
            drop,
            batch_first=True
        )
        self.ln3 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.ffn1 = torch.nn.Linear(dimv, dimv)
        self.ffn2 = torch.nn.Linear(dimv, dimv)

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor
    ) -> torch.Tensor:
        a = self.ln1(x)
        a, _ = self.attn1(a, a, a, key_padding_mask=None)
        x = self.ln2(a + x)
        a, _ = self.attn2(x, enc, enc, key_padding_mask=None)
        x = self.ln3(a + x)
        a = self.ffn2(torch.nn.ELU()(self.ffn1(x)))
        return x + a


###############################################################################
#                                                                             #
# ITST Module                                                                 #
#                                                                             #
###############################################################################
class ITST_LitModule(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.input_size = (40, 34)
        self.d_model = 64
        self.heads = 4
        self.nencoder = 4
        self.ndecoder = 2
        self.dim_val = self.d_model
        self.dec_l = 6
        self.output_size = 12

        self.time_encoder = torch.nn.ModuleList(
            [Encoder(self.dim_val, self.d_model, self.heads, 0)
             for _ in range(self.nencoder)])

        self.freq_encoder = torch.nn.ModuleList(
            [Encoder(self.dim_val, self.d_model, self.heads, 0)
             for _ in range(self.nencoder)])

        self.sens_encoder = torch.nn.ModuleList(
            [Encoder(self.dim_val, self.d_model, self.heads, 0)
             for _ in range(self.nencoder)])

        self.decoder = torch.nn.ModuleList(
            [Decoder(self.dim_val, self.d_model, self.heads, 0)
             for _ in range(self.ndecoder)])

        self.time_pos = torch.nn.ModuleList(
            [PositionalEncoding(self.input_size[0], self.dim_val)])

        self.freq_pos = torch.nn.ModuleList(
            [PositionalEncoding(self.input_size[0], self.dim_val)])

        self.sens_pos = torch.nn.ModuleList(
            [PositionalEncoding(self.input_size[1], self.dim_val)])

        self.decinp = torch.nn.ModuleList(
            [ParamExtraction()]
        )

        self.time_encemb = torch.nn.Linear(self.input_size[1], self.dim_val)
        self.freq_encemb = torch.nn.Linear(self.input_size[1], self.dim_val)
        self.sens_encemb = torch.nn.Linear(self.input_size[0], self.dim_val)
        self.decemb = torch.nn.Linear(self.input_size[1], self.dim_val)

        self.ln1 = torch.nn.LayerNorm(self.dim_val, eps=1e-5)
        self.out = torch.nn.Linear(self.dec_l*self.dim_val,
                                   self.output_size)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x_time = x
        x_freq = torch.fft.fft2(x).abs()
        x_sens = torch.transpose(x, dim0=1, dim1=2)

        e_time = self.time_encoder[0](self.time_pos[0](
            self.time_encemb(x_time)))
        for enc in self.time_encoder[1:]:
            e_time = enc(e_time)

        e_freq = self.freq_encoder[0](self.freq_pos[0](
            self.freq_encemb(x_freq)))
        for enc in self.freq_encoder[1:]:
            e_freq = enc(e_freq)

        e_sens = self.sens_encoder[0](self.sens_pos[0](
            self.sens_encemb(x_sens)))
        for enc in self.sens_encoder[1:]:
            e_sens = enc(e_sens)

        p = torch.cat((e_time, e_freq, e_sens), dim=1)
        p = self.ln1(p)

        d = self.decoder[0](self.decemb(self.decinp[0](x)), p)
        for dec in self.decoder[1:]:
            d = dec(d, p)

        x = self.out(torch.nn.ReLU()(d.flatten(start_dim=1)))

        return x

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int
    ) -> torch.Tensor:
        _, loss = self._get_classify_loss_accuracy(batch)

        # Log loss and metric
        self.log('train_CC', loss, sync_dist=True)

        return loss

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int
    ) -> torch.Tensor:
        classify, loss = self._get_classify_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_CC', loss, sync_dist=True)

        return classify

    def test_step(
        self,
        batch: torch.Tensor,
        batch_idx: int
    ) -> torch.Tensor:
        _, loss = self._get_classify_loss_accuracy(batch)

        # Log loss and metric
        self.log('test_CC', loss, sync_dist=True)

    def predict_step(
        self,
        batch: torch.Tensor,
        batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        return torch.argmax(self(x), dim=1)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5)
        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": 'val_CC'}

    def _get_classify_loss_accuracy(
        self,
        batch: torch.Tensor
    ) -> torch.Tensor:
        x, y = batch
        classify = self(x)
        loss = torch.nn.CrossEntropyLoss()(classify, y)
        return classify, loss
