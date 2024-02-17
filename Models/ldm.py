# Python libraries
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics as tm
from torchmetrics import image as tmi
from einops import rearrange
from functools import partial

# Project imports
from Utilities.Distributions import DiagonalGaussianDistribution
from Utilities.Losses import LPIPSWithDiscriminator
from Models.ldm2d_encoder_decoder import Encoder, Decoder
from Models.ldm3d_encoder_decoder import Encoder3D, Decoder3D


class AutoencoderKL(pl.LightningModule):
    """
    Adopted from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py#L285
    """

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 learning_rate,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="fmri",
                 colorize_nlabels=None,
                 monitor=None,
                 version3d=False,
                 latent_scale=None,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.image_key = image_key

        self.learning_rate = learning_rate

        if not version3d:
            self.encoder = Encoder(**ddconfig)
            self.decoder = Decoder(**ddconfig)
        else:
            self.encoder = Encoder3D(**ddconfig)
            self.decoder = Decoder3D(**ddconfig)

        self.loss = LPIPSWithDiscriminator()
        assert ddconfig["double_z"]
        if not version3d:
            self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        else:
            self.quant_conv = torch.nn.Conv3d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.latent_scale = latent_scale

        self.automatic_optimization = False

        self.mse = tm.MeanSquaredError()
        self.ssim = tmi.StructuralSimilarityIndexMeasure()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        # if self.version4d:
        #     x = einops.rearrange(x, 'b t u h w c -> b c u t h w')
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def _rescale_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self.latent_scale is not None and self.latent_scale != 'none':
            if self.latent_scale == 'minmax_norm':
                return (z - z.min()) / (z.max() - z.min()) * 2 - 1
            elif self.latent_scale == 'std_norm':
                return (z - z.mean()) / z.std()
        else:
            return z

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        z = self._rescale_latent(z)

        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k=None):
        x = batch[k if k is not None else self.image_key]
        return x

    def training_step(self, batch, batch_idx):
        if self.loss.disc_factor > 0.0:
            opt_ae, opt_disc = self.optimizers()
        else:
            opt_ae = self.optimizers()

        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        if self.loss.disc_factor > 0.0:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            opt_disc.zero_grad()
            self.manual_backward(discloss)
            opt_disc.step()

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        # return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log_dict(log_dict_ae)

        if self.loss.disc_factor > 0.0:
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")
            self.log_dict(log_dict_disc)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        return self.log_dict

    def test_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log_dict(log_dict_ae)

        if self.loss.disc_factor > 0.0:
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")
            self.log_dict(log_dict_disc)

        ra = partial(rearrange, pattern='b c f h w -> (b f) c h w')
        self.log("mse", self.mse(ra(reconstructions), ra(inputs)), prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)
        self.log("ssim", self.ssim(ra(reconstructions), ra(inputs)), prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        if self.loss.disc_factor > 0.0:
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
            return opt_ae, opt_disc
        else:
            return opt_ae

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # TODO: POSSIBLY NOT NEEDED FOR US? ################################################################################
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x