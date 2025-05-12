import torch
import torch.nn as nn
from .decomposition import Decomposition
from .branch import ResolutionBranch

class WPMixerCore(nn.Module):
    def __init__(self,
                 input_length=[],
                 pred_length=[],
                 wavelet_name=[],
                 level=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 tfactor=[],
                 dfactor=[],
                 device=[],
                 patch_len=[],
                 patch_stride=[],
                 no_decomposition=[]):
        super(WPMixerCore, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device
        self.no_decomposition = no_decomposition
        self.tfactor = tfactor
        self.dfactor = dfactor

        self.Decomposition_model = Decomposition(input_length=self.input_length,
                                                 pred_length=self.pred_length,
                                                 wavelet_name=self.wavelet_name,
                                                 level=self.level,
                                                 batch_size=self.batch_size,
                                                 channel=self.channel,
                                                 d_model=self.d_model,
                                                 tfactor=self.tfactor,
                                                 dfactor=self.dfactor,
                                                 device=self.device,
                                                 no_decomposition=self.no_decomposition)

        self.input_w_dim = self.Decomposition_model.input_w_dim  # list of the length of the input coefficient series
        self.pred_w_dim = self.Decomposition_model.pred_w_dim  # list of the length of the predicted coefficient series

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        # (m+1) number of resolutionBranch
        self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq=self.input_w_dim[i],
                                                                pred_seq=self.pred_w_dim[i],
                                                                batch_size=self.batch_size,
                                                                channel=self.channel,
                                                                d_model=self.d_model,
                                                                dropout=self.dropout,
                                                                embedding_dropout=self.embedding_dropout,
                                                                tfactor=self.tfactor,
                                                                dfactor=self.dfactor,
                                                                patch_len=self.patch_len,
                                                                patch_stride=self.patch_stride) for i in
                                               range(len(self.input_w_dim))])

    def forward(self, xL):
        '''
        Parameters
        ----------
        xL : Look back window: [Batch, look_back_length, channel]

        Returns
        -------
        xT : Prediction time series: [Batch, prediction_length, output_channel]
        '''
        x = xL.transpose(1, 2)  # [batch, channel, look_back_length]

        # xA: approximation coefficient series,
        # xD: detail coefficient series
        # yA: predicted approximation coefficient series
        # yD: predicted detail coefficient series

        xA, xD = self.Decomposition_model.transform(x)

        yA = self.resolutionBranch[0](xA)
        yD = []
        for i in range(len(xD)):
            yD_i = self.resolutionBranch[i + 1](xD[i])
            yD.append(yD_i)

        y = self.Decomposition_model.inv_transform(yA, yD)
        y = y.transpose(1, 2)
        xT = y[:, -self.pred_length:, :]  # decomposition output is always even, but pred length can be odd

        return xT


class WPMixer(nn.Module):
    """
        Paper: WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting
        Link: https://arxiv.org/abs/2412.17176
        Official Code: https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer
        Venue: AAAI 2025
        Task: Long-term Time Series Forecasting
    """
    def __init__(self, **model_args):
        super(WPMixer, self).__init__()
        self.wpmixerCore = WPMixerCore(input_length=model_args["seq_len"],
                                       pred_length=model_args["pred_len"],
                                       wavelet_name=model_args["wavelet"],
                                       level=model_args["level"],
                                       batch_size=model_args["batch_size"],
                                       channel=model_args["c_out"],
                                       d_model=model_args["d_model"],
                                       dropout=model_args["dropout"],
                                       embedding_dropout=model_args["dropout"],
                                       tfactor=model_args["tfactor"],
                                       dfactor=model_args["dfactor"],
                                       device=model_args["device"],
                                       patch_len=model_args["patch_len"],
                                       patch_stride=model_args["stride"],
                                       no_decomposition=model_args["no_decomposition"])
        
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """
            history_data [ B , L , N , C ] C=1
        """
        y = self.wpmixerCore(history_data.squeeze(-1))
        return y.unsqueeze(-1)