from meds_torch.models.components.lstm import LstmModel
from meds_torch.models.components.mamba import MambaModel
from meds_torch.models.components.transformer_decoder import TransformerDecoderModel

AUTOREGRESSIVE_MODELS = (LstmModel, MambaModel, TransformerDecoderModel)
