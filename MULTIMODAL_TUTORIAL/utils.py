from pathlib import Path

import torch
import wfdb

from meds_torch.utils.custom_multimodal_tokenization import MultimodalReader


class WaveformReader(MultimodalReader):
    def read_modality(self, relative_modality_fp: str) -> torch.Tensor:
        """
        Convert a WFDB (Waveform Database) record to a NumPy array.

        Parameters:
        ----------
        relative_modality_fp : str
            relative Path to the WFDB record file. For example:
            'files/p1000/p10000635/s45386375/45386375'

        Returns:
        -------
        numpy.ndarray
            A 2D NumPy array containing the signal data from the WFDB record,
            shape (T, 12). Each row corresponds to a sample point in time and
            each column corresponds to a different ECG lead.
        """
        data = wfdb.rdrecord(Path(self.base_path) / relative_modality_fp)
        return torch.tensor(data.p_signal.T, dtype=torch.float32).contiguous()
