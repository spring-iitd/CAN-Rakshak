from synthetic_data.base import DataAugmentor
from synthetic_data.augment import generate_dos_dataset


class DoSAugmentor(DataAugmentor):
    """
    Pipeline wrapper for DoS dataset augmentation.

    Replaces '0000' attack CAN IDs with valid IDs drawn from the dataset,
    preserving the timing structure of the original trace.
    """

    def augment(self, input_path, output_path):
        payload_mode  = self.cfg.get('payload_mode', 'random')
        predictor_cfg = self.cfg.get('predictor', {})
        print(f"  Attack type    : DoS")
        print(f"  Payload mode   : {payload_mode}")
        generate_dos_dataset(
            input_path, output_path,
            payload_mode=payload_mode,
            predictor_cfg=predictor_cfg,
        )
