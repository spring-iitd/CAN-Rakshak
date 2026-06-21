from synthetic_data.base import DataAugmentor
from synthetic_data.augment import generate_spoof_dataset


class SpoofAugmentor(DataAugmentor):
    """
    Pipeline wrapper for spoof/impersonation dataset augmentation.

    Injects spoofed packets between consecutive occurrences of a target CAN ID
    in a normal (attack-free) trace, labelling injected packets as 'T'.
    """

    def augment(self, input_path, output_path):
        target_id      = self.cfg.get('target_id')
        payload_mode   = self.cfg.get('payload_mode', 'random')
        max_injections = self.cfg.get('max_injections', 10)
        predictor_cfg  = self.cfg.get('predictor', {})

        if not target_id:
            raise ValueError("data_augmentation.target_id must be set for spoof augmentation")

        print(f"  Attack type    : Spoof")
        print(f"  Target ID      : {target_id}")
        print(f"  Payload mode   : {payload_mode}")
        print(f"  Max injections : {max_injections}")
        generate_spoof_dataset(
            input_path, output_path,
            target_id=target_id,
            payload_mode=payload_mode,
            max_injections=max_injections,
            predictor_cfg=predictor_cfg,
        )
