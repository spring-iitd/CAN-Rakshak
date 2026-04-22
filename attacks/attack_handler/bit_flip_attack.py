import os
from datetime import datetime

from .base import Attack
from ..Bit_Flip_attack.attack import run as run_attack
from ..Bit_Flip_attack.traffic_decoder import run as run_decode
from ..Bit_Flip_attack.evaluate_attack import run as run_evaluate
from ..Bit_Flip_attack.update_labels import run as run_update


class BitFlipAttack(Attack):
    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, **kwargs):
        cfg    = self.cfg
        rounds = cfg.get('rounds', 1)

        attack_cfg   = cfg.get('attack', {})
        decode_cfg   = cfg.get('decode', {})
        evaluate_cfg = cfg.get('evaluate', {})
        update_cfg   = cfg.get('update', {})

        tracksheet = attack_cfg['original_tracksheet']

        # Derive the dataset directory from the tracksheet path.
        # Tracksheet lives at <dataset_dir>/csv_files/<name>.csv,
        # so two dirname() calls reach <dataset_dir>.
        dataset_dir = os.path.dirname(os.path.dirname(tracksheet))
        test_dataset_dir = cfg.get('test_dataset_dir', '')
        original_test_dir = os.path.join(dataset_dir, "test", test_dataset_dir)

        label_file         = os.path.join(original_test_dir, "labels.txt")
        timestamp          = datetime.now().strftime("%Y%m%d_%H%M%S")
        attack_result_path = os.path.join(dataset_dir, "Results", "BitFlipAttack", timestamp)
        os.makedirs(attack_result_path, exist_ok=True)

        output_dir            = os.path.join(attack_result_path, attack_cfg['output_dir'])
        decoded_output_dir    = os.path.join(attack_result_path, decode_cfg['decoded_output_dir'])
        prediction_output_dir = os.path.join(attack_result_path, evaluate_cfg['prediction_output_dir'])
        tracksheet_dir        = os.path.join(attack_result_path, update_cfg['tracksheet_dir'])

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(decoded_output_dir, exist_ok=True)
        os.makedirs(prediction_output_dir, exist_ok=True)
        os.makedirs(tracksheet_dir, exist_ok=True)
        print("orginal dataset dir: ", original_test_dir)
        print("Label file : ", label_file)
        # Round 0 reads the original images; subsequent rounds read the
        # perturbed images that were written to output_dir by the previous round.
        current_test_dir = original_test_dir
        for round_num in range(rounds):
            print(f"\n{'='*60}")
            print(f"  BitFlipAttack — Round {round_num}")
            print(f"{'='*60}")

            # Generates perturbed images and packet_level_data_{round}.csv
            print(f"\n[Stage 1] Running attack (round={round_num}) ...")
            run_attack({
                "test_data_dir":     current_test_dir,
                "test_label_file":   label_file,
                "packet_level_data": tracksheet,
                "model_path":        cfg['surrogate_model'],
                "output_path":       output_dir,
                "rounds":            round_num,
            })

            # packet_level_data_{round_num}.csv is written by run_attack into output_dir;
            # it contains the columns traffic_decoder requires: timestamp, can_id,
            # original_label, operation_label (and pred_label for rounds > 0).
            packet_level_csv = os.path.join(output_dir, f"packet_level_data_{round_num}.csv")
            traffic_file = os.path.join(decoded_output_dir, f"traffic_{round_num}.txt")
            print(f"\n[Stage 2] Decoding perturbed images (round={round_num}) ...")
            run_decode({
                "rounds":       round_num,
                "input_images": output_dir,
                "csv_file":     packet_level_csv,
                "output_file":  traffic_file,
            })

            # Scores decoded traffic against the target model;
            # writes tracksheets_CH/dos_test_track_{round_num}.csv
            pred_output = os.path.join(prediction_output_dir, f"preds_{round_num}.csv")
            print(f"\n[Stage 3] Evaluating against target model (round={round_num}) ...")
            run_evaluate({
                "rounds":         round_num,
                "model_path":     cfg['target_model'],
                "traffic_path":   traffic_file,
                "tracksheet":     packet_level_csv,
                "output_path":    pred_output,
                "tracksheet_dir": tracksheet_dir,
            })

            # save_preds writes dos_test_track_{round_num}.csv into tracksheet_dir
            next_tracksheet = os.path.join(tracksheet_dir, f"dos_test_track_{round_num}.csv")

            # Updates the label file using the new tracksheet predictions
            updated_label_file = os.path.join(
                original_test_dir,
                f"labels_round_{round_num + 1}.txt"
            )
            print(f"\n[Stage 4] Updating label file (round={round_num}) ...")
            run_update({
                "tracksheet":         next_tracksheet,
                "label_file":         label_file,
                "updated_label_file": updated_label_file,
            })

            tracksheet = next_tracksheet
            label_file = updated_label_file
            # Perturbed images (perturbed_image_*.png) live in output_dir;
            # all rounds after 0 must read from there.
            current_test_dir = output_dir

            print(f"\n  Round {round_num} complete.")
            print(f"  Next tracksheet : {tracksheet}")
            print(f"  Next label file : {label_file}")
