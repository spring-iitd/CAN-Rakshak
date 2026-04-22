
import torch
import torch.nn.functional as F
import os
import time
import numpy as np
from datetime import datetime
from .attack_utilities import *
from .generate_mask import *
from evaluate import evaluation_metrics
import pandas as pd
import json






def load_model(test_model_path, surr_model_path):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(surr_model_path, map_location=device, weights_only = False)
    model.to(device)
    model.eval()   # optional but recommended if you only do inference

    test_model = None
    if test_model_path:
        test_model = torch.load(test_model_path, map_location=device, weights_only = False)
        test_model.to(device)
        test_model.eval()

    return model, test_model

def perturbation_constraints(image, perturbed_image,mask,existing_hex_ids, packet_level_data, image_no, flag):
    ID_len = 11
    mid_bits = "0001000"

    existing_int_ids = [int(h, 16) for h in existing_hex_ids]


    for b in range(image.shape[0]):
        rows = mask[b, 0].nonzero(as_tuple=True)[0]
        rows = torch.unique(rows)


        injection_row = rows.item()
        i = injection_row - 1
        packets_before_injection = []

        while i >= 0:
            first_pixel = image[b, 0, i, 0].item()  # First pixel in row i, channel 0
            second_pixel = image[b, 1, i, 0].item()  # Second pixel in row i, channel 1
            third_pixel = image[b, 2, i, 0].item()  # Third pixel in row i, channel 2
            if first_pixel == 0.0 and second_pixel == 0.0 and third_pixel == 0.0:
                packets_before_injection.append(i)
            i -= 1

        image_packets = packet_level_data[packet_level_data["image_no"] == image_no]
        target_index = len(packets_before_injection) - 1


        if flag == 'injection':
            start_row = packets_before_injection[0]
            end_row = injection_row

            red_pixel_count = 0
            for row_idx in range(start_row, end_row):
                red_pixels_mask = (
                    (perturbed_image[b, 0, row_idx, :] == 1.0) &  # Red channel is 1
                    (perturbed_image[b, 1, row_idx, :] == 0.0) &  # Green channel is 0
                    (perturbed_image[b, 2, row_idx, :] == 0.0)    # Blue channel is 0
                )
                red_pixel_count += red_pixels_mask.sum().item()


            timestamp = image_packets.iloc[target_index]["timestamp"]
            new_timestamp = timestamp + (injection_row-packets_before_injection[0])*128*0.000002 - red_pixel_count*0.000002
        
        for row in rows:
            decoded_bits = ''
            for col in range(1, 1 + ID_len):
                pix = perturbed_image[b, :, row, col]
                dot1 = torch.dot(pix, torch.tensor([1.0, 1.0, 1.0], device=image.device))
                dot0 = torch.dot(pix, torch.tensor([0.0, 0.0, 0.0], device=image.device))
                decoded_bits += '1' if dot1 >= dot0 else '0'

            gen_int = int(decoded_bits, 2)
            def hamming_dist(a, b, bitlen=ID_len):
                return bin(a ^ b).count('1')

            best_int = min(existing_int_ids,
                           key=lambda eid: hamming_dist(eid, gen_int, bitlen=ID_len))
            
            new_id = format(best_int, 'X')
        

            if flag == 'injection':
                start_index = packet_level_data.index[packet_level_data["image_no"] == image_no][0]
                df_part_1 = packet_level_data.iloc[:start_index+target_index+1]
                df_part_2 = packet_level_data.iloc[start_index+target_index+1:]
                packet_level_data = pd.concat([df_part_1, pd.DataFrame({"timestamp": [new_timestamp], "can_id": [new_id], "image_no": [image_no], "row_no": [injection_row],"valid_flag": [1], "label": [1], "perturbation_type": "I"}), df_part_2], ignore_index=True)
            elif flag == 'modification':   
                start_index = packet_level_data.index[packet_level_data["image_no"] == image_no][0]
                packet_level_data.loc[start_index + target_index+1, ["can_id", "perturbation_type"]] = [new_id, "M"]

            proj_bits = bin(best_int)[2:].zfill(ID_len)

            for idx, bit in enumerate(proj_bits, start=1):
                val = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, idx] = val

            if flag == 'modification':
                mid_bits = ''
                for col in range(1 + ID_len, 1 + ID_len + 7):
                    pix = perturbed_image[b, :, row, col]
                    bit = int((pix > 0.0).any().item())
                    mid_bits += str(bit)


            data_bits = ''
            start = 1 + ID_len + len(mid_bits)
            for col in range(start, start + 64):
                pix = perturbed_image[b, :, row, col]
                dot1 = torch.dot(pix, torch.tensor([1.0, 1.0, 1.0], device=image.device))
                dot0 = torch.dot(pix, torch.tensor([0.0, 0.0, 0.0], device=image.device))
                data_bits += '1' if dot1 >= dot0 else '0'

            frame_start = '0' + proj_bits + mid_bits + data_bits
            crc_val = calculate_crc(frame_start)
            crc_bits = bin(crc_val)[2:].zfill(15)
            
            stuffed = stuff_bits(frame_start + crc_bits)

            for i, bit in enumerate(stuffed):
                val = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, i] = val

            ending = '1011111111111'
            offset = len(stuffed)
            for i, bit in enumerate(ending):
                val = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, offset + i] = val

            for i in range(offset + len(ending), perturbed_image.shape[-1]):
                perturbed_image[b, 1, row, i] = 1.0
                perturbed_image[b, 0, row, i] = 0.0
                perturbed_image[b, 2, row, i] = 0.0


    return perturbed_image, packet_level_data

def fgsm_attack_injection(image, data_grad,ep,perturbation_type, existing_hex_ids, packet_level_data, image_no):
    sign_data_grad = data_grad.sign()
    print_image(sign_data_grad,0,1)

    if perturbation_type == "Random":
        mask = generate_multiple_mask_random(image, pack=1) 
    else:
        mask = generate_max_grad_mask(image, data_grad)

    if mask == None:
        return image, packet_level_data
    

    perturbed_image = image + ep * sign_data_grad*mask
    print_image(perturbed_image,0,1)
 
    perturbed_image, packet_level_data = perturbation_constraints(image, perturbed_image,mask,existing_hex_ids, packet_level_data, image_no,'injection')
    return perturbed_image, packet_level_data

def apply_injection(test_model,target,data_grad,data_denorm,ep,perturbation_type,existing_hex_ids, packet_level_data, image_no, feedback):
    
    perturbed_data, packet_level_data = fgsm_attack_injection(data_denorm, data_grad,ep,perturbation_type,existing_hex_ids, packet_level_data, image_no)

    if perturbed_data is None:
        print("No more space to inject")
        output = test_model(data_denorm)
        feedback += 1
        final_pred = output.max(1, keepdim=True)[1]
        return True, final_pred, data_denorm, packet_level_data, feedback 
    
    with torch.no_grad():
        output = test_model(perturbed_data)
        feedback += 1

    pred_probs = torch.softmax(output, dim=1)
    final_pred = output.max(1, keepdim=True)[1] # index of the maximum log-probability
   
    if final_pred.item() == target.item():
        return True, final_pred, perturbed_data, packet_level_data, feedback  # Indicate that we need to reapply
    else:
        return False, final_pred, perturbed_data, packet_level_data, feedback  # Indicate that we can stop

def fixed_id_data_perturbation(image, perturbed_image,mask,ID,Data):
    ID_len = 11
    middle_bits = "0001000"
    for b in range(image.shape[0]):

        rows = mask[b, 0].nonzero(as_tuple=True)[0]  # Identified row indices
        rows = torch.unique(rows)

        for row in rows:

            starting_bits = '0' + ID + middle_bits + Data
            
            crc_output = calculate_crc(starting_bits)
            crc_output = bin(crc_output)[2:].zfill(15)

            stuffing = starting_bits + crc_output
            stuffed_perturbation_bits = stuff_bits(stuffing)
            
            for i,bit in enumerate(stuffed_perturbation_bits):
                value = 1.0 if bit =='1' else 0.0
                perturbed_image[b, :, row, i] = value

            ending_part = '1011111111111'
            for i, bit in enumerate(ending_part):
                value = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, len(stuffed_perturbation_bits)+ i] = value
                
            for i in range(len(stuffed_perturbation_bits)+len(ending_part), 128):
                perturbed_image[b, 1, row, i] = 1.0  # Set green channel to maximum
                perturbed_image[b, 0, row, i] = 0.0
                perturbed_image[b, 2, row, i] = 0.0
           
    return perturbed_image

def fgsm_attack_modification(image,data_grad, epsilon,perturbation_type ,ID,Data, matched_rows,selected_rows_set,bit_pattern,existing_hex_ids, packet_level_data, n_image):
    sign_data_grad = data_grad.sign()


    mask,matched_rows,selected_rows_set = generate_mask_modify(image, data_grad,matched_rows,selected_rows_set,bit_pattern, perturbation_type)
    perturbed_image = image + epsilon * sign_data_grad * mask

    if perturbation_type == "Gradient":
        perturbed_image, packet_level_data = perturbation_constraints(image, perturbed_image,mask,existing_hex_ids, packet_level_data, n_image, 'modification')
    elif perturbation_type == "Random":
        perturbed_image, packet_level_data = perturbation_constraints(image, perturbed_image,mask,existing_hex_ids, packet_level_data, n_image, 'modification')
    
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image,matched_rows,selected_rows_set, packet_level_data

def apply_modification(test_model,target,data_grad,data_denorm,ep,perturbation_type,ID,Data,matched_rows,selected_rows_set,bit_pattern,existing_hex_ids, packet_level_data, n_image, feedback):
    
    perturbed_data,matched_rows,selected_rows_set,packet_level_data = fgsm_attack_modification(data_denorm,data_grad, ep,perturbation_type ,ID,Data, matched_rows,selected_rows_set,bit_pattern,existing_hex_ids, packet_level_data, n_image)
    
    with torch.no_grad():
        output = test_model(perturbed_data)
        feedback += 1

    pred_probs = torch.softmax(output, dim=1)
    final_pred = output.max(1, keepdim=True)[1] # index of the maximum log-probability

    if final_pred.item() == target.item():
        return True, final_pred, perturbed_data,matched_rows,selected_rows_set, packet_level_data, feedback  # Indicate that we need to reapply
    else:
        return False, final_pred, perturbed_data,matched_rows,selected_rows_set,packet_level_data, feedback  # Indicate that we can stop
    
def Attack_procedure(model, test_model, device, test_loader, injection_type, modification_type, ep, max_injection_perturbations, output_path, bit_pattern, existing_hex_ids, start_image_number, packet_level_data, adv_attack_type):
    all_preds = []
    all_labels = []
    n_image = start_image_number
    target_ID = "00100110000"
    target_Data = "1010100110111101010101001101100101001110101101110100101011001101"

    summary_path = os.path.join(output_path, "perturbation_summary.csv")
    print("Summary path : ", summary_path)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)  
    csv_file = open(summary_path, "w")
    csv_file.write("image_name, target_label, injection_count, modification_count, final_prediction_label, model_feedback\n")
    
    rgb_pattern = [(0.0, 0.0, 0.0) if bit == '0' else (1.0, 1.0, 1.0) for bit in bit_pattern]
    pattern_length = len(rgb_pattern)

    if adv_attack_type.lower() == "whitebox" and model is None:
        model = test_model

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        feedback = 0

        initial_output = test_model(data)
        feedback += 1
        final_pred = initial_output.max(1, keepdim=True)[1]
        injection_count = 0
        modification_count = 0
        if final_pred == 1:
            print("\nImage no:", n_image, "(Attack image)")
            
            data.requires_grad = True
            model.eval()
            
            initial_output = model(data)
            loss = F.nll_loss(initial_output, target)
            
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            
            data_denorm = data
            continue_perturbation = True
            matched_rows = None
            selected_rows_set = None
            perturbation_type = "injection"  # Start with injection
            _,max_modification_perturbations = find_max_perturbations(data_denorm,pattern_length,rgb_pattern,matched_rows,ifprint=False)
            print("max_modification_perturbations",max_modification_perturbations)

            while continue_perturbation:
                perturbed_data = data_denorm.clone().detach().to(device)
                perturbed_data.requires_grad = True
                model.eval()

                if perturbation_type == "injection" and injection_count < max_injection_perturbations:
                    continue_perturbation, final_pred, data_denorm, packet_level_data, feedback = apply_injection(
                        test_model, target, data_grad, perturbed_data, ep,injection_type,existing_hex_ids, packet_level_data, n_image, feedback
                    )
                    injection_count += 1
                    if continue_perturbation and modification_count < max_modification_perturbations:
                        perturbation_type = "modification"  # Switch to modification on failure
                elif perturbation_type == "modification" and modification_count < max_modification_perturbations:
                    continue_perturbation, final_pred, data_denorm,matched_rows,selected_rows_set, packet_level_data, feedback = apply_modification(
                        test_model, target, data_grad, perturbed_data, ep,modification_type,target_ID,target_Data,matched_rows,selected_rows_set,bit_pattern,existing_hex_ids, packet_level_data, n_image, feedback
                    )
                    modification_count += 1
                    if continue_perturbation and injection_count < max_injection_perturbations:
                        perturbation_type = "injection"  # Switch to injection on failure
                else:
                    if injection_count >= max_injection_perturbations and modification_count >= max_modification_perturbations:
                        continue_perturbation = False
                    elif injection_count < max_injection_perturbations:
                        perturbation_type = "injection"
                    elif modification_count < max_modification_perturbations:
                        perturbation_type = "modification"


            saving_image(data_denorm, n_image,output_path)
        else:
            data.requires_grad = True
            test_model.eval()
            initial_output = test_model(data)
            final_pred = initial_output.max(1, keepdim=True)[1]

            print(f"Image {n_image}: Benign Image (Skipping Perturbation)")
            saving_image(data, n_image,output_path)

        print(f"Final perturbations: Injection={injection_count}, Modification={modification_count}")
        print(f"Image {n_image}, Truth Labels {target.item()}, Final Pred {final_pred.cpu().numpy()}")

        all_preds.append(final_pred.item())
        all_labels.append(target.item())

        image_name = f"image_{n_image}.png"
        target_label = target.item()
        final_label = final_pred.item()

        csv_file.write(f"{image_name}, {target_label}, {injection_count}, {modification_count}, {final_label}, {feedback}\n")
        n_image += 1

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    csv_file.close()
    
    return all_preds, all_labels, packet_level_data
    
for_Carla = False

def safe_to_csv(df, path, **kwargs):
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)

    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path)

    df.to_csv(path, **kwargs)

def FGSM_attack(surr_model_path, test_model_path, cfg):
    print("inside FGSM attack function")
    print("Inside fgsm attack")

    dir_path              = cfg['dir_path']
    dataset_name          = cfg['dataset_name']
    file_name             = cfg['file_name']
    test_dataset_dir_name = cfg['test_dataset_dir']
    adv_attack            = cfg['adv_attack']
    adv_attack_type       = cfg['adv_attack_type']
    epsilon               = cfg['epsilon']
    can_id                = cfg['can_id']
    dlc                   = cfg['dlc']
    max_inj_limit         = cfg['max_injection_limit']

    dataset_path     = os.path.join(dir_path, "..", "datasets", dataset_name)
    test_dataset_dir = os.path.join(dataset_path, "test", test_dataset_dir_name)
    test_label_file  = os.path.join(test_dataset_dir, "labels.txt")
    


    timestamp   = datetime.now().strftime("_%Y_%m_%d_%H%M%S")
    output_path = os.path.join(dataset_path, "adversarial_images", f"{adv_attack}_{adv_attack_type}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)

    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    csv_file_path = os.path.join(dir_path, "..", "datasets", dataset_name, "test", test_dataset_dir_name, "track.csv")

    packet_level_data = pd.read_csv(csv_file_path)
    packet_level_data.columns = packet_level_data.columns.str.strip()
    packet_level_data["perturbation_type"] = "None"

    image_datasets, test_loader, start_image_number = load_dataset(test_dataset_dir, test_label_file, device, is_train=False)
    print("loaded test dataset")

    model, test_model = load_model(test_model_path, surr_model_path)

    injection_type    = "Gradient"
    modification_type = "Gradient"
    bit_pattern       = "0" + can_id + "000" + dlc

    dci_file_path = os.path.join(dir_path, "..", "datasets", dataset_name, "json_files", "distinct_can_ids.json")
    with open(dci_file_path, "r") as f:
        existing_hex_ids = json.load(f)

    max_perturbations_list = [max_inj_limit]
    st = time.time()
    print("Start time:", st)

    for max_injection_perturbations in max_perturbations_list:
        print("--------------------------------")
        print(f"Testing with max_injections  {max_injection_perturbations} and Injection_type {injection_type}")
        print(f"Testing with max_modification depending on each image and Modification_type {modification_type}")

        preds, labels, packet_level_data = Attack_procedure(
            model, test_model, device, test_loader,
            injection_type, modification_type, epsilon,
            max_injection_perturbations, output_path, bit_pattern,
            existing_hex_ids, start_image_number, packet_level_data,
            adv_attack_type,
        )
        et = time.time()
        print("End time:", et)

    packet_csv_file_dir = os.path.join(dataset_path, "csv_files", file_name[:-4] + "_blackbox_final_rgb_random")
    packet_csv_file     = os.path.join(packet_csv_file_dir, "packet_level_data.csv")
    os.makedirs(packet_csv_file_dir, exist_ok=True)
    safe_to_csv(packet_level_data, packet_csv_file, index=False)

    return preds, labels, output_path

