from common_imports import (
    os, np, plt, datetime,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
)


def evaluation_metrics(all_preds, all_labels, cfg):
    if all_labels is None or all_preds is None:
        return

    print("Inside evaluation matrix")
    dir_path      = cfg['dir_path']
    dataset_name  = cfg['dataset_name']
    adv_attack    = cfg['adv_attack']
    adv_attack_type = cfg['adv_attack_type']
    model         = cfg['test_model']
    model_name    = cfg['test_model_name']

    dataset_path = os.path.join(dir_path, "..", "datasets", dataset_name)
    if adv_attack:
        result_path = os.path.join(dataset_path, "Results", adv_attack, adv_attack_type)
    else:
        result_path = os.path.join(dataset_path, "Results", model + "_" + model_name)
    os.makedirs(result_path, exist_ok=True)

    timestamp = datetime.now().strftime("_%Y_%m_%d_%H%M%S")

    if adv_attack:
        filename = f"{adv_attack_type}_dos_{timestamp}.png"
    else:
        filename = f"{model_name}_{timestamp}.png"

    print("Number of predictions:", len(all_preds))
    print("Unique predictions:", np.unique(all_preds, return_counts=True))
    print("Unique labels:", np.unique(all_labels, return_counts=True))

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')

    output_path = os.path.join(result_path, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    true_negatives  = cm[0, 0]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives  = cm[1, 1]

    IDS_accu   = accuracy_score(all_labels, all_preds)
    IDS_prec   = precision_score(all_labels, all_preds, zero_division=0)
    IDS_recall = recall_score(all_labels, all_preds, zero_division=0)
    IDS_F1     = f1_score(all_labels, all_preds, zero_division=0)

    print("----------------IDS Performance Metric----------------")
    print(f'Accuracy: {IDS_accu:.4f}')
    print(f'Precision: {IDS_prec:.4f}')
    print(f'Recall: {IDS_recall:.4f}')
    print(f'F1 Score: {IDS_F1:.4f}')

    if adv_attack:
        tnr = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
        mdr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        misclassified_attack_packets = ((all_labels == 1) & (all_preds == 0)).sum().item()
        total_attack_packets = (all_labels == 1).sum().item()

        oa_asr = misclassified_attack_packets / total_attack_packets
        print("----------------Adversarial Attack Performance Metric----------------")
        print("TNR:", tnr)
        print("Malicious Detection Rate:", mdr)
        print("Attack Success Rate:", oa_asr)
