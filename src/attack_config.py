
# Options for adv_attack : Blackbox , Whitebox or your custom attack type name
ADV_ATTACK_TYPE = "attack type here"  # e.g., Blackbox

# If adv_attack_type is Blackbox, then choose different surrogate and target models
SURROGATE_MODEL = "your surrogate model name here"  # e.g., Densenet161_demo
TARGET_MODEL =  "your target model name here"  # e.g., ResNet_demo

# conventional attack parameters for string matching in case of modification (type : string)
# ID : 11 bits and DLC : 4 bits
ID = "bits in string format here"  # e.g., 00000010101
DLC = "dlc bits in string format here"  # e.g., 1010

# Attack parameters 
EPSILON = 0.03  # Perturbation limit for adversarial attacks
MAX_INJECTION_LIMIT = 100  # Maximum number of messages that can be injected during an attack

