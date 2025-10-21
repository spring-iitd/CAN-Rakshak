import os

from get_ids import get_model

def test_model(modelName, modelPath, adv_attack,image = None, TestSplit = None ):
    """Tests the specified IDS model unless an adversarial attack is specified."""
    if(adv_attack):
        return 
    model = get_model(modelName)
    print(f"Loading model from {os.path.normpath(modelPath)}")
    model.load(modelPath)    
    model.test()
    print("Testing Completed")