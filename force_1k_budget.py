import glob
import yaml

files = glob.glob("configs/matrix/*.yaml")
print(f"Found {len(files)} config files.")

for fpath in files:
    with open(fpath, "r") as f:
        data = yaml.safe_load(f)
    
    # Force 1k budget
    if "budget" not in data:
        data["budget"] = {}
        
    data["budget"]["max_budget"] = 1000
    data["budget"]["checkpoints"] = [1000]
    
    with open(fpath, "w") as f:
        yaml.dump(data, f, sort_keys=False)

print(f"Updated all config files to 1k budget.")