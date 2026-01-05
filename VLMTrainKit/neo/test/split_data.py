import json

if __name__ == "__main__":
    data_path = ""
    save_path = ""

    # Read JSONL file line by line
    data = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))

    # Take first 1 samples
    split_data = data[:1]

    # Write to JSONL file
    with open(save_path, "w") as f:
        for item in split_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
