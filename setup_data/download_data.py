##run on headnode

import torch
from pathlib import Path
#downloads the data to DFS
data_dir = "data"
#snips = AttributeSnippets(data_dir)

files = ["attribute_snippets.json", "counterfact.json", "idf.npy", "known_1000.json", "multi_counterfact.json", "tfidf_vocab.json", "zsre_mend_eval.json", "zsre_mend_train.json"]

for file in files:
    data_dir_path = Path(data_dir)
    loc = data_dir_path / file
    url = f"https://memit.baulab.info/data/dsets/{file}"
    if not loc.exists():
        print(f"{loc} does not exist. Downloading from {url}")
        data_dir_path.mkdir(exist_ok=True, parents=True)
        torch.hub.download_url_to_file(url, loc)