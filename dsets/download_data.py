
#downloads the data to DFS
data_dir = "data"
#snips = AttributeSnippets(data_dir)

files = ["attribute_snippets.json", "counterfact.json", "idf.npy", "known_1000.json", "multi_counterfact.json", "tfidf_vocab.json", "zsre_mend_eval.json", "zsre_mend_train.json"]

for file in files:
    url = f"{REMOTE_ROOT_URL}/data/dsets/{file}"
    loc = f"{data_dir}/{file}"
    torch.hub.download_url_to_file(url, loc)


#vec = get_tfidf_vectorizer(data_dir)
