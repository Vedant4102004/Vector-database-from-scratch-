from datasets import load_dataset

dataset = load_dataset("fancyzhx/ag_news")

print(dataset)

for i in range(5):
    print("\n---")
    print(dataset["train"][i]["text"])