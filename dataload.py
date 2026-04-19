from datasets import load_dataset

ds = load_dataset(
    "wmt/wmt17",
    "zh-en",
    split="train[:10000]",
)

print(ds)
print(ds[0])