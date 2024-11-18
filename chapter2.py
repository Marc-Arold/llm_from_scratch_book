
with open("the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()

print("Total number of character: ", len(raw_text))

print(raw_text[:99])

import re

from numpy import integer

text = raw_text[:200]

preprocessed = re.split(r'([,.:;?_!"(){}\']--|\s)', raw_text)



# we may need to remove the whitespace

result = [item.strip() for item in preprocessed if item.strip()]
print(len(result))

all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words)

print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
  print(item)
  if i>50:
    break

class SimpleTokenizerV1:
  def __init__(self, vocab):
    self.str_to_int = vocab
    self.int_to_str = {i:s for s,i in vocab.items()}
  
  def encode (self, text):
    preprocessed = re.split(r'([,.:;?_!"(){}\']--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    ids = [self.str_to_int[s] for s in preprocessed]
    return ids
  
  def decode(self, ids):
    text = " ".join([self.int_to_str[i] for i in ids])
    text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
    return text
  

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said"""
ids= tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

class SimpleTokenizerV2:
  def __init__(self, vocab):
    self.str_to_int = vocab
    self.int_to_str = {i:s for s,i in vocab.items()}
  
  def encode (self, text):
    preprocessed = re.split(r'([,.:;?_!"(){}\']--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
    ids = [self.str_to_int[s] for s in preprocessed]
    return ids
  
  def decode(self, ids):
    text = " ".join([self.int_to_str[i] for i in ids])
    text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
    return text
  
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

# tokenizer = SimpleTokenizerV2(vocab)
# print(tokenizer.encode(text))
from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))

tokenizer =  tiktoken.get_encoding("gpt2")

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings =  tokenizer.decode(integers)
print(strings)

text_unknown = "Akwiew ier"
integers2 = tokenizer.encode(text_unknown)
print(integers2)
strings2 =  tokenizer.decode(integers2)
print("string 2",strings2)


with open("the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text2 = f.read()

enc_text = tokenizer.encode(raw_text2)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:     {y}")

for i in range(1, context_size+1):
  context = enc_sample[:i]
  desired = enc_sample[i]
  print(context, "---->", desired)


import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
  def __init__(self, txt, tokenizer, max_length, stride):
    self.tokenizer = tokenizer
    self.input_ids = []
    self.target_ids = []
    token_ids = tokenizer.encode(txt)

    for i in range(0, len(token_ids)-max_length, stride):
      input_chunk = token_ids[i:i+max_length]
      target_chunk = token_ids[i+1: i+max_length+1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))
  
  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]
  

def create_dataloader_v1(txt, batch_size=4, max_length =256, stride=128, shuffle=True, drop_last = True):
  tokenizer = tiktoken.get_encoding("gpt2")
  dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
  return dataloader

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False, drop_last=True)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)