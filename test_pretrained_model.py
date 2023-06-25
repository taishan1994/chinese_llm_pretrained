import os
import torch
from transformers import BertTokenizer,GPT2LMHeadModel, AutoModelForCausalLM
from peft import PeftModel
hf_model_path = 'IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese'
tokenizer = BertTokenizer.from_pretrained(hf_model_path)
# model = GPT2LMHeadModel.from_pretrained(hf_model_path)
model = AutoModelForCausalLM.from_pretrained(hf_model_path)

model_vocab_size = model.get_output_embeddings().weight.size(0)
model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(model, os.path.join("output_dir", "adapter_model"), torch_dtype=torch.float32)
model.cuda()
model.eval()

def generate_word_level(input_text,n_return=5,max_length=128,top_p=0.9):
    inputs = tokenizer(input_text,return_tensors='pt',add_special_tokens=False).to(model.device)
    gen = model.generate(
                            inputs=inputs['input_ids'],
                            max_length=max_length,
                            do_sample=True,
                            top_p=top_p,
                            eos_token_id=21133,
                            pad_token_id=0,
                            num_return_sequences=n_return)

    sentences = tokenizer.batch_decode(gen)
    for idx,sentence in enumerate(sentences):
        print(f'sentence {idx}: {sentence}')
        print('*'*20)
    return gen

outputs = generate_word_level('眼角斜瞥着柳翎那略微有些阴沉的脸庞。萧炎',n_return=5,max_length=128)
print(outputs)