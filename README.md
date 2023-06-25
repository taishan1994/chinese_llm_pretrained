# chinese_llm_pretrained
使用自己的tokenizer继续预训练大语言模型。为了方便起见，这里使用的模型为：IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese 。

更多介绍请查看知乎：

# 依赖

```python
transformers==4.30.2
accelerate==0.20.3
deepspeed==0.9.5
peft==0.3.0
datasets==2.13.1
evaluate==0.4.0
sentencepiece==0.1.99
```

# 一般步骤

## 准备数据

在data下放置自己的数据，可以是多个txt，每个txt里面每行为一句话或者几句话。这里使用的数据为斗破苍穹小说。

## 训练

```python
torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py \
--deepspeed ds_zero2_no_offload.json \
--model_name_or_path IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese \
--tokenizer_name_or_path IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese \
--dataset_dir data \
--data_cache_dir temp_data_cache_dir \
--validation_split_percentage 0.001 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 16 \
--do_train --seed $RANDOM \
--fp16 \
--max_steps 2500 \
--lr_scheduler_type cosine \
--learning_rate 2e-4 \
--warmup_ratio 0.05 \
--weight_decay 0.01 \
--logging_strategy steps \
--logging_steps 10 \
--save_strategy steps \
--save_total_limit 3 \
--save_steps 50 \
--gradient_accumulation_steps 1 \
--preprocessing_num_workers 8 \
--block_size 512 \
--output_dir output_dir \
--overwrite_output_dir \
--ddp_timeout 30000 \
--logging_first_step True \
--lora_rank 8 \
--lora_alpha 32 \
--trainable c_attn \
--modules_to_save transformer.wte,lm_head \
--lora_dropout 0.05 \
--torch_dtype float16 \
--gradient_checkpointing \
--ddp_find_unused_parameters False
```

## 使用模型

```python
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

"""
sentence 0: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 淡 淡 的 道 。 <|endoftext|> [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
********************
sentence 1: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 一 怔 。 手 掌 猛 然 一 僵 。 手 指 一 扯 。 旋 即 在 房 门 内 停 留 。 旋 即 一 口 鲜 血 喷 涌 而 出 。 <|endoftext|>
********************
sentence 2: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 顿 时 愣 了 愣 。 他 这 是 何 人 ？ 怎 能 知 道 这 位 灰 袍 老 者 出 手 啊 ？ <|endoftext|> [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
********************
sentence 3: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 心 中 有 着 什 么 感 触 ？ <|endoftext|> [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
********************
sentence 4: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 微 皱 着 眉 头 。 转 过 身 。 轻 声 道 ： “ 柳 翎 。 是 你 的 人 ？ ” <|endoftext|> [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
"""
```

对比原始模型结果：

```python
"""
sentence 0: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎, 男, 1964 年 生, 河 北 齐 齐 哈 尔 市 人 。 1979 年 毕 业 于 武 汉 工 学 院 中 文 系, 1988 年 毕 业 于 中 国 人 民 大 学 中 文 系, 历 任 中 国 人 民 大 学 高 级 教 师 、 教 育 部 大 学 文 学 系 主 任, 中 国 语 言 文 学 会 理 事, 中 国 人 民 大 学 历 史 学 会 副 会 长, 中 国 作 家 协 会 员, 中 国 作 家 协 会 会
********************
sentence 1: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 的 脸 庞 在 不 同 时 期 会 发 出 来 ， 这 样 的 眉 目 和 眉 目 能 够 很 容 易 的 在 一 起 ， 能 够 让 人 看 得 见 的 就 是 这 样 的 眉 目 。 那 一 对 情 侣 还 是 非 常 喜 欢 的 ， 不 过 他 们 的 交 往 方 式 也 是 各 种 多 样 的 ， 最 后 的 交 往 方 式 就 是 让 所 有 的 人 都 看 到 了 自 己 的 内 心 。 他 们 俩 是 非 常 相
********************
sentence 2: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 眼 睛 看 向 柳 翎, 眼 眸 里 满 是 伤 痕 。 “ 天 边 来 客 。 ” 柳 翎 那 无 情 的 目 光 中 透 着 几 分 冷 漠 的 微 笑 。 “ 没 有 你 的 名 字, 你 只 是 名 字 。 ” 柳 翎 在 柳 翎 眼 前 一 怔, 无 意 中 却 看 出 了 柳 翎 已 经 在 想 要 离 开 了 。 柳 翎 说 这 些 东 西 有 的 是 一 次 次 的 意 外, 她 还 是 有 意 的,
********************
sentence 3: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 的 脸 上 只 有 几 分 阴 沉, 但 却 能 够 带 着 微 微 的 怜 惜 之 心 。 萧 炎 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 眼 角
********************
sentence 4: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 已 经 是 年 轻 貌 美 的 人, 在 某 处 留 下 的 是 无 尽 的 光 影 。 她 的 微 笑 也 在 耳 畔 闪 烁 着 光 影 。 他 不 断 地 伸 出 手 指, 他 在 他 的 微 笑 中 轻 松 地 走 着, 而 柳 翎 却 始 终 沉 默 。 他 已 经 是 个 女 孩 子, 在 某 处 也 许 你 听 不 见 。 他 轻 轻 地 接 过 他 的 手, 轻 轻 地 说 道 : " 没 有 人 听
********************
"""
```

# 参考

> https://github.com/hiyouga/LLaMA-Efficient-Tuning
