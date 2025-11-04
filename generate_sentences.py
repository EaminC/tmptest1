#!/usr/bin/env python3
"""
生成20个测试句子，每句正好20个token
用于保证实验结果的统一性
"""
import torch
import numpy as np
from transformers import GPT2Tokenizer
import json
import os

def generate_sentences(tokenizer, num_sentences=20, target_tokens=20):
    """生成指定数量的句子，每句正好target_tokens个token"""
    sentences = []
    
    # 直接使用token ID列表，不进行解码编码，确保正好target_tokens个token
    for i in range(num_sentences):
        # 随机选择token ID，确保正好target_tokens个
        token_ids = []
        for _ in range(target_tokens):
            # 从有效token ID范围中选择
            token_id = np.random.randint(0, tokenizer.vocab_size)
            token_ids.append(token_id)
        
        # 直接解码token_ids，确保正好20个token
        sentence = tokenizer.decode(token_ids)
        
        # 验证token数量
        encoded = tokenizer.encode(sentence, add_special_tokens=False)
        if len(encoded) != target_tokens:
            # 如果解码后token数变化（通常不会发生），使用token_ids直接作为输入
            # 这种情况下，我们直接使用token_ids，不进行解码
            sentences.append(token_ids)  # 存储为token_ids列表
        else:
            sentences.append(sentence)
    
    return sentences

def main():
    print("Generating test sentences...")
    print("Loading GPT-2 tokenizer...")
    
    # 加载tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 生成20个句子，每句20个token
    sentences = generate_sentences(tokenizer, num_sentences=20, target_tokens=20)
    
    # 验证所有句子都是20个token
    print("\nValidating sentences...")
    valid_sentences = []
    for idx, sentence in enumerate(sentences):
        if isinstance(sentence, list):
            # 如果是token_ids列表
            token_count = len(sentence)
            sentence_text = tokenizer.decode(sentence)
        else:
            # 如果是字符串
            encoded = tokenizer.encode(sentence, add_special_tokens=False)
            token_count = len(encoded)
            sentence_text = sentence
        
        if token_count == 20:
            valid_sentences.append(sentence)
            print(f"  Sentence {idx+1}: {token_count} tokens - {sentence_text[:50]}...")
        else:
            print(f"  WARNING: Sentence {idx+1} has {token_count} tokens, skipping")
    
    if len(valid_sentences) < 20:
        print(f"\nERROR: Only {len(valid_sentences)} valid sentences generated, need 20")
        return
    
    # 保存句子到文件
    output_file = 'sentences.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(valid_sentences, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Generated {len(valid_sentences)} sentences")
    print(f"✓ Saved to: {output_file}")
    print("\nYou can now run the experiment with: python3 experiment.py")

if __name__ == "__main__":
    main()

