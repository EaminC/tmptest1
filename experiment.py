#!/usr/bin/env python3
"""
GPT-2 Prefill阶段多精度误差分析实验
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Tuple, Any
import json
import os
from dataclasses import dataclass
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import platform
import re
warnings.filterwarnings('ignore')

# 设置字体（使用英文，不需要中文字体）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class PrecisionConfig:
    name: str
    dtype: torch.dtype
    device: str

def get_device():
    """获取可用设备，优先GPU"""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def get_hardware_name():
    """获取硬件名称，用于创建结果文件夹"""
    device = get_device()
    
    if device == 'cuda':
        # 获取GPU名称
        gpu_name = torch.cuda.get_device_name(0)
        # 清理名称，移除特殊字符，适合用作文件夹名
        hardware_name = re.sub(r'[^\w\s-]', '', gpu_name).strip()
        hardware_name = re.sub(r'[-\s]+', '-', hardware_name)
        return f"GPU-{hardware_name}"
    else:
        # 获取CPU信息
        cpu_info = platform.processor()
        if not cpu_info or cpu_info == '':
            # 如果processor()返回空，尝试从uname获取
            cpu_info = platform.machine()
        # 清理名称
        hardware_name = re.sub(r'[^\w\s-]', '', cpu_info).strip()
        hardware_name = re.sub(r'[-\s]+', '-', hardware_name)
        if not hardware_name:
            hardware_name = "CPU-Unknown"
        return f"CPU-{hardware_name}"

# generate_sentences函数已移动到generate_sentences.py中

def extract_kv_cache_and_attention(model, inputs, device):
    """提取KV cache和attention权重"""
    kv_cache_dict = {}
    attention_dict = {}
    
    # 使用hooks来捕获KV cache（key和value）
    hooks = []
    
    def make_kv_hook(layer_idx, storage_dict):
        def hook(module, input, output):
            # GPT2Attention的forward返回(attn_output, attn_weights, present)
            # 在prefill阶段，present是(key, value)的元组
            if isinstance(output, tuple):
                if len(output) >= 3:
                    present = output[2]  # present是(key, value)的元组
                    if present is not None and isinstance(present, tuple):
                        if len(present) >= 2:
                            key, value = present[0], present[1]
                            # 确保 key 和 value 不是 None，并且有 detach 方法
                            if key is not None and value is not None:
                                if hasattr(key, 'detach') and hasattr(value, 'detach'):
                                    try:
                                        storage_dict[f'layer_{layer_idx}_key'] = key.detach().cpu()
                                        storage_dict[f'layer_{layer_idx}_value'] = value.detach().cpu()
                                    except AttributeError:
                                        pass  # 如果 detach 失败，跳过
        return hook
    
    # 注册hooks到transformer blocks的attention层
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.attn.register_forward_hook(make_kv_hook(i, kv_cache_dict)))
    
    # 运行前向传播
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True, use_cache=True)
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    # 提取attention weights
    if hasattr(outputs, 'attentions') and outputs.attentions:
        for layer_idx, attn in enumerate(outputs.attentions):
            # attn shape: (batch, num_heads, seq_len, seq_len)
            # 检查 attn 是否为 None
            if attn is not None and hasattr(attn, 'detach'):
                try:
                    attention_dict[f'layer_{layer_idx}_full'] = attn.detach().cpu()
                except AttributeError:
                    # 如果 detach 失败，跳过这一层
                    continue
    
    # 如果hooks没有捕获到，尝试从past_key_values获取
    if not kv_cache_dict and hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
        past_key_values = outputs.past_key_values
        # 处理 DynamicCache 或其他缓存类型
        # DynamicCache 支持索引访问，返回 (key, value) 元组
        for layer_idx in range(len(past_key_values)):
            try:
                layer_cache = past_key_values[layer_idx]
                # past_key_values[layer_idx] 返回 (key, value) 元组
                if isinstance(layer_cache, tuple) and len(layer_cache) >= 2:
                    key = layer_cache[0]
                    value = layer_cache[1]
                    # 确保 key 和 value 不是 None，并且是 Tensor
                    if key is not None and value is not None:
                        # 检查是否是 Tensor 类型（有 detach 方法）
                        if hasattr(key, 'detach') and hasattr(value, 'detach'):
                            try:
                                kv_cache_dict[f'layer_{layer_idx}_key'] = key.detach().cpu()
                                kv_cache_dict[f'layer_{layer_idx}_value'] = value.detach().cpu()
                            except AttributeError:
                                # 如果 detach 失败，跳过这一层
                                continue
                        else:
                            # 如果 key 或 value 没有 detach 方法，跳过
                            continue
                    else:
                        # key 或 value 是 None，跳过这一层
                        continue
            except (IndexError, TypeError, AttributeError) as e:
                # 如果访问失败，跳过这一层
                continue
    
    return kv_cache_dict, attention_dict

def compute_error(reference, target):
    """计算误差（相对误差和绝对误差）"""
    if reference.shape != target.shape:
        return None, None
    
    # 绝对误差
    abs_error = torch.abs(reference - target)
    
    # 相对误差（避免除零）
    rel_error = torch.abs(reference - target) / (torch.abs(reference) + 1e-10)
    
    # 返回均值
    mean_abs_error = abs_error.mean().item()
    mean_rel_error = rel_error.mean().item()
    
    return mean_abs_error, mean_rel_error

def process_sentence_parallel(args):
    """并行处理单个句子的函数"""
    (sentence_idx, sentence, tokenizer, model_name, device, precision_configs, ref_model) = args
    
    # 处理句子（可能是字符串或token_ids列表）
    if isinstance(sentence, list):
        input_ids = torch.tensor([sentence], dtype=torch.long).to(device)
        sentence_text = tokenizer.decode(sentence)
    else:
        sentence_text = sentence
        inputs_dict = tokenizer(sentence, return_tensors="pt", padding=False)
        input_ids = inputs_dict['input_ids'].to(device)
    
    seq_len = input_ids.shape[1]
    
    # 验证token数量
    if seq_len != 20:
        return None
    
    # 构建inputs字典
    inputs = {'input_ids': input_ids}
    
    # 提取参考结果（FP32）
    try:
        ref_kv_cache, ref_attention = extract_kv_cache_and_attention(ref_model, inputs, device)
    except Exception as e:
        import traceback
        print(f"Error extracting reference KV cache for sentence {sentence_idx+1}: {e}")
        print(f"Traceback: {''.join(traceback.format_exception(type(e), e, e.__traceback__))}")
        return None
    
    # 检查是否成功提取
    if not ref_kv_cache:
        print(f"Warning: No reference KV cache extracted for sentence {sentence_idx+1}")
        return None
    
    sentence_results = {}
    
    # 对每个精度进行测试
    for prec_config in precision_configs:
        if prec_config.name == "FP32":
            continue  # 跳过参考精度
        
        # 加载模型到指定精度（每个进程加载自己的模型副本）
        # 使用 dtype 参数直接指定精度，避免 meta tensor 问题
        try:
            test_model = GPT2LMHeadModel.from_pretrained(
                model_name,
                dtype=prec_config.dtype,
                low_cpu_mem_usage=True
            )
            test_model = test_model.to(device)
            test_model.eval()
        except Exception as e:
            # 如果直接指定 dtype 失败，尝试先加载再转换
            try:
                test_model = GPT2LMHeadModel.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True
                )
                test_model = test_model.to(device)
                test_model = test_model.to(dtype=prec_config.dtype)
                test_model.eval()
            except Exception as e2:
                print(f"Error loading model for {prec_config.name}: {e2}")
                continue
        
        # 转换输入到对应精度（input_ids不需要转换）
        test_inputs = inputs.copy()
        
        # 提取测试结果
        try:
            test_kv_cache, test_attention = extract_kv_cache_and_attention(test_model, test_inputs, device)
        except Exception as e:
            print(f"Error extracting test KV cache for sentence {sentence_idx+1}, precision {prec_config.name}: {e}")
            del test_model
            if device == 'cuda':
                torch.cuda.empty_cache()
            continue
        
        # 检查是否成功提取
        if not test_kv_cache:
            print(f"Warning: No test KV cache extracted for sentence {sentence_idx+1}, precision {prec_config.name}")
            del test_model
            if device == 'cuda':
                torch.cuda.empty_cache()
            continue
        
        # 清理模型以释放内存
        del test_model
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # 计算误差
        kv_errors_by_token = []
        attn_errors_by_token = []
        
        # 遍历每一层，计算key和value的误差
        for layer_idx in range(len(ref_model.transformer.h)):
            # 计算Key的误差
            ref_key_key = f'layer_{layer_idx}_key'
            test_key_key = f'layer_{layer_idx}_key'
            
            if ref_key_key in ref_kv_cache and test_key_key in test_kv_cache:
                ref_key = ref_kv_cache[ref_key_key].float()
                test_key = test_kv_cache[test_key_key].float()
                
                if len(ref_key.shape) == 4:
                    for token_idx in range(seq_len):
                        ref_token_key = ref_key[0, :, token_idx, :].float()
                        test_token_key = test_key[0, :, token_idx, :].float()
                        abs_err, rel_err = compute_error(ref_token_key, test_token_key)
                        if abs_err is not None:
                            kv_errors_by_token.append({
                                'layer': layer_idx,
                                'token': token_idx,
                                'type': 'key',
                                'abs_error': abs_err,
                                'rel_error': rel_err
                            })
            
            # 计算Value的误差
            ref_value_key = f'layer_{layer_idx}_value'
            test_value_key = f'layer_{layer_idx}_value'
            
            if ref_value_key in ref_kv_cache and test_value_key in test_kv_cache:
                ref_value = ref_kv_cache[ref_value_key]
                test_value = test_kv_cache[test_value_key]
                
                # 检查是否为 None
                if ref_value is None or test_value is None:
                    continue
                
                ref_value = ref_value.float()
                test_value = test_value.float()
                
                if len(ref_value.shape) == 4:
                    for token_idx in range(seq_len):
                        ref_token_value = ref_value[0, :, token_idx, :].float()
                        test_token_value = test_value[0, :, token_idx, :].float()
                        abs_err, rel_err = compute_error(ref_token_value, test_token_value)
                        if abs_err is not None:
                            kv_errors_by_token.append({
                                'layer': layer_idx,
                                'token': token_idx,
                                'type': 'value',
                                'abs_error': abs_err,
                                'rel_error': rel_err
                            })
        
        # 计算Attention误差
        for layer_idx in range(len(ref_model.transformer.h)):
            ref_attn_key = f'layer_{layer_idx}_full'
            if ref_attn_key in ref_attention:
                ref_attn = ref_attention[ref_attn_key]
                test_attn_key = f'layer_{layer_idx}_full'
                
                if test_attn_key in test_attention:
                    test_attn = test_attention[test_attn_key]
                    
                    # 检查是否为 None
                    if ref_attn is None or test_attn is None:
                        continue
                    
                    if len(ref_attn.shape) == 4:
                        for token_idx in range(seq_len):
                            ref_token_attn = ref_attn[0, :, token_idx, :].float()
                            test_token_attn = test_attn[0, :, token_idx, :].float()
                            abs_err, rel_err = compute_error(ref_token_attn, test_token_attn)
                            if abs_err is not None:
                                attn_errors_by_token.append({
                                    'layer': layer_idx,
                                    'token': token_idx,
                                    'abs_error': abs_err,
                                    'rel_error': rel_err
                                })
        
        sentence_results[prec_config.name] = {
            'kv_cache_errors': kv_errors_by_token,
            'attention_errors': attn_errors_by_token
        }
    
    return (sentence_idx, sentence_text, sentence_results)

def run_experiment():
    """运行主实验"""
    device = get_device()
    hardware_name = get_hardware_name()
    print(f"Device: {device}")
    print(f"Hardware: {hardware_name}")
    
    # 创建结果文件夹
    output_dir = os.path.join('results', hardware_name)
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 加载模型和tokenizer
    print("Loading GPT-2 model...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 定义精度配置（尽可能多的精度）
    precision_configs = [
        PrecisionConfig("FP32", torch.float32, device),
        PrecisionConfig("FP16", torch.float16, device),
    ]
    
    # 如果支持BF16，添加它（即使CPU也尝试，虽然可能不支持）
    if device == 'cuda' and torch.cuda.is_bf16_supported():
        precision_configs.append(PrecisionConfig("BF16", torch.bfloat16, device))
    elif device == 'cpu':
        # CPU上也可以尝试BF16，虽然可能性能不好
        try:
            # 检查CPU是否支持BF16
            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16)
            precision_configs.append(PrecisionConfig("BF16", torch.bfloat16, device))
        except:
            pass
    
    # 尝试添加更多精度：FP64（double precision）
    # 注意：FP64在GPU上通常不被支持或性能很差，但我们可以尝试
    if device == 'cpu':
        # CPU上可以测试FP64
        precision_configs.append(PrecisionConfig("FP64", torch.float64, device))
    elif device == 'cuda':
        # 在GPU上尝试FP64（虽然可能很慢或不支持）
        try:
            test_tensor = torch.tensor([1.0], dtype=torch.float64).to(device)
            precision_configs.append(PrecisionConfig("FP64", torch.float64, device))
        except:
            # GPU可能不支持FP64，跳过
            pass
    
    precision_configs = [p for p in precision_configs if p is not None]
    
    # 读取已生成的句子
    sentences_file = 'sentences.json'
    if not os.path.exists(sentences_file):
        print(f"ERROR: {sentences_file} not found!")
        print("Please run generate_sentences.py first to generate test sentences.")
        return
    
    print(f"Loading sentences from {sentences_file}...")
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = json.load(f)
    
    if len(sentences) != 20:
        print(f"WARNING: Expected 20 sentences, found {len(sentences)}")
    
    # 验证句子
    valid_sentences = []
    for idx, sentence in enumerate(sentences):
        if isinstance(sentence, list):
            token_count = len(sentence)
        else:
            encoded = tokenizer.encode(sentence, add_special_tokens=False)
            token_count = len(encoded)
        
        if token_count == 20:
            valid_sentences.append(sentence)
        else:
            print(f"WARNING: Sentence {idx+1} has {token_count} tokens, skipping")
    
    if len(valid_sentences) < 20:
        print(f"ERROR: Only {len(valid_sentences)} valid sentences, need 20")
        return
    
    sentences = valid_sentences
    print(f"Loaded {len(sentences)} valid sentences (20 tokens each)")
    
    # 保存句子到结果目录（用于记录）
    sentences_path = os.path.join(output_dir, 'sentences.json')
    with open(sentences_path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)
    
    # 以FP32作为参考（最高精度）
    print("Loading FP32 reference model...")
    try:
        ref_model = GPT2LMHeadModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        ref_model = ref_model.to(device)
        ref_model.eval()
    except Exception as e:
        # 如果直接指定 dtype 失败，尝试默认方式
        ref_model = GPT2LMHeadModel.from_pretrained(
            model_name,
            low_cpu_mem_usage=True
        )
        ref_model = ref_model.to(device)
        ref_model.eval()
    
    # 存储所有结果
    all_results = {}
    for prec_config in precision_configs:
        if prec_config.name != "FP32":
            all_results[prec_config.name] = {
                'kv_cache_errors': [],
                'attention_errors': []
            }
    
    print(f"Precision configs: {[p.name for p in precision_configs if p.name != 'FP32']}")
    
    # 并行处理句子
    print(f"\nStarting parallel processing of {len(sentences)} sentences...")
    max_workers = min(4, len(sentences))  # 限制并发数，避免GPU内存不足
    if device == 'cuda':
        # GPU上使用较少的workers，避免内存溢出
        max_workers = min(2, len(sentences))
    else:
        # CPU上可以使用更多workers
        max_workers = min(mp.cpu_count(), len(sentences))
    
    print(f"Using {max_workers} parallel workers")
    
    # 准备任务参数
    tasks = [(i, sentence, tokenizer, model_name, device, 
              [p for p in precision_configs if p.name != "FP32"], ref_model) 
             for i, sentence in enumerate(sentences)]
    
    # 使用线程池并行处理（GPU计算时，线程池通常比进程池更高效）
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_sentence_parallel, task): task 
                         for task in tasks}
        
        for future in as_completed(future_to_task):
            completed += 1
            try:
                result = future.result()
                if result is None:
                    continue
                
                sentence_idx, sentence_text, sentence_results = result
                print(f"Completed sentence {sentence_idx+1}/20: {sentence_text[:50]}... ({completed}/{len(sentences)})")
                
                # 聚合结果
                for prec_name, prec_results in sentence_results.items():
                    all_results[prec_name]['kv_cache_errors'].append(prec_results['kv_cache_errors'])
                    all_results[prec_name]['attention_errors'].append(prec_results['attention_errors'])
                    
            except Exception as e:
                task = future_to_task[future]
                print(f"Error processing sentence {task[0]+1}: {e}")
    
    # 清理参考模型
    del ref_model
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # 生成可视化图表和统计结果
    print("\nGenerating plots and statistics...")
    generate_error_plots(all_results, sentences, plots_dir)
    generate_summary_plot(all_results, plots_dir)
    generate_error_summary_json(all_results, output_dir)
    
    print(f"\nExperiment completed! Results saved to: {output_dir}")

def generate_error_plots(all_results, sentences, plots_dir):
    """生成误差变化图"""
    os.makedirs(plots_dir, exist_ok=True)
    
    for prec_name, results in all_results.items():
        print(f"  Generating plot for {prec_name}...")
        
        # 聚合所有句子的数据
        kv_errors_by_token = {}
        attn_errors_by_token = {}
        
        # 聚合KV cache误差（key和value合并）
        for sentence_errors in results['kv_cache_errors']:
            for err_data in sentence_errors:
                token_idx = err_data['token']
                if token_idx not in kv_errors_by_token:
                    kv_errors_by_token[token_idx] = []
                kv_errors_by_token[token_idx].append(err_data['abs_error'])
        
        # 聚合Attention误差
        for sentence_errors in results['attention_errors']:
            for err_data in sentence_errors:
                token_idx = err_data['token']
                if token_idx not in attn_errors_by_token:
                    attn_errors_by_token[token_idx] = []
                attn_errors_by_token[token_idx].append(err_data['abs_error'])
        
        # 计算每个token位置的平均误差
        token_positions = sorted(kv_errors_by_token.keys())
        if not token_positions:
            print(f"    Warning: {prec_name} has no KV cache error data")
            continue
        
        kv_mean_errors = [np.mean(kv_errors_by_token[pos]) for pos in token_positions]
        kv_std_errors = [np.std(kv_errors_by_token[pos]) for pos in token_positions]
        
        token_positions_attn = sorted(attn_errors_by_token.keys())
        has_attn_data = bool(token_positions_attn)
        
        if has_attn_data:
            attn_mean_errors = [np.mean(attn_errors_by_token[pos]) for pos in token_positions_attn]
            attn_std_errors = [np.std(attn_errors_by_token[pos]) for pos in token_positions_attn]
        else:
            print(f"    Warning: {prec_name} has no Attention error data, will only plot KV cache")
        
        # 创建图表 - 如果有 attention 数据创建两个子图，否则只创建 KV cache 图
        if has_attn_data:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
            ax2 = None
        
        # KV Cache误差图
        ax1.plot(token_positions, kv_mean_errors, 'b-o', label='Mean Absolute Error', linewidth=2, markersize=6)
        ax1.fill_between(token_positions, 
                         [m - s for m, s in zip(kv_mean_errors, kv_std_errors)],
                         [m + s for m, s in zip(kv_mean_errors, kv_std_errors)],
                         alpha=0.3, color='blue', label='±1 Std Dev')
        ax1.set_xlabel('Token Position', fontsize=12)
        ax1.set_ylabel('Absolute Error', fontsize=12)
        ax1.set_title(f'{prec_name} - KV Cache Error vs Token Position', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        # 设置x轴只显示整数
        ax1.set_xticks(token_positions)
        ax1.set_xticklabels([int(x) for x in token_positions])
        
        # Attention Kernel误差图（如果有数据）
        if has_attn_data and ax2 is not None:
            ax2.plot(token_positions_attn, attn_mean_errors, 'r-o', label='Mean Absolute Error', linewidth=2, markersize=6)
            ax2.fill_between(token_positions_attn,
                             [m - s for m, s in zip(attn_mean_errors, attn_std_errors)],
                             [m + s for m, s in zip(attn_mean_errors, attn_std_errors)],
                             alpha=0.3, color='red', label='±1 Std Dev')
            ax2.set_xlabel('Token Position', fontsize=12)
            ax2.set_ylabel('Absolute Error', fontsize=12)
            ax2.set_title(f'{prec_name} - Attention Kernel Error vs Token Position', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            # 设置x轴只显示整数
            ax2.set_xticks(token_positions_attn)
            ax2.set_xticklabels([int(x) for x in token_positions_attn])
        
        plt.tight_layout()
        output_path = os.path.join(plots_dir, f'error_analysis_{prec_name}.jpg')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {output_path}")

def generate_summary_plot(all_results, plots_dir):
    """生成综合图表，显示所有精度在一张图上"""
    if not all_results:
        print("    Warning: No results to plot")
        return
    
    print("  Generating summary plot with all precisions...")
    
    # 收集所有精度的数据
    precision_data = {}
    
    for prec_name, results in all_results.items():
        # 聚合KV cache误差
        kv_errors_by_token = {}
        attn_errors_by_token = {}
        
        for sentence_errors in results['kv_cache_errors']:
            for err_data in sentence_errors:
                token_idx = err_data['token']
                if token_idx not in kv_errors_by_token:
                    kv_errors_by_token[token_idx] = []
                kv_errors_by_token[token_idx].append(err_data['abs_error'])
        
        for sentence_errors in results['attention_errors']:
            for err_data in sentence_errors:
                token_idx = err_data['token']
                if token_idx not in attn_errors_by_token:
                    attn_errors_by_token[token_idx] = []
                attn_errors_by_token[token_idx].append(err_data['abs_error'])
        
        if kv_errors_by_token:
            token_positions = sorted(kv_errors_by_token.keys())
            kv_mean_errors = [np.mean(kv_errors_by_token[pos]) for pos in token_positions]
            
            token_positions_attn = sorted(attn_errors_by_token.keys()) if attn_errors_by_token else []
            attn_mean_errors = [np.mean(attn_errors_by_token[pos]) for pos in token_positions_attn] if token_positions_attn else []
            
            precision_data[prec_name] = {
                'kv_positions': token_positions,
                'kv_errors': kv_mean_errors,
                'attn_positions': token_positions_attn,
                'attn_errors': attn_mean_errors
            }
    
    if not precision_data:
        print("    Warning: No valid precision data for summary plot")
        return
    
    # 检查是否有 attention 数据
    has_any_attn_data = any(data['attn_positions'] for data in precision_data.values())
    
    # 创建综合图表 - 如果有 attention 数据创建两个子图，否则只创建 KV cache 图
    if has_any_attn_data:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 5))
        ax2 = None
    
    # 颜色列表
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    # KV Cache综合图
    for idx, (prec_name, data) in enumerate(precision_data.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax1.plot(data['kv_positions'], data['kv_errors'], 
                marker=marker, color=color, label=prec_name, 
                linewidth=2, markersize=6)
    
    ax1.set_xlabel('Token Position', fontsize=12)
    ax1.set_ylabel('Absolute Error', fontsize=12)
    ax1.set_title('KV Cache Error vs Token Position (All Precisions)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    if precision_data:
        # 获取所有token位置并合并
        all_positions = set()
        for data in precision_data.values():
            all_positions.update(data['kv_positions'])
        token_positions = sorted(all_positions)
        ax1.set_xticks(token_positions)
        ax1.set_xticklabels([int(x) for x in token_positions])
    
    # Attention Kernel综合图（如果有数据）
    if has_any_attn_data and ax2 is not None:
        for idx, (prec_name, data) in enumerate(precision_data.items()):
            if data['attn_positions']:  # 只绘制有 attention 数据的精度
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                ax2.plot(data['attn_positions'], data['attn_errors'], 
                        marker=marker, color=color, label=prec_name, 
                        linewidth=2, markersize=6)
        
        ax2.set_xlabel('Token Position', fontsize=12)
        ax2.set_ylabel('Absolute Error', fontsize=12)
        ax2.set_title('Attention Kernel Error vs Token Position (All Precisions)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        if precision_data:
            # 获取所有token位置并合并
            all_positions = set()
            for data in precision_data.values():
                all_positions.update(data['attn_positions'])
            token_positions_attn = sorted(all_positions)
            if token_positions_attn:
                ax2.set_xticks(token_positions_attn)
                ax2.set_xticklabels([int(x) for x in token_positions_attn])
    
    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'error_analysis_summary.jpg')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {output_path}")

def generate_error_summary_json(all_results, output_dir):
    """生成JSON文件，包含不同精度的平均误差统计"""
    print("  Generating error summary JSON...")
    
    summary = {
        'precisions': {}
    }
    
    for prec_name, results in all_results.items():
        # 计算KV cache的平均误差
        all_kv_errors = []
        all_attn_errors = []
        
        for sentence_errors in results['kv_cache_errors']:
            for err_data in sentence_errors:
                all_kv_errors.append(err_data['abs_error'])
        
        for sentence_errors in results['attention_errors']:
            for err_data in sentence_errors:
                all_attn_errors.append(err_data['abs_error'])
        
        if all_kv_errors or all_attn_errors:
            summary['precisions'][prec_name] = {
                'kv_cache': {
                    'mean_error': float(np.mean(all_kv_errors)) if all_kv_errors else None,
                    'std_error': float(np.std(all_kv_errors)) if all_kv_errors else None,
                    'min_error': float(np.min(all_kv_errors)) if all_kv_errors else None,
                    'max_error': float(np.max(all_kv_errors)) if all_kv_errors else None,
                    'total_samples': len(all_kv_errors)
                },
                'attention': {
                    'mean_error': float(np.mean(all_attn_errors)) if all_attn_errors else None,
                    'std_error': float(np.std(all_attn_errors)) if all_attn_errors else None,
                    'min_error': float(np.min(all_attn_errors)) if all_attn_errors else None,
                    'max_error': float(np.max(all_attn_errors)) if all_attn_errors else None,
                    'total_samples': len(all_attn_errors)
                }
            }
    
    # 保存JSON文件
    json_path = os.path.join(output_dir, 'error_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"    Saved: {json_path}")

if __name__ == "__main__":
    run_experiment()

