#!/usr/bin/env python3
"""
生成结果汇总表格
横坐标：硬件
纵坐标：精度
"""
import json
import os
import glob
from collections import defaultdict
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, using basic table generation")

def scan_results_directory(results_dir='results'):
    """扫描results目录，收集所有硬件和精度的数据"""
    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory '{results_dir}' not found!")
        return None
    
    # 收集所有数据
    all_data = defaultdict(lambda: defaultdict(dict))
    
    # 遍历所有硬件文件夹
    hardware_dirs = [d for d in os.listdir(results_dir) 
                    if os.path.isdir(os.path.join(results_dir, d))]
    
    print(f"Found {len(hardware_dirs)} hardware configurations:")
    
    for hardware_name in hardware_dirs:
        hardware_path = os.path.join(results_dir, hardware_name)
        error_summary_path = os.path.join(hardware_path, 'error_summary.json')
        
        if not os.path.exists(error_summary_path):
            print(f"  ⚠ {hardware_name}: error_summary.json not found")
            continue
        
        print(f"  ✓ {hardware_name}")
        
        # 读取error_summary.json
        try:
            with open(error_summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            # 提取每个精度的数据
            if 'precisions' in summary:
                for prec_name, prec_data in summary['precisions'].items():
                    # 提取KV cache和Attention的平均误差
                    kv_mean = prec_data.get('kv_cache', {}).get('mean_error')
                    attn_mean = prec_data.get('attention', {}).get('mean_error')
                    
                    # 存储数据（可以选择使用kv_cache或attention，或两者）
                    # 这里使用kv_cache的平均误差作为主要指标
                    if kv_mean is not None:
                        all_data[prec_name][hardware_name] = {
                            'kv_cache_mean': kv_mean,
                            'attention_mean': attn_mean,
                            'kv_cache_std': prec_data.get('kv_cache', {}).get('std_error'),
                            'attention_std': prec_data.get('attention', {}).get('std_error'),
                        }
        
        except Exception as e:
            print(f"  ✗ {hardware_name}: Error reading file - {e}")
            continue
    
    return all_data

def generate_table_basic(all_data, output_format='csv'):
    """生成表格（不使用pandas）"""
    if not all_data:
        print("ERROR: No data found!")
        return None
    
    # 收集所有精度和硬件
    all_precisions = sorted(all_data.keys())
    all_hardwares = set()
    for prec_data in all_data.values():
        all_hardwares.update(prec_data.keys())
    all_hardwares = sorted(all_hardwares)
    
    print(f"\nPrecisions found: {len(all_precisions)}")
    print(f"  {', '.join(all_precisions)}")
    print(f"\nHardwares found: {len(all_hardwares)}")
    print(f"  {', '.join(all_hardwares)}")
    
    # 生成CSV
    if output_format == 'csv':
        output_file = 'summary_table.csv'
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write('Precision,' + ','.join(all_hardwares) + '\n')
            # 写入数据
            for prec in all_precisions:
                row = [prec]
                for hw in all_hardwares:
                    if hw in all_data[prec]:
                        kv_mean = all_data[prec][hw]['kv_cache_mean']
                        if kv_mean is not None:
                            row.append(f"{kv_mean:.6e}")
                        else:
                            row.append("")
                    else:
                        row.append("")
                f.write(','.join(row) + '\n')
        print(f"\n✓ Generated CSV table: {output_file}")
    
    # 生成Markdown
    elif output_format == 'markdown':
        output_file = 'summary_table.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Error Summary Table\n\n")
            f.write("**KV Cache Mean Error** (compared to FP32 reference)\n\n")
            f.write("| Precision | " + " | ".join(all_hardwares) + " |\n")
            f.write("|" + "---|" * (len(all_hardwares) + 1) + "\n")
            for prec in all_precisions:
                row = [prec]
                for hw in all_hardwares:
                    if hw in all_data[prec]:
                        kv_mean = all_data[prec][hw]['kv_cache_mean']
                        if kv_mean is not None:
                            row.append(f"{kv_mean:.6e}")
                        else:
                            row.append("-")
                    else:
                        row.append("-")
                f.write("| " + " | ".join(row) + " |\n")
        print(f"\n✓ Generated Markdown table: {output_file}")
    
    # 生成HTML
    elif output_format == 'html':
        output_file = 'summary_table.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Error Summary Table</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
    </style>
</head>
<body>
    <h1>Error Summary Table</h1>
    <p><strong>KV Cache Mean Error</strong> (compared to FP32 reference)</p>
    <table>
        <thead>
            <tr>
                <th>Precision</th>
""")
            for hw in all_hardwares:
                f.write(f"                <th>{hw}</th>\n")
            f.write("""            </tr>
        </thead>
        <tbody>
""")
            for prec in all_precisions:
                f.write("            <tr>\n")
                f.write(f"                <td>{prec}</td>\n")
                for hw in all_hardwares:
                    if hw in all_data[prec]:
                        kv_mean = all_data[prec][hw]['kv_cache_mean']
                        if kv_mean is not None:
                            f.write(f"                <td>{kv_mean:.6e}</td>\n")
                        else:
                            f.write("                <td>-</td>\n")
                    else:
                        f.write("                <td>-</td>\n")
                f.write("            </tr>\n")
            f.write("""        </tbody>
    </table>
</body>
</html>
""")
        print(f"\n✓ Generated HTML table: {output_file}")
    
    # 打印表格到控制台
    print("\n" + "="*80)
    print("Summary Table (KV Cache Mean Error):")
    print("="*80)
    
    # 计算列宽
    col_widths = [max(len("Precision"), max([len(p) for p in all_precisions]))]
    for hw in all_hardwares:
        max_width = len(hw)
        for prec in all_precisions:
            if hw in all_data[prec]:
                kv_mean = all_data[prec][hw]['kv_cache_mean']
                if kv_mean is not None:
                    val_str = f"{kv_mean:.6e}"
                    max_width = max(max_width, len(val_str))
        col_widths.append(max_width)
    
    # 打印表头
    header = "Precision".ljust(col_widths[0])
    for i, hw in enumerate(all_hardwares):
        header += " | " + hw.ljust(col_widths[i+1])
    print(header)
    print("-" * len(header))
    
    # 打印数据行
    for prec in all_precisions:
        row = prec.ljust(col_widths[0])
        for i, hw in enumerate(all_hardwares):
            if hw in all_data[prec]:
                kv_mean = all_data[prec][hw]['kv_cache_mean']
                if kv_mean is not None:
                    val_str = f"{kv_mean:.6e}".ljust(col_widths[i+1])
                else:
                    val_str = "-".ljust(col_widths[i+1])
            else:
                val_str = "-".ljust(col_widths[i+1])
            row += " | " + val_str
        print(row)
    
    print("="*80)
    
    return all_data

def generate_table(all_data, output_format='csv'):
    """生成表格（使用pandas如果可用，否则使用基础方法）"""
    if HAS_PANDAS:
        return generate_table_pandas(all_data, output_format)
    else:
        return generate_table_basic(all_data, output_format)

def generate_table_pandas(all_data, output_format='csv'):
    """生成表格（使用pandas）"""
    if not all_data:
        print("ERROR: No data found!")
        return None
    
    # 收集所有精度和硬件
    all_precisions = sorted(all_data.keys())
    all_hardwares = set()
    for prec_data in all_data.values():
        all_hardwares.update(prec_data.keys())
    all_hardwares = sorted(all_hardwares)
    
    print(f"\nPrecisions found: {len(all_precisions)}")
    print(f"  {', '.join(all_precisions)}")
    print(f"\nHardwares found: {len(all_hardwares)}")
    print(f"  {', '.join(all_hardwares)}")
    
    # 创建表格数据
    table_data = []
    for prec in all_precisions:
        row = {'Precision': prec}
        for hw in all_hardwares:
            if hw in all_data[prec]:
                kv_mean = all_data[prec][hw]['kv_cache_mean']
                if kv_mean is not None:
                    row[hw] = f"{kv_mean:.6e}"
                else:
                    row[hw] = ""
            else:
                row[hw] = ""
        table_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(table_data)
    df = df.set_index('Precision')
    
    # 生成输出
    if output_format == 'csv':
        output_file = 'summary_table.csv'
        df.to_csv(output_file)
        print(f"\n✓ Generated CSV table: {output_file}")
    
    elif output_format == 'markdown':
        output_file = 'summary_table.md'
        try:
            from tabulate import tabulate
            markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=True)
        except ImportError:
            markdown_table = df.to_markdown()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Error Summary Table\n\n")
            f.write("**KV Cache Mean Error** (compared to FP32 reference)\n\n")
            f.write(markdown_table)
            f.write("\n\n")
        print(f"\n✓ Generated Markdown table: {output_file}")
    
    elif output_format == 'html':
        output_file = 'summary_table.html'
        html_content = df.to_html(classes='table table-striped', table_id='summary-table')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Error Summary Table</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .table th { background-color: #4CAF50; color: white; }
        .table tr:nth-child(even) { background-color: #f2f2f2; }
        .table tr:hover { background-color: #f5f5f5; }
    </style>
</head>
<body>
    <h1>Error Summary Table</h1>
    <p><strong>KV Cache Mean Error</strong> (compared to FP32 reference)</p>
""")
            f.write(html_content)
            f.write("""
</body>
</html>
""")
        print(f"\n✓ Generated HTML table: {output_file}")
    
    # 打印表格到控制台
    print("\n" + "="*80)
    print("Summary Table:")
    print("="*80)
    print(df.to_string())
    print("="*80)
    
    return df

def generate_detailed_table(all_data):
    """生成详细表格，包含KV Cache和Attention两个指标"""
    if not all_data:
        return
    
    all_precisions = sorted(all_data.keys())
    all_hardwares = set()
    for prec_data in all_data.values():
        all_hardwares.update(prec_data.keys())
    all_hardwares = sorted(all_hardwares)
    
    # 生成两个CSV文件
    for metric in ['KV Cache', 'Attention']:
        output_file = f'summary_table_{metric.lower().replace(" ", "_")}.csv'
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write('Precision,' + ','.join(all_hardwares) + '\n')
            # 写入数据
            for prec in all_precisions:
                row = [prec]
                for hw in all_hardwares:
                    if hw in all_data[prec]:
                        if metric == 'KV Cache':
                            mean_val = all_data[prec][hw]['kv_cache_mean']
                        else:
                            mean_val = all_data[prec][hw]['attention_mean']
                        
                        if mean_val is not None:
                            row.append(f"{mean_val:.6e}")
                        else:
                            row.append("")
                    else:
                        row.append("")
                f.write(','.join(row) + '\n')
        print(f"✓ Generated {metric} table: {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate summary table from experiment results')
    parser.add_argument('--results-dir', default='results', 
                       help='Directory containing results (default: results)')
    parser.add_argument('--format', choices=['csv', 'markdown', 'html', 'all'], 
                       default='all', help='Output format (default: all)')
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed tables for KV Cache and Attention separately')
    
    args = parser.parse_args()
    
    print("Scanning results directory...")
    all_data = scan_results_directory(args.results_dir)
    
    if not all_data:
        print("No data found. Exiting.")
        return
    
    # 生成表格
    if args.format == 'all':
        for fmt in ['csv', 'markdown', 'html']:
            generate_table(all_data, fmt)
    else:
        generate_table(all_data, args.format)
    
    # 生成详细表格
    if args.detailed:
        print("\nGenerating detailed tables...")
        generate_detailed_table(all_data)

if __name__ == "__main__":
    main()

