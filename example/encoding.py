import json

def recover_json_encoding(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8', errors='replace') as infile:
        # 读取整个文件内容，使用 replace 处理解码错误
        data = infile.read()
    
    # 尝试将修复后的数据作为 JSON 进行加载
    try:
        json_data = json.loads(data)
        
        # 将修复后的 JSON 写入到新的文件
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(json_data, outfile, ensure_ascii=False, indent=4)
        
        print("JSON 文件已成功恢复并保存。")
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")

# 调用函数
recover_json_encoding('air_JSON.json', 'output.json')

