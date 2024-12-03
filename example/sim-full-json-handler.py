import json
from collections import defaultdict


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

def clean_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 保留OBJECTS并清理掉没有entity属性的对象
    if 'OBJECTS' in data:
        data['OBJECTS'] = [obj for obj in data['OBJECTS'] if 'entity' in obj]

    # 清空其他字段
    data = {'OBJECTS': data.get('OBJECTS', [])}

    # 将处理后的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def filter_entities(input_file, output_file):
    # 允许的entity类型
    allowed_entities = [
        "CIRCLE", "TEXT", "LINE", "HATCH",
        "LWPOLYLINE", "MTEXT", "ARC", "SOLID", "SPLINE"
    ]

    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查是否存在'OBJECTS'字段
    if 'OBJECTS' in data:
        # 过滤OBJECTS数组，只保留指定的entity类型
        data['OBJECTS'] = [
            obj for obj in data['OBJECTS'] if obj.get('entity') in allowed_entities
        ]

    # 将过滤后的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def remove_unwanted_properties(input_file, output_file):
    # 不需要的属性列表
    unwanted_properties = [
        "handle", "type", "size", "bitsize", "_subclass", "layer", "preview_exists",
        "entmode", "is_xdic_missing", "has_ds_data", "ltype_scale", "ltype_flags",
        "plotstyle_flags", "material_flags", "shadow_flags", "has_full_visualstyle",
        "has_face_visualstyle", "has_edge_visualstyle", "invisible", "linewt",
        "extrusion", "thickness", "style"
    ]

    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查是否存在'OBJECTS'字段
    if 'OBJECTS' in data:
        for obj in data['OBJECTS']:
            # 删除不需要的属性
            for prop in unwanted_properties:
                if prop in obj:
                    del obj[prop]

    # 将处理后的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def categorize_by_entity(input_file, output_file):
    # 创建一个字典，键是entity类型，值是包含该类型对象的列表
    categorized_data = defaultdict(list)

    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查是否存在'OBJECTS'字段
    if 'OBJECTS' in data:
        for obj in data['OBJECTS']:
            # 获取entity类型并将对象添加到对应的类型列表中
            entity_type = obj.get('entity', 'Unknown')  # 如果没有entity字段，默认用'Unknown'
            categorized_data[entity_type].append(obj)

    # 创建最终输出结构
    final_data = {'TYPES': dict(categorized_data)}  # 转换为普通字典以便写入文件

    # 将分类后的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)


input_file = 'ocean-full-json.json'  # 输入JSON文件路径
output_file = 'ocean-sim-handled.json'  # 输出JSON文件路径


# 调用函数
recover_json_encoding(input_file, output_file)
clean_json(output_file, output_file)
filter_entities(output_file, output_file)
remove_unwanted_properties(output_file, output_file)
categorize_by_entity(output_file, output_file)