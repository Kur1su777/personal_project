#这个是把搜集的数据集转化成符合的要求的格式
import json
import os
from collections import defaultdict

def convert_coco_to_yolo_optimized(json_file_path, output_dir, yolo_class_names_list):
    

    with open(json_file_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 1. 构建类别映射 (category_id -> class_index)
    category_id_to_index = {}
    for category in coco_data.get('categories', []): # 使用 .get 避免 categories 不存在
        class_id = category['id']
        class_name = category['name']
        if class_name in yolo_class_names_list:
            category_id_to_index[class_id] = yolo_class_names_list.index(class_name)
        else:
            print(f"警告：类别 '{class_name}' (ID: {class_id}) 在 yolo_class_names_list 中未找到。将被忽略。")

    # 2. 组织标注信息，按 image_id 分组
    annotations_by_image = defaultdict(list)
    for ann in coco_data.get('annotations', []): # 使用 .get 避免 annotations 不存在
        annotations_by_image[ann['image_id']].append(ann)

    # 3. 遍历图像信息，并为每个图像生成 TXT 文件
    for image_info in coco_data.get('images', []): # 使用 .get 避免 images 不存在
        image_id = image_info['id']
        file_name = image_info['file_name']
        img_width = image_info.get('width') # 使用 .get 确保 width/height 存在
        img_height = image_info.get('height')

        if not img_width or not img_height:
            print(f"警告：图像 {file_name} (Image ID: {image_id}) 在 JSON 中缺少 'width' 或 'height' 信息。跳过此图像。")
            continue # 如果 JSON 中没有宽高信息，则跳过

        txt_filename = os.path.splitext(file_name)[0] + '.txt'
        txt_file_path = os.path.join(output_dir, txt_filename)

        # 检查该图像是否有标注
        if image_id not in annotations_by_image:
            
            continue

        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            for ann in annotations_by_image[image_id]:
                category_id = ann['category_id']

                if category_id not in category_id_to_index:
                    
                    continue # 忽略不在您指定类别列表中的标注

                class_index = category_id_to_index[category_id] # 这是 YOLO 所需的 0-based index

                x_min, y_min, box_width, box_height = ann['bbox']

                # 计算 YOLO 格式所需的中心点坐标
                x_center = x_min + box_width / 2
                y_center = y_min + box_height / 2

                # 归一化坐标和宽高
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = box_width / img_width
                height_norm = box_height / img_height

                # 写入 YOLO TXT 文件
                txt_file.write(f"{class_index} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

# --- 配置部分 ---
if __name__ == "__main__":
    # 1. 输入 JSON 文件路径
    coco_json_file = r'D:\datasets\judge_head\annotations\instances_train2017.json' 

    # 2. 输出 YOLO TXT 文件的根目录

    output_base_dir = r'D:\datasets\judge_head\labels' 


    yolo_class_names_list = ['raise head', 'lower head'] 

    output_dir_for_this_json = os.path.join(output_base_dir, 'train2017') 
    os.makedirs(output_dir_for_this_json, exist_ok=True)

    print(f"开始转换 JSON 文件: {coco_json_file}")
    print(f"输出到目录: {output_dir_for_this_json}")
    print(f"使用的类别列表 (YOLOv5 dataset.yaml names): {yolo_class_names_list}")

    # 打印出实际的类别映射，方便检查
    print("实际的类别映射 (JSON Category ID -> YOLO Class Index):")
    try:
        with open(coco_json_file, 'r', encoding='utf-8') as f:
            temp_coco_data = json.load(f)
            categories = temp_coco_data.get('categories', [])
            if not categories:
                print("  - 警告：JSON 文件中未找到 'categories' 部分，无法建立类别映射。")
            else:
                for category in categories:
                    class_id = category['id']
                    class_name = category['name']
                    if class_name in yolo_class_names_list:
                        yolo_index = yolo_class_names_list.index(class_name)
                        print(f"  - JSON Category ID: {class_id} (Name: '{class_name}') -> YOLO Class Index: {yolo_index}")
                    
    except Exception as e:
        print(f"  - 尝试读取类别信息时发生错误: {e}")


    convert_coco_to_yolo_optimized(coco_json_file, output_dir_for_this_json, yolo_class_names_list)
    print("转换完成！")
