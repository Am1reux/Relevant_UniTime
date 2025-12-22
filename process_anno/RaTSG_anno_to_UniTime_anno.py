'''

根据process_anno/annos/charades/charades_sta_test2.0.json 和 process_anno/annos/charades/charades_sta_val2.0.json
处理/home/fwj/workspace/code/UniTime/UniTime_data/charades/test.json

2.0是我构建了负样本的版本
将test.json charades_sta_test2.0.json 和 charades_sta_val2.0.json进行分割
现在test.json里面包含test和val的内容

charades_sta_val2.0.json版本格式如下
{
    "4KK20": {
        "sts": [
            {
                "sentence": "person talking on their phone.",
                "id": "4KK20-0000",
                "timestamp": [
                    1.8,
                    13.7
                ],
                "no_answer": false
            },
            {
                "sentence": "person throwing a box down.",
                "id": "4KK20-0001",
                "timestamp": [
                    24.4,
                    32.0
                ],
                "no_answer": false
            },
            {
                "sentence": "person closes the cabinet door.",
                "id": "UR5TU-1000",
                "timestamp": [],
                "no_answer": true
            },
            {
                "sentence": "person opens a book.",
                "id": "AVL8A-1000",
                "timestamp": [],
                "no_answer": true
            }
        ],
        "duration": 30.67
    },

例如 person closes the cabinet door.就是其他视频中过滤筛选出的高质量文本, 与4KK20构建成负样本

将test.json 分割成 test2.0.json 和 val2.0.json 按照charades_sta_test2.0.json 和 charades_sta_val2.0.json中的视频id进行划分
以4KK20为例,
输出的val2.0.json为
[
  {
    "qid": 1331,
    "id": "4KK20",
    "annos": [
      {
        "query": "person talking on their phone.",
        "window": [
          [
            1.8,
            13.7
          ]
        ],
        "relevant": true
      }
    ],
    "duration": 30.67,
    "mode": "mr",
    "video_path": "/home/fwj/workspace/VisualSearch/charades/Charades_v1/4KK20.mp4"
  },
  {
    "qid": 1332,
    "id": "4KK20",
    "annos": [
      {
        "query": "person throwing a box down.",
        "window": [
          [
            24.4,
            32.0
          ]
        ],
        "relevant": true
      }
    ],
    "duration": 30.67,
    "mode": "mr",
    "video_path": "/home/fwj/workspace/VisualSearch/charades/Charades_v1/4KK20.mp4"
  },
  {
    "qid": 1333,
    "id": "4KK20",
    "annos": [
      {
        "query": "person close the cabinet door.",
        "window": [
          [
          ]
        ],
        "relevant": false
      }
    ],
    "duration": 30.67,
    "mode": "mr",
    "video_path": "/home/fwj/workspace/VisualSearch/charades/Charades_v1/4KK20.mp4"
  },
  {
    "qid": 1334,
    "id": "4KK20",
    "annos": [
      {
        "query": "person opens a book.",
        "window": [
          [
          ]
        ],
        "relevant": false
      }
    ],
    "duration": 30.67,
    "mode": "mr",
    "video_path": "/home/fwj/workspace/VisualSearch/charades/Charades_v1/4KK20.mp4"
  },
  
]

最终生成的test2.0.json 和 val2.0.json格式如上所示
添加relevant字段, 为true代表正样本, false代表负样本
qid可以不用按照原先的qid顺序进行编号, 只要保证唯一即可
'''



import json
import os

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """保存JSON文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_video_id(video_path):
    """从视频路径中提取视频ID"""
    return os.path.basename(video_path).replace('.mp4', '')

def convert_ratsg_to_unitime(ratsg_anno_path, test_json_path, output_path):
    """
    将RaTSG格式的注释转换为UniTime格式
    
    Args:
        ratsg_anno_path: RaTSG注释文件路径(charades_sta_test2.0.json或val2.0.json)
        test_json_path: 原始test.json文件路径
        output_path: 输出文件路径
    """
    # 加载数据
    ratsg_data = load_json(ratsg_anno_path)
    test_data = load_json(test_json_path)
    
    # 获取RaTSG中的所有视频ID
    ratsg_video_ids = set(ratsg_data.keys())
    
    # 创建视频ID到test.json条目的映射
    video_id_to_entries = {}
    for entry in test_data:
        video_id = extract_video_id(entry['video_path'])
        if video_id not in video_id_to_entries:
            video_id_to_entries[video_id] = []
        video_id_to_entries[video_id].append(entry)
    
    # 生成输出数据
    output_data = []
    qid_counter = 1
    
    # 遍历RaTSG中的每个视频
    for video_id in sorted(ratsg_video_ids):
        if video_id not in ratsg_data:
            continue
            
        video_info = ratsg_data[video_id]
        duration = video_info.get('duration', 0)
        video_path = f"/home/fwj/workspace/VisualSearch/charades/Charades_v1/{video_id}.mp4"
        
        # 遍历该视频的所有句子
        for st in video_info.get('sts', []):
            query = st.get('sentence', '')
            timestamp = st.get('timestamp', [])
            no_answer = st.get('no_answer', False)
            
            # 创建UniTime格式的条目
            entry = {
                "qid": qid_counter,
                "id": video_id,
                "annos": [
                    {
                        "query": query,
                        "window": [timestamp] if timestamp else [[]],
                        "relevant": not no_answer  # no_answer为False表示正样本,relevant为True
                    }
                ],
                "duration": duration,
                "mode": "mr",
                "video_path": video_path
            }
            
            output_data.append(entry)
            qid_counter += 1
    
    # 保存结果
    save_json(output_data, output_path)
    print(f"转换完成! 共生成 {len(output_data)} 条数据")
    print(f"输出文件: {output_path}")

def main():
    # 定义路径
    base_dir = "/home/fwj/workspace/code/UniTime"
    anno_dir = os.path.join(base_dir, "process_anno/annos/charades")
    data_dir = os.path.join(base_dir, "UniTime_data/charades")
    
    # 输入文件
    test_ratsg_path = os.path.join(anno_dir, "charades_sta_test2.0.json")
    val_ratsg_path = os.path.join(anno_dir, "charades_sta_val2.0.json")
    test_json_path = os.path.join(data_dir, "test.json")
    
    # 输出文件
    output_test_path = os.path.join(data_dir, "test2.0.json")
    output_val_path = os.path.join(data_dir, "val2.0.json")
    
    # 转换test集
    print("正在转换test集...")
    convert_ratsg_to_unitime(test_ratsg_path, test_json_path, output_test_path)
    
    # 转换val集
    print("\n正在转换val集...")
    convert_ratsg_to_unitime(val_ratsg_path, test_json_path, output_val_path)

if __name__ == "__main__":
    main()