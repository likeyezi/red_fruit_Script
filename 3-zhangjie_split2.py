import jsonlines
import numpy as np
import argparse
from collections import defaultdict


def load_chapters(jsonl_path):
    """加载章节数据（需包含chapter_index，确保有序）"""
    chapters = []
    with jsonlines.open(jsonl_path, 'r') as reader:
        for line in reader:
            chapters.append({
                'index': line.get('chapter_index'),
                'title': line.get('title'),
                'summary': line.get('summary')
            })
    # 按章节索引升序排序，确保连续性
    chapters.sort(key=lambda x: x['index'])
    return chapters


def load_split_scores(jsonl_path):
    """加载语义割裂评分，建立{前章节索引: 割裂分数}映射"""
    split_score_map = {}  # key: prev_chapter_index, value: split_score（整数）
    score_details = defaultdict(dict)  # 存储完整评分信息（字典）
    with jsonlines.open(jsonl_path, 'r') as reader:
        for line in reader:
            prev_idx = line.get('prev_chapter_index')
            curr_idx = line.get('curr_chapter_index')
            score = line.get('split_score')
            if prev_idx is not None and score is not None and 0 <= score <= 10:
                split_score_map[prev_idx] = score
                score_details[prev_idx] = {
                    'curr_chapter_index': curr_idx,
                    'score_definition': line.get('score_definition'),
                    'judge_reason': line.get('judge_reason'),
                    'status': line.get('status')
                }
    return split_score_map, score_details


def split_into_volumes(chapters, split_score_map, score_details, num_volumes):
    """按高割裂分数+长度约束切分成指定卷数"""
    total_chapters = len(chapters)
    if total_chapters <= num_volumes:
        raise ValueError(f"章节总数({total_chapters})少于目标卷数({num_volumes})，无法切分")
    
    # 1. 计算长度约束范围（平均±20%，确保不过长过短）
    avg_per_volume = total_chapters / num_volumes
    min_per_volume = max(1, int(avg_per_volume * 0.8))  # 最小章节数（≥1）
    max_per_volume = int(avg_per_volume * 1.2)  # 最大章节数
    print(f"长度约束：每卷{min_per_volume}-{max_per_volume}章（平均{avg_per_volume:.1f}章）")
    
    # 2. 整理所有可能的切分点（相邻章节对的前章节索引，带分数）
    chapter_indices = [chap['index'] for chap in chapters]
    chapter_idx_to_pos = {idx: i for i, idx in enumerate(chapter_indices)}  # 索引→列表位置映射
    
    possible_splits = []  # 元素：(prev_chapter_index, split_score, position)
    for i, prev_idx in enumerate(chapter_indices[:-1]):  # 最后一章无后续章节，不构成切分点
        score = split_score_map.get(prev_idx, 0)  # 无评分默认0分（不优先选）
        possible_splits.append((prev_idx, score, i))
    
    # 3. 按割裂分数降序排序（优先选高分切分点）
    possible_splits.sort(key=lambda x: (-x[1], x[0]))  # 分数降序，索引升序（同分按位置）
    
    # 4. 筛选切分点（满足长度约束，最终选num_volumes-1个）
    selected_splits = []
    current_start_pos = 0  # 当前卷起始位置（使用位置索引而非章节索引）
    remaining_volumes = num_volumes
    
    # 计算剩余章节需要满足的长度约束
    def get_remaining_constraints(start_pos, remaining_vols):
        remaining_chapters = total_chapters - start_pos
        if remaining_vols <= 0:
            return 0, 0
        avg_remaining = remaining_chapters / remaining_vols
        min_remaining = max(1, int(avg_remaining * 0.8))
        max_remaining = min(remaining_chapters, int(avg_remaining * 1.2))
        return min_remaining, max_remaining
    
    while len(selected_splits) < num_volumes - 1:
        min_len, max_len = get_remaining_constraints(current_start_pos, remaining_volumes)
        if min_len == 0 or max_len == 0:
            break
        
        # 找到当前可选择的切分点（满足长度约束）
        valid_candidates = []
        for (prev_idx, score, pos) in possible_splits:
            # 切分点必须在当前卷范围内
            if pos < current_start_pos:
                continue
            
            # 计算当前卷长度
            volume_length = pos - current_start_pos + 1
            
            # 检查长度是否在约束范围内，且切分点未被选中
            if (min_len <= volume_length <= max_len) and (prev_idx not in selected_splits):
                # 检查剩余章节是否能满足剩余卷数的约束
                next_min, next_max = get_remaining_constraints(pos + 1, remaining_volumes - 1)
                remaining_chapters = total_chapters - (pos + 1)
                if next_min > 0 and remaining_chapters >= next_min * (remaining_volumes - 1):
                    valid_candidates.append((prev_idx, score, volume_length, pos))
        
        if not valid_candidates:
            # 放宽约束：允许稍微超出最大长度
            relaxed_max = int(max_len * 1.3)
            for (prev_idx, score, pos) in possible_splits:
                if pos < current_start_pos:
                    continue
                volume_length = pos - current_start_pos + 1
                if (min_len <= volume_length <= relaxed_max) and (prev_idx not in selected_splits):
                    next_min, next_max = get_remaining_constraints(pos + 1, remaining_volumes - 1)
                    remaining_chapters = total_chapters - (pos + 1)
                    if next_min > 0 and remaining_chapters >= next_min * (remaining_volumes - 1):
                        valid_candidates.append((prev_idx, score, volume_length, pos))
        
        if not valid_candidates:
            # 如果还是找不到，选择最接近平均长度的点
            target_length = avg_per_volume
            closest_candidate = None
            min_diff = float('inf')
            
            for (prev_idx, score, pos) in possible_splits:
                if pos < current_start_pos:
                    continue
                volume_length = pos - current_start_pos + 1
                if volume_length >= min_len and (prev_idx not in selected_splits):
                    diff = abs(volume_length - target_length)
                    if diff < min_diff:
                        min_diff = diff
                        closest_candidate = (prev_idx, score, volume_length, pos)
            
            if closest_candidate:
                valid_candidates.append(closest_candidate)
        
        if not valid_candidates:
            # 最后的手段：选择当前范围内的最高分切分点
            for (prev_idx, score, pos) in possible_splits:
                if pos < current_start_pos:
                    continue
                volume_length = pos - current_start_pos + 1
                if volume_length >= 1 and (prev_idx not in selected_splits):
                    valid_candidates.append((prev_idx, score, volume_length, pos))
        
        if not valid_candidates:
            print(f"警告：无法找到合适的切分点，当前起始位置{current_start_pos}，剩余卷数{remaining_volumes}")
            break
        
        # 选分数最高的候选点（同分选长度最接近平均值的）
        valid_candidates.sort(key=lambda x: (-x[1], abs(x[2] - avg_per_volume)))
        best_split, best_score, best_length, best_pos = valid_candidates[0]
        selected_splits.append(best_split)
        
        # 更新当前位置和剩余卷数
        current_start_pos = best_pos + 1
        remaining_volumes -= 1
        print(f"选择切分点：第{best_split}章后（分数{best_score}，当前卷{best_length}章）")
    
    # 5. 生成指定卷数
    volumes = []
    start_pos = 0
    selected_splits_positions = []
    
    # 获取所有选中的切分点的位置
    for split_prev_idx in selected_splits:
        split_pos = chapter_idx_to_pos[split_prev_idx] + 1
        selected_splits_positions.append(split_pos)
    
    # 按位置排序切分点
    selected_splits_positions.sort()
    
    # 根据切分点生成卷
    for split_pos in selected_splits_positions:
        volumes.append(chapters[start_pos:split_pos])
        start_pos = split_pos
    
    # 添加最后一卷
    if start_pos < total_chapters:
        volumes.append(chapters[start_pos:])
    
    # 6. 最终校验和调整
    final_volumes = []
    final_splits = []
    
    # 确保卷数正确且长度合理
    if len(volumes) > num_volumes:
        # 合并过短的卷
        volumes = merge_short_volumes(volumes, min_per_volume, split_score_map, chapter_idx_to_pos)
    elif len(volumes) < num_volumes:
        # 拆分过长的卷
        volumes = split_long_volumes(volumes, max_per_volume, split_score_map, chapter_idx_to_pos, num_volumes)
    
    # 重新生成切分点列表
    start_pos = 0
    for i in range(len(volumes) - 1):
        end_pos = start_pos + len(volumes[i]) - 1
        split_prev_idx = chapters[end_pos]['index']
        final_splits.append(split_prev_idx)
        start_pos = end_pos + 1
    
    return volumes[:num_volumes], final_splits


def merge_short_volumes(volumes, min_per_volume, split_score_map, chapter_idx_to_pos):
    """合并过短的卷"""
    merged = []
    i = 0
    while i < len(volumes):
        if len(volumes[i]) < min_per_volume and i < len(volumes) - 1:
            # 合并到下一卷
            merged_vol = volumes[i] + volumes[i + 1]
            merged.append(merged_vol)
            i += 2
        else:
            merged.append(volumes[i])
            i += 1
    return merged


def split_long_volumes(volumes, max_per_volume, split_score_map, chapter_idx_to_pos, target_volumes):
    """拆分过长的卷"""
    result = volumes.copy()
    
    while len(result) < target_volumes:
        # 找到最长的卷
        longest_idx = max(range(len(result)), key=lambda i: len(result[i]))
        longest_vol = result[longest_idx]
        
        if len(longest_vol) <= max_per_volume:
            break
        
        # 在长卷中找最高分切分点
        best_split_pos = -1
        best_score = -1
        
        for j in range(len(longest_vol) - 1):
            prev_idx = longest_vol[j]['index']
            score = split_score_map.get(prev_idx, 0)
            if score > best_score:
                best_score = score
                best_split_pos = j + 1
        
        if best_split_pos > 0:
            # 拆分长卷
            vol1 = longest_vol[:best_split_pos]
            vol2 = longest_vol[best_split_pos:]
            result[longest_idx:longest_idx + 1] = [vol1, vol2]
        else:
            # 没有找到合适的切分点，在中间位置拆分
            mid_pos = len(longest_vol) // 2
            vol1 = longest_vol[:mid_pos]
            vol2 = longest_vol[mid_pos:]
            result[longest_idx:longest_idx + 1] = [vol1, vol2]
    
    return result


def save_volumes_jsonl(volumes, split_splits, score_details, split_score_map, output_jsonl):
    """保存切分结果为JSONL文件，包含每卷的开始、结束及切分点信息"""
    with jsonlines.open(output_jsonl, 'w') as writer:
        for i, vol in enumerate(volumes, 1):
            # 基础卷信息
            volume_info = {
                "volume_index": i,
                "start_chapter": vol[0]['index'],
                "end_chapter": vol[-1]['index'],
                "chapter_count": len(vol),
                "start_title": vol[0]['title'],
                "end_title": vol[-1]['title']
            }
            
            # 如果不是最后一卷，添加切分点信息
            if i < len(volumes):
                split_prev_idx = split_splits[i - 1] if i - 1 < len(split_splits) else vol[-1]['index']
                split_detail = score_details.get(split_prev_idx, {})
                split_info = {
                    "split_prev_chapter": split_prev_idx,
                    "split_next_chapter": split_detail.get('curr_chapter_index', split_prev_idx + 1),
                    "split_score": split_score_map.get(split_prev_idx, 0),
                    "score_definition": split_detail.get('score_definition', '无'),
                    "judge_reason": split_detail.get('judge_reason', '无')
                }
                volume_info.update(split_info)
            
            writer.write(volume_info)


def main():
    parser = argparse.ArgumentParser(description='基于语义割裂评分切分章节为指定卷数')
    parser.add_argument('--chapter_jsonl', default='chapter_summaries.jsonl', help='章节概要JSONL文件路径')
    parser.add_argument('--score_jsonl', default='split_judge_result.jsonl', help='语义割裂评分JSONL文件路径')
    parser.add_argument('--output_jsonl', default='volume_split_result.jsonl', help='切分结果JSONL文件路径')
    parser.add_argument('--num_volumes', type=int, default=120, help='目标卷数（需小于章节总数）')
    args = parser.parse_args()
    
    # 加载数据
    print("正在加载章节数据...")
    chapters = load_chapters(args.chapter_jsonl)
    print(f"成功加载 {len(chapters)} 个章节")
    
    print("正在加载语义割裂评分...")
    split_score_map, score_details = load_split_scores(args.score_jsonl)
    print(f"成功加载 {len(split_score_map)} 对章节的割裂评分")
    
    # 切分处理
    print(f"开始按高割裂分数+长度约束切分为{args.num_volumes}卷...")
    try:
        volumes, selected_splits = split_into_volumes(
            chapters,
            split_score_map,
            score_details,
            args.num_volumes
        )
    except ValueError as e:
        print(f"错误：{e}")
        return
    except RuntimeError as e:
        print(f"切分失败：{e}")
        return
    
    # 保存结果
    save_volumes_jsonl(volumes, selected_splits, score_details, split_score_map, args.output_jsonl)
    
    # 输出统计信息
    print(f"\n===== 切分完成 =====")
    print(f"目标卷数：{args.num_volumes}卷")
    print(f"实际卷数：{len(volumes)}卷")
    vol_lengths = [len(vol) for vol in volumes]
    print(f"每卷章节数范围：{min(vol_lengths)}-{max(vol_lengths)}章")
    print(f"平均每卷章节数：{np.mean(vol_lengths):.1f}章")
    print(f"切分结果已保存至：{args.output_jsonl}")
    
    # 打印前5卷和后5卷概要
    print("\n前5卷概要：")
    for i in range(min(5, len(volumes))):
        vol = volumes[i]
        print(f"第{i + 1}卷：第{vol[0]['index']}-{vol[-1]['index']}章（{len(vol)}章）")
    
    print("\n后5卷概要：")
    start_idx = max(0, len(volumes) - 5)
    for i in range(start_idx, len(volumes)):
        vol = volumes[i]
        print(f"第{i + 1}卷：第{vol[0]['index']}-{vol[-1]['index']}章（{len(vol)}章）")


if __name__ == "__main__":
    main()