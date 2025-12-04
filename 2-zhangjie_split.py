import json
import concurrent.futures
import time
from openai import OpenAI
from openai import Timeout, APIError
import argparse

# 大模型配置（按用户指定）
OPENAI_API_KEY = "2d4b537d-f8b8-4536-955a-fb1b7b7a5101"
OPENAI_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "doubao-seed-1-6-lite-251015"

# 初始化OpenAI客户端
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    timeout=600  # 全局超时10分钟
)

# 语义割裂打分标准（固定规则，嵌入提示词）
SCORING_CRITERIA = """
语义割裂分数（0-10分）定义，分数越高表示割裂程度越高：
0-5分【无割裂】：同一情节/副本内直接推进，核心场景、目标、冲突完全连续（如A章在某地调查，B章继续在该地对峙）；
6-10分【重度割裂】：留下重大悬念或者翻转、重大情节收尾、核心目标变更、境界突破或时间跳跃（如A章结束黑狼帮主线，B章开启修炼新篇章；或主角突破新境界）。
"""


def load_chapters(jsonl_path):
    """读取JSONL文件，按chapter_index排序章节"""
    chapters = []
    with open(jsonl_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            line= json.loads(line)
            chapters.append({
                'index': line.get('chapter_index'),
                'title': line.get('title'),
                'summary': line.get('summary')
            })
    # 按章节索引升序排序（确保相邻章节连续）
    chapters.sort(key=lambda x: x['index'])
    return chapters


def build_prompt(prev_chap, curr_chap):
    """构建判断语义割裂的提示词"""
    return f"""
任务：分析以下两章小说概要的语义割裂程度，严格按照给定标准打分，不用说明理由。
{SCORING_CRITERIA}
前一章（索引{prev_chap['index']}，标题：{prev_chap['title']}）：
{prev_chap['summary']}

后一章（索引{curr_chap['index']}，标题：{curr_chap['title']}）：
{curr_chap['summary']}

要求输出格式（严格遵守，不要额外内容，分数必须是0-10的整数）：
分数：X分
"""


def judge_split_single(prev_chap, curr_chap):
    """单对章节语义割裂判断（供并发调用）"""
    prompt = build_prompt(prev_chap, curr_chap)
    result = {
        "prev_chapter_index": prev_chap['index'],
        "curr_chapter_index": curr_chap['index'],
        "split_score": None,
        "score_definition": "未生成有效结果",
        "judge_reason": "未生成理由",
        "status": "success"
    }
    
    try:
        # 调用大模型（按用户指定格式）
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=12288,
            temperature=0.3,
            top_p=0.9,
            timeout=600,
        )
        content = response.choices[0].message.content.strip()
        
        # 解析响应结果
        if "分数：" in content:
            score_str = content.split("分数：")[-1].split("分")[0].strip()
            result["split_score"] = int(score_str) if score_str.isdigit() and 0 <= int(score_str) <= 10 else None
        
    
    except Timeout:
        result["status"] = "timeout"
        result["judge_reason"] = "调用超时，未完成判断"
    except APIError as e:
        result["status"] = "api_error"
        result["judge_reason"] = f"API错误：{str(e)}"
    except Exception as e:
        result["status"] = "error"
        result["judge_reason"] = f"未知错误：{str(e)}"
    
    return result


def main():
    parser = argparse.ArgumentParser(description='小说章节语义割裂判断（并发调用+OpenAI兼容接口）')
    parser.add_argument('--input_jsonl', default='chapter_summaries.jsonl', help='输入的JSONL文件路径')
    parser.add_argument('--output_jsonl', default='split_judge_result.jsonl', help='输出结果的JSONL文件路径')
    parser.add_argument('--max_workers', type=int, default=10, help='并发线程数（默认10，根据接口承载调整）')
    args = parser.parse_args()
    
    # 加载章节
    print("正在加载章节概要...")
    chapters = load_chapters(args.input_jsonl)
    total_chaps = len(chapters)
    print(f"成功加载 {total_chaps} 个章节（已按索引排序）")
    
    # 生成待处理的相邻章节对
    tasks = []
    for i in range(total_chaps - 1):
        tasks.append((chapters[i], chapters[i + 1]))
    print(f"共需处理 {len(tasks)} 对相邻章节")
    
    # 并发执行判断任务
    print(f"启动并发处理（线程数：{args.max_workers}）...")
    start_time = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务并获取结果
        future_to_task = {executor.submit(judge_split_single, prev, curr): (prev['index'], curr['index']) for prev, curr
                          in tasks}
        
        for future in concurrent.futures.as_completed(future_to_task):
            chap_pair = future_to_task[future]
            try:
                res = future.result()
                results.append(res)
                print(f"完成：第{chap_pair[0]}章→第{chap_pair[1]}章（状态：{res['status']}）")
            except Exception as e:
                print(f"失败：第{chap_pair[0]}章→第{chap_pair[1]}章，错误：{str(e)}")
    
    # 按章节对顺序排序结果（并发执行顺序不固定，需还原）
    results.sort(key=lambda x: (x['prev_chapter_index'], x['curr_chapter_index']))
    
    # 保存结果
    with open(args.output_jsonl, 'w', encoding='utf-8') as writer:
        for i in results:
            writer.write(json.dumps(i, ensure_ascii=False) + '\n')
        # writer.write_all(results)
    
    # 统计信息
    end_time = time.time()
    success_count = sum(1 for r in results if r['status'] == 'success' and r['split_score'] is not None)
    fail_count = len(results) - success_count
    scores = [r['split_score'] for r in results if r['split_score'] is not None]
    
    print(f"\n===== 处理完成 =====")
    print(f"总耗时：{(end_time - start_time):.1f} 秒")
    print(f"成功判断：{success_count} 对")
    print(f"失败/超时：{fail_count} 对")
    if scores:
        print(f"割裂分数范围：{min(scores)} - {max(scores)}")
        print(f"平均割裂分数：{sum(scores) / len(scores):.1f}")
    print(f"结果已保存至：{args.output_jsonl}")


if __name__ == "__main__":
    main()