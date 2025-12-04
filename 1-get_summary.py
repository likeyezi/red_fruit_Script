import re
import json
import asyncio
import argparse
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# === 默认配置（可通过命令行覆盖）===
DEFAULT_OUTPUT_JSONL = "chapter_summaries.jsonl"
OPENAI_API_KEY = "2d4b537d-f8b8-4536-955a-fb1b7b7a5101"
OPENAI_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_NAME = "doubao-1-5-pro-32k-250115"
PROMPT_TEMPLATE = "请为以下小说章节内容生成一段300字的简短摘要：\n\n%s"

# 并发控制（根据服务器负载调整）
MAX_CONCURRENT = 5

# === 初始化异步 OpenAI 客户端 ===
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)


# === 切分章节函数 ===
def split_chapters(text):
    # 匹配“第...章”作为分隔符（支持中文数字、阿拉伯数字、带空格/标点等情况）
    pattern = r'(第\s*[零一二三四五六七八九十百千\d]+\s*章\s*.*)'
    parts = re.split(pattern, text)
    
    chapters = []
    current_title = None
    current_content = ""
    
    for part in parts:
        if re.match(r'第\s*[零一二三四五六七八九十百千\d]+\s*章', part):
            # 保存上一章
            if current_title is not None:
                chapters.append((current_title.strip(), current_content.strip()))
            current_title = part
            current_content = ""
        else:
            if current_title is not None:
                current_content += part
            else:
                # 文件开头可能有前言等内容，暂时忽略（不加入任何章节）
                pass
    
    # 添加最后一章
    if current_title is not None:
        chapters.append((current_title.strip(), current_content.strip()))
    
    return chapters


# === 异步调用大模型 ===
async def get_summary(idx, title, content):
    prompt = PROMPT_TEMPLATE % content[:12000]  # 防止过长，可按需调整截断长度
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0,
            top_p=0.0001,
            timeout=200,
        )
        summary = response.choices[0].message.content.strip()
        if '</think>\n\n' in summary:
            summary = summary.split('</think>\n\n')[-1]
        return idx, {"chapter_index": idx, "title": title, "summary": summary, 'content': content}
    except Exception as e:
        print(f"Error at chapter {idx} ({title}): {e}")
        return idx, {"chapter_index": idx, "title": title, "summary": "[ERROR]", 'content': content}


# === 主函数 ===
async def main(input_file, output_file):
    # 读取小说
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 切分章节
    chapters = split_chapters(text)
    print(f"共切分出 {len(chapters)} 章")
    
    # 创建信号量控制并发
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    async def bounded_get_summary(idx, title, content):
        async with semaphore:
            return await get_summary(idx, title, content)
    
    # 并发请求
    tasks = [
        bounded_get_summary(i, title, content)
        for i, (title, content) in enumerate(chapters)
    ]
    results = await tqdm_asyncio.gather(*tasks)
    
    # 按原始顺序排序（其实已经是顺序，但保险起见）
    results.sort(key=lambda x: x[0])
    
    # 写入 JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for _, data in results:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    print(f"摘要已保存至 {output_file}")


# === 命令行入口 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对小说文件按章节生成摘要")
    parser.add_argument("input", help="输入的小说文本文件路径（如 ./novel.txt）")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_JSONL,
                        help=f"输出的 JSONL 文件路径（默认: {DEFAULT_OUTPUT_JSONL})")
    
    args = parser.parse_args()
    
    asyncio.run(main(args.input, args.output))