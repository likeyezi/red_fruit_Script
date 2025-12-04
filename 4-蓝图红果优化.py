import json
import requests
from typing import List, Dict, Any, Optional
import time
import os
import argparse
import re

class BlueprintGenerator:
    def __init__(
            self,
            api_key: str,
            model: str = "gpt-4",
            base_url: str = "https://api.openai.com/v1"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.volumes = []
        self.chapters = {}
        self.blueprints = {}

    def load_split_data(self, split_judge_path: str):
        print(f"正在加载大故事单元文件: {split_judge_path}")
        with open(split_judge_path, 'r', encoding='utf-8') as f:
            self.volumes = [json.loads(line) for line in f]
        print(f"加载了 {len(self.volumes)} 个大故事单元")

    def load_chapter_data(self, chapters_data: List[Dict]):
        for chapter in chapters_data:
            self.chapters[chapter['chapter_index']] = chapter
        print(f"加载了 {len(self.chapters)} 个章节")

    def load_chapters_from_file(self, chapters_file: str):
        print(f"正在加载章节摘要文件: {chapters_file}")
        chapters_data = []
        with open(chapters_file, 'r', encoding='utf-8') as f:
            for line in f:
                chapters_data.append(json.loads(line))
        self.load_chapter_data(chapters_data)

    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        增强型 JSON 提取器，处理 Markdown 和 纯文本混合的情况
        """
        if not text:
            return None

        # 1. 尝试清理 Markdown 代码块标记
        text = text.replace("```json", "").replace("```", "").strip()

        # 2. 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 3. 如果失败，尝试正则提取最外层的 {}
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                clean_text = match.group(0)
                try:
                    return json.loads(clean_text)
                except:
                    pass
            return None

    def call_llm_api(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096, 
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"  [API Error] Status: {response.status_code} - {response.text[:100]}")
                return None

        except Exception as e:
            print(f"  [Network Error] {e}")
            return None

    def generate_single_blueprint_with_retry(
        self, 
        volume: Dict, 
        previous_data: Dict = None, 
        max_retries: int = 3
    ) -> Optional[Dict]:
        """
        生成单个蓝图，包含上下文衔接逻辑
        :param volume: 当前单元数据
        :param previous_data: 上一单元生成的结果（用于衔接剧情）
        :param max_retries: 最大重试次数
        """
        
        # 1. 准备当前单元的章节内容
        volume_chapters = []
        for chap_idx in range(volume['start_chapter'], volume['end_chapter'] + 1):
            if chap_idx in self.chapters:
                chapter = self.chapters[chap_idx]
                # 截取前3000字以防 Token 溢出
                content_preview = chapter.get('content', '')[:3000]
                volume_chapters.append({
                    'title': chapter['title'],
                    'summary': chapter.get('summary', ''),
                    'content_preview': content_preview
                })

        # 2. 构建前情回顾 Prompt (实现零秒衔接的核心)
        previous_context_prompt = ""
        if previous_data:
            # 提取上一集的看点
            prev_analysis = previous_data.get('analysis', '')
            if isinstance(prev_analysis, dict):
                prev_analysis_str = json.dumps(prev_analysis, ensure_ascii=False)
            else:
                prev_analysis_str = str(prev_analysis)
            
            # 提取上一集集纲的最后一段作为钩子
            prev_outline = str(previous_data.get('episode_outline', ''))
            # 取最后600字以确保包含完整的悬念场景
            prev_outline_hook = prev_outline[-600:] if len(prev_outline) > 600 else prev_outline

            previous_context_prompt = f"""
### 【重要！前情回顾与衔接要求】
上一集（第 {volume['volume_index'] - 1} 集）生成信息如下，请务必阅读：
1. **上一集核心看点**：{prev_analysis_str}
2. **上一集结尾剧情（最后留下的钩子/悬念）**：
   "...{prev_outline_hook}"

**【续写强制指令】**：
- **零秒衔接**：本集的开篇**必须**直接回应上一集结尾留下的悬念。不要重新介绍背景。
- **场景延续**：如果上一集结尾是在宴会上被嘲讽，这一集开头必须紧接着打脸，严禁直接跳转到“第二天”或无关场景。
- **情绪承接**：保持上一集结尾的情绪张力，不要断崖式降温。
"""
        else:
            previous_context_prompt = "这是全剧的第一集，请务必精彩开篇，迅速立住人设和核心矛盾，切入点一定要第一时间吸引观众眼球。"

        # 3. 完整的 System Prompt (无删减)
        system_prompt = """
你是一位精通红果/男频短剧的“金牌编剧”。你深知短剧的核心公式：
一个有意思的金手指 + 主角一直在装逼的路上 + 流畅剧情 + 反转 + 卡点。
你的任务是将小说章节改编为节奏极快、画面感强、情绪极致的短剧集纲。

框架问题：
短剧最重要的就是套框架，想好框架之后，再去跟风。
跟风的话，第一，最简单的就是性转，男主变成女主，女主变成男主。
第二，套用人物关系，然后换一个金手指，比如《财富自由：我能看见未来新闻》，他的亮点一个就是能看到未来的新闻、先知先觉，第二个就是女主还是富家千金。
所以我们可以换一个金手指，然后直接套用人设，如《我有一个未来手机》、《我能看到未来朋友圈》、《我十年后的老婆给我打电话并且还帮我追现在的她》……根据这一个框架，就已经过了三个。
第三，增加新人物的出现，或者把原有的人物关系进行变动。这点太过复杂，暂时不用学。

【跟核心、强创新、强联系、强爽点、强素材、节奏快、轻喜剧、美女多】
切记：一定要密集又快速、且有画面感的小爽点，不要无脑一直在嘴炮逼逼
最主要的还是开头、第一集的切入点，然后就是一卡结尾剧情、二卡结尾剧情、三卡结尾剧情的设定，一定要震惊或者是核心爽点的强烈爆发
集纲模板：
第一集：立人设，核心看点，主线。下半部分反派登场或者结尾就展开剧情。
第二集：展开剧情或者铺垫（如果第一集不好展开的话），结尾反转
第三集：继续拉扯，结尾主角装逼，再次反转
第四集：结尾剧情，展开新剧情，新的装逼剧情
第五集——第六集：过渡剧情、再次装逼或者是成长，但必须要跟一卡高潮剧情进行关联。
第七集：开始铺垫一卡剧情
    第八集：拉扯，结尾反转
第九集：继续拉扯，再次反转
第十集：拉扯，真正反转，震惊所有人，大爽点爆发，等于小说章节的上架高潮。
（注意：第一集切入点一定要吸引人，必须第一时间吸引观众的眼球。同时短剧的第七集开始很重要，必须要开始铺垫高潮了、大装逼的场面）

短剧套路：主角做了什么事情或者应该怎么怎么样——遭受众人嘲讽、质疑——然后主角进行反转、装逼——众人不信，继续嘲讽，甚至是生气（进行情绪升级）——主角再次进行反转——众人震惊——然后进行剧情收尾、展开新的剧情——主角又做了什么事情——遭受众人嘲讽、质疑……接着就是无限循环
然后要注意的就是：第一，看你是想快节奏还是大高潮慢慢拉扯。快节奏可以把剧情快速发展，慢节奏拉高潮就是多一点拉扯、嘴炮等，但一定要注意不要太多嘴炮，适当性的增加主角小装逼的剧情
第二，每次男主装逼的时候必须要有“观众”在场！就是要有观众在场
第三，就是场面、地点、人物的升级。（这个尤为重要，打个比方，比如你最开始只有一个“观众”，就是在家里的时候，男主和女主在一起，男主做了事情，导致女主震惊；接着“观众”数量增加或者是地点的更换，增加了反派、女主妹妹，然后换一个装逼点，那么这里的爽点就肯定会比之前的爽；接着宴会厅什么的，邀请了南省所有名贵，这里就等于大高潮，汇聚了所有南省“观众”）——以上，就是节奏和爽点的升级，123或者是125

解决结果的方式必须是夸张式、好玩的。透过现象看本质。你的目的是主角穿越，但市场上大多数穿越都太平凡了，所以直接弄大运开到三楼来撞死主角穿越。又比如保镖的那个男主用女主的黑丝和反派狙击枪中门对狙。又或者是治病的时候，用一米的长针治病。
要想清楚你的结果想要的是什么，但是方式必须是那种夸张式剧情，这样才好看、好玩

人物群像：出场人物过多不知道怎么控场的时候，利用空姐切割方式来铺垫人物，采取递进人物登场迅速，如果必须要同步登场，就应该让不应该说话的人直接消失。
故事找准打点：单元剧情的时候，当你要写的主线是什么，铺垫的爽点是什么，一切人物，故事，道具，剧情都要围绕一个点展开，明确观众想看什么，将这个点给打透。
新剧情或者换地图的切入点：任何剧情的开局切入点，都需要进行设计，例如铺垫人物，人物动机，故事主线，用对话台词，群演，或者冲突来提前铺垫，然后后续剧情围绕前期铺垫的主线发展，将主线前置，用人物，台词，场景的空间关系来具象人物关系，剧情发展。
剧情的发展：起承转合是任何故事的发展，在短剧尤为重要，大致要让观众明确一个点，过去发生了什么，现在在发生什么，接下来要发生什么。
请输出严格的 JSON 格式。
"""

        # 4. 完整的 Style Sample (无删减)
        style_sample = """
        示例风格参考：
        叶秋拿着鲜花和戒指，满脸笑容的敲响了柳如烟家的房间，柳如烟打开房门看到是叶秋有些意外和惊慌。叶秋没有发现这一点，而是单膝下跪向柳如烟求婚。柳如烟眉头一皱，表示自己从来没打算和叶秋结婚。这时张伟提着裤子从卧室走了出来，一把搂住柳如烟，并看着叶秋发出嘲笑，你这傻小子居然还真的来求婚了。叶秋明白了一切，气的满眼通红，质问柳如烟为什么，明明自己对她那么好。但柳如烟反而理直气壮，表示她只是看在叶秋每个月都把工资上交，所以和叶秋玩玩而已，是叶秋自己一厢情愿。而且叶秋三年挣的钱还没张伟一晚上花的多，癞蛤蟆想吃天鹅肉，叶秋不配。
叶秋握紧拳头，愤怒到了极点，这时他突然感觉一阵眩晕，眼睛发痛，他眨了眨眼，发现眼前两人的头顶上出现了属性词条。【柳如烟，拜金，无工作，已被包养两年半】【张伟，好色，已有家室】叶秋OS，我这是觉醒异能，能看到其他人的属性词条了？
柳如烟看着叶秋一直红着眼睛盯着她看，厌恶的表示，要是叶秋还是个男人的话，就有点自知之明，别再纠缠她了，张伟可以给她真正想要的生活。叶秋冷笑，表示原来你的生活就是当小三啊，这种生活我确实给不了你。
        """

        # 5. 完整的 User Prompt (无删减，并整合了前情回顾)
        prompt = f"""
请根据以下小说原始素材，创作一份符合【红果男频爆款短剧】标准的集纲。

{previous_context_prompt}

当前单元原始章节素材：
{json.dumps(volume_chapters, ensure_ascii=False, indent=2)}

### 核心创作理论（必须严格遵守）：
1. **无限爽点循环**：严格执行“主角做事 -> 遭受嘲讽/质疑 -> 初步反转 -> 众人不信/生气（情绪升级） -> 彻底打脸/展示金手指 -> 众人震惊 -> 留下悬念”的循环结构 。
2. **观众见证原则**：主角每次装逼或反转时，**必须要有“观众”在场**！通过路人、反派、美女的反应（从鄙视到震惊）来衬托爽点 。
3. **夸张式解决**：拒绝平淡。解决问题的方式必须是**夸张式、好玩、画面感强**的（例如：穿越是被大运撞上三楼；治病用一米长的针）。
4. **节奏卡点**：
   - 如果是**第1集**：必须立住人设，确立核心看点，结尾反派登场或展开冲突 。
   - 如果是**第10集/高潮集**：必须是真正的大反转，震惊所有人，相当于小说上架前的大高潮 。
   - 每集结尾（卡点）必须停在**最大的悬念**或**爽点爆发前一秒**。
   - **承上启下**：开头必须接住上一集的钩子，结尾必须抛出新的钩子。

### 任务一：生成结构化蓝图 (analysis)
请分析本单元并提取以下字段：
- 【核心看点】：一句话概括本集金手指或夸张的爽点。
- 【情绪拉扯】：描述“嘲讽-不信-生气”的情绪升级过程。
- 【场面升级】：描述本集涉及的观众层级（如：只有女友 -> 围观群众 -> 全城名流）。

### 任务二：撰写短剧集纲 (episode_outline)
**这是核心任务**。请模仿以下风格，撰写一段300字左右的剧本大纲：
{style_sample}

**写作要求**：
1. **拒绝嘴炮**：去除无意义对话，将文字转化为**密集的动作和画面**。
2. **内心独白**：主角心理活动用 "主角名OS" 表示。
3. **金手指可视化**：系统/属性面板内容用【】包裹。
4. **格式要求**：纯叙述性文字，不要带小标题，像讲故事一样写。
5. **结尾必须有悬念**：必须有一个未解决的问题或未被确认的事件，为后续集延续提供动力。
6. **字数要求**：请将爽点集中在三百字左右都讲完。

### 输出格式（JSON）：
{{
    "analysis": "结构化分析内容...",
    "episode_outline": "集纲正文内容..."
}}
"""
        
        # 6. 执行 API 调用（带重试）
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"  ...正在进行第 {attempt + 1} 次重试...")
            
            raw_content = self.call_llm_api(prompt, system_prompt)
            
            if raw_content:
                data = self._extract_json(raw_content)
                if data and 'analysis' in data and 'episode_outline' in data:
                    return data
                else:
                    print(f"  [Parse Error] JSON解析失败或字段缺失 (尝试 {attempt + 1}/{max_retries})")
            
            time.sleep(2)

        print(f"  [Failure] 超过最大重试次数，跳过此单元。")
        return None

    def _save_results(self, output_file: str):
        try:
            # 将字典转换为列表并排序，保证顺序
            sorted_blueprints = [self.blueprints[k] for k in sorted(self.blueprints.keys())]
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sorted_blueprints, f, ensure_ascii=False, indent=2)
            print(f"  [系统] 进度已自动保存至: {output_file}")
        except Exception as e:
            print(f"  [错误] 自动保存失败: {e}")

    def generate_all_blueprints(self, output_file: str = "blueprints.json"):
        print(f"开始生成故事蓝图（使用模型: {self.model}）...")
        
        # === 新增变量：用于存储上一集的生成结果，实现上下文记忆 ===
        last_generated_data = None

        # 使用 try-finally 结构确保异常退出也能保存
        try:
            for i, volume in enumerate(self.volumes):
                vol_idx = volume['volume_index']
                print(f"正在生成第 {i + 1}/{len(self.volumes)} 个大故事的蓝图 (Volume {vol_idx})...")

                # 调用生成，传入上一集的数据
                result_data = self.generate_single_blueprint_with_retry(
                    volume, 
                    previous_data=last_generated_data, 
                    max_retries=3
                )
                
                if result_data:
                    # 更新上下文记忆
                    last_generated_data = result_data

                    # 处理 analysis 字段，确保是字符串以便后续存储
                    raw_analysis = result_data.get('analysis', '')
                    if isinstance(raw_analysis, dict):
                        analysis_str = json.dumps(raw_analysis, ensure_ascii=False, indent=2)
                    else:
                        analysis_str = str(raw_analysis).strip()

                    self.blueprints[vol_idx] = {
                        'volume_index': vol_idx,
                        'start_chapter': volume['start_chapter'],
                        'end_chapter': volume['end_chapter'],
                        'start_title': volume['start_title'],
                        'end_title': volume['end_title'],
                        'blueprint': analysis_str,  # 存储分析
                        'episode_outline': str(result_data.get('episode_outline', '')).strip(), # 存储正文
                        'chapter_count': volume['chapter_count']
                    }
                    print(f"✓ 第{vol_idx}集 生成成功")
                    
                    # 打印预览
                    outline_preview = str(result_data.get('episode_outline', ''))[:50].replace('\n', '')
                    print(f"  集纲预览: {outline_preview}...")
                else:
                    # 如果生成失败，为了安全起见，重置上下文，防止下一集接在错误的幻觉上
                    last_generated_data = None
                    
                    self.blueprints[vol_idx] = {
                        'volume_index': vol_idx,
                        'blueprint': "GENERATION_FAILED",
                        'episode_outline': "GENERATION_FAILED"
                    }
                    print(f"✗ 第{vol_idx}集 生成失败")
                
                # 每10集保存一次，降低IO频率
                if (i + 1) % 10 == 0:
                    self._save_results(output_file)

                # 避免触发 API 速率限制
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n[警告] 检测到用户停止程序 (Ctrl+C)")
            print("正在保存已生成的数据，请稍候...")
        
        finally:
            self._save_results(output_file)
            print(f"程序结束。所有蓝图已确认保存到: {output_file}")
        
        return self.blueprints


def main():
    parser = argparse.ArgumentParser(description="生成小说大故事单元的情节蓝图")
    parser.add_argument("--api_key", default="2d4b537d-f8b8-4536-955a-fb1b7b7a5101", help="API 密钥")
    parser.add_argument("--model", default="doubao-1-5-pro-32k-250115", help="使用的模型名称")
    parser.add_argument("--base_url", default="https://ark.cn-beijing.volces.com/api/v3", help="API 基础 URL")
    # parser.add_argument("--api_key", default="sk-dlflTkRgeCoUTXPA17M7JR11mUZwzNireXabwWisKyDluuce", help="API 密钥")
    # parser.add_argument("--model", default="gemini-3-pro-preview", help="使用的模型名称")
    # parser.add_argument("--base_url", default="https://jeniya.cn/v1", help="API 基础 URL")
    parser.add_argument("--split_file", default="volume_split_result.jsonl", help="大故事单元划分文件路径")
    parser.add_argument("--chapters_file", default="chapter_summaries.jsonl", help="章节摘要文件路径")
    parser.add_argument("--output_file", default="blueprints.json", help="输出蓝图文件路径")

    args = parser.parse_args()

    generator = BlueprintGenerator(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url
    )

    if os.path.exists(args.split_file) and os.path.exists(args.chapters_file):
        generator.load_split_data(args.split_file)
        generator.load_chapters_from_file(args.chapters_file)
        generator.generate_all_blueprints(args.output_file)
    else:
        print("错误：找不到输入文件，请检查文件路径。")

if __name__ == "__main__":
    main()