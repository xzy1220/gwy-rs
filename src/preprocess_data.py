import pandas as pd
import numpy as np
import os
import json

DATA_DIR = "data/raw/岗位表"
SCORE_DIR = "data/raw/进面分数线"
CACHE_DIR = "data/cache"

# 学历映射
edu_mapping = {
    "大专": 1,
    "本科": 2,
    "硕士研究生": 3,
    "博士研究生": 4
}

# 政治面貌映射
pol_mapping = {
    "不限": 1,
    "中共党员或共青团员": 2,
    "中共党员": 3
}

# 创建缓存目录
os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 60)
print("开始预加载数据...")
print("=" * 60)

# ==================== 第一步：读取并处理进面分数线 ====================
print("\n📊 处理进面分数线数据...")
all_scores_by_year = {}

score_file = os.path.join(SCORE_DIR, "国考18-26年进面分数线.xlsx")
if os.path.exists(score_file):
    xl = pd.ExcelFile(score_file)
    
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(score_file, sheet_name=sheet_name)
        # 从工作表名称中提取年份
        year = None
        for y in range(2018, 2027):
            if str(y) in sheet_name:
                year = y
                break
        if year:
            df['年份'] = year
            all_scores_by_year[year] = df
    print(f"  ✓ 已加载 {len(all_scores_by_year)} 个年份的进面分数线数据")

# ==================== 第二步：处理岗位表数据 ====================
print("\n📋 处理岗位表数据...")
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".xlsx")])

for file in files:
    year = file.replace(".xlsx", "")
    print(f"\n处理 {year}年...")
    
    file_path = os.path.join(DATA_DIR, file)
    xl = pd.ExcelFile(file_path)
    
    # 加载并合并所有工作表
    all_dfs = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet, header=1)
        df['工作表'] = sheet  # 添加工作表来源列
        all_dfs.append(df)
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # ==================== 数据合并：连接进面分数线 ====================
    if int(year) in all_scores_by_year:
        year_scores = all_scores_by_year[int(year)]
        if not year_scores.empty:
            # 尝试用职位代码+部门代码作为唯一标识
            merge_keys = []
            if '职位代码' in merged_df.columns and '职位代码' in year_scores.columns and '部门代码' in merged_df.columns and '部门代码' in year_scores.columns:
                merge_keys = ['职位代码', '部门代码']
            elif '职位代码' in merged_df.columns and '职位代码' in year_scores.columns:
                merge_keys = ['职位代码']
            elif '部门代码' in merged_df.columns and '部门代码' in year_scores.columns:
                merge_keys = ['部门代码']
            
            if merge_keys:
                # 只保留分数表中不在主表中的列
                score_cols = []
                for col in year_scores.columns:
                    if col not in ['年份'] + merge_keys and col not in merged_df.columns:
                        score_cols.append(col)
                
                if score_cols:
                    merge_df = year_scores[merge_keys + score_cols].copy()
                    merge_df = merge_df.drop_duplicates(subset=merge_keys)
                    merged_df = merged_df.merge(merge_df, on=merge_keys, how='left')
                    print(f"  ✓ 已合并进面分数线数据")
    
    # ==================== 删除冗余列 ====================
    columns_to_drop = ["部门网站", "咨询电话1", "咨询电话2", "咨询电话3", "落户地点", "学位", "招录机关", "部门名称"]
    for col in columns_to_drop:
        if col in merged_df.columns:
            merged_df = merged_df.drop(columns=[col])
    
    # ==================== 新增标准化列 ====================
    
    # 1. 学历映射列（精确匹配9种情况）
    if '学历' in merged_df.columns:
        def parse_edu_requirement(edu_req):
            if pd.isna(edu_req):
                return json.dumps([1, 2, 3, 4], ensure_ascii=False)
            
            edu_req = str(edu_req).strip()
            
            # 精确匹配9种学历要求
            edu_map = {
                "仅限博士研究生": [4],
                "仅限大专": [1],
                "仅限本科": [2],
                "仅限硕士研究生": [3],
                "大专及以上": [1, 2, 3, 4],
                "大专或本科": [1, 2],
                "本科及以上": [2, 3, 4],
                "本科或硕士研究生": [2, 3],
                "硕士研究生及以上": [3, 4],
                "不限": [1, 2, 3, 4]
            }
            
            if edu_req in edu_map:
                return json.dumps(edu_map[edu_req], ensure_ascii=False)
            
            # 如果不在9种情况中，视为不限
            return json.dumps([1, 2, 3, 4], ensure_ascii=False)
        
        merged_df['学历映射'] = merged_df['学历'].apply(parse_edu_requirement)
    
    # 2. 政治面貌映射列（精确匹配3种情况）
    if '政治面貌' in merged_df.columns:
        def parse_pol_requirement(pol_req):
            if pd.isna(pol_req):
                return 1
            
            pol_req = str(pol_req).strip()
            
            # 精确匹配3种政治面貌要求
            pol_map = {
                "不限": 1,
                "中共党员或共青团员": 2,
                "中共党员": 3
            }
            
            if pol_req in pol_map:
                return pol_map[pol_req]
            
            # 如果不在3种情况中，视为不限
            return 1
        
        merged_df['政治面貌映射'] = merged_df['政治面貌'].apply(parse_pol_requirement)
    
    # 3. 专业要求数列（分学历层次）
    if '专业' in merged_df.columns:
        def parse_major_requirements(major_str):
            if pd.isna(major_str):
                return 999, 999, 999, 999, 999
            
            major_str = str(major_str).strip()
            if not major_str or '不限' in major_str:
                return 999, 999, 999, 999, 999
            
            # 计算总专业数
            separators = [',', '，', '；', ';', '/', '、']
            total_count = 1
            for sep in separators:
                if sep in major_str:
                    parts = [p.strip() for p in major_str.split(sep) if p.strip()]
                    total_count = max(total_count, len(parts))
            
            # 分学历层次统计
            degree_counts = {
                '大专': 999,
                '本科': 999,
                '研究生': 999,
                '博士': 999
            }
            
            degree_keywords = ['大专：', '专科：', '本科：', '研究生：', '硕士：', '博士：', '硕士博士：']
            has_hierarchy = any(kw in major_str for kw in degree_keywords)
            
            if has_hierarchy:
                # 按学历层次分割
                parts = major_str.split('；')
                if len(parts) == 1:
                    parts = major_str.split(';')
                
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    
                    target_degree = None
                    content_part = part
                    
                    if '硕士博士：' in part:
                        target_degree = '研究生'
                        if '：' in part:
                            content_part = part.split('：', 1)[1].strip()
                        elif ':' in part:
                            content_part = part.split(':', 1)[1].strip()
                    elif '大专：' in part or '专科：' in part:
                        target_degree = '大专'
                        if '：' in part:
                            content_part = part.split('：', 1)[1].strip()
                        elif ':' in part:
                            content_part = part.split(':', 1)[1].strip()
                    elif '本科：' in part or '大学本科：' in part:
                        target_degree = '本科'
                        if '：' in part:
                            content_part = part.split('：', 1)[1].strip()
                        elif ':' in part:
                            content_part = part.split(':', 1)[1].strip()
                    elif '研究生：' in part or '硕士：' in part:
                        target_degree = '研究生'
                        if '：' in part:
                            content_part = part.split('：', 1)[1].strip()
                        elif ':' in part:
                            content_part = part.split(':', 1)[1].strip()
                    elif '博士：' in part:
                        target_degree = '博士'
                        if '：' in part:
                            content_part = part.split('：', 1)[1].strip()
                        elif ':' in part:
                            content_part = part.split(':', 1)[1].strip()
                    
                    if target_degree:
                        count = 1
                        for sep in separators:
                            if sep in content_part:
                                parts_list = [p.strip() for p in content_part.split(sep) if p.strip()]
                                count = max(count, len(parts_list))
                        
                        degree_counts[target_degree] = count
            else:
                # 没有学历层次，所有学历层次用同一个数量
                degree_counts['大专'] = total_count
                degree_counts['本科'] = total_count
                degree_counts['研究生'] = total_count
                degree_counts['博士'] = total_count
            
            return total_count, degree_counts['大专'], degree_counts['本科'], degree_counts['研究生'], degree_counts['博士']
        
        major_data = merged_df['专业'].apply(lambda x: pd.Series(parse_major_requirements(x)))
        merged_df['专业要求数'] = major_data[0]
        merged_df['专业要求数_大专'] = major_data[1]
        merged_df['专业要求数_本科'] = major_data[2]
        merged_df['专业要求数_研究生'] = major_data[3]
        merged_df['专业要求数_博士'] = major_data[4]
    
    # 4. 机构层级映射列
    if '机构层级' in merged_df.columns:
        def get_institution_level(level_str):
            if pd.isna(level_str):
                return 1
            level_str = str(level_str).strip()
            if level_str == '中央':
                return 4
            elif '省' in level_str or '副省级' in level_str:
                return 3
            elif '市' in level_str or '地' in level_str:
                return 2
            else:
                return 1
        
        merged_df['机构层级映射'] = merged_df['机构层级'].apply(get_institution_level)
    
    # 5. 性别要求列 和 备注限制数列
    if '备注' in merged_df.columns:
        # 性别关键词
        gender_keywords = ['限男性', '仅限男性', '男性，', '，男性', '限男', '仅男',
                          '限女性', '仅限女性', '女性，', '，女性', '限女', '仅女']
        
        def parse_gender_requirement(remark):
            if pd.isna(remark):
                return None
            remark_str = str(remark)
            
            for kw in ['限男性', '仅限男性', '男性，', '，男性', '限男', '仅男']:
                if kw in remark_str:
                    return '男'
            for kw in ['限女性', '仅限女性', '女性，', '，女性', '限女', '仅女']:
                if kw in remark_str:
                    return '女'
            return None
        
        def count_restrictions(remark):
            if pd.isna(remark):
                return 0
            
            remark_str = str(remark)
            
            # 先移除性别相关词语
            for kw in gender_keywords:
                remark_str = remark_str.replace(kw, '')
            
            # 统计限制词
            exclude_keywords = [
                '咨询电话', '联系电话', '联系方式', '电话：',
                '专业考试信息请参见', '考试大纲',
                '工资待遇', '加班补贴', '值勤岗位津贴', '探亲假', '休假',
                '请考生及时关注', '微信小程序',
                '请参见', '请咨询'
            ]
            
            # 尝试用数字标号分割
            import re
            has_numbered = re.search(r'\d+\.', remark_str)
            
            items = []
            if has_numbered:
                pattern = r'\d+\.\s*'
                items = re.split(pattern, remark_str)
            else:
                processed = remark_str.replace('，', '；').replace(',', '；').replace('。', '；').replace('；；', '；')
                items = processed.split('；')
            
            valid_count = 0
            for item in items:
                item = item.strip()
                if not item or len(item) < 3:
                    continue
                
                # 排除无关词
                excluded = False
                for kw in exclude_keywords:
                    if kw in item:
                        excluded = True
                        break
                if excluded:
                    continue
                
                valid_count += 1
            
            # 备用方案：统计"限"等词
            if valid_count == 0 and '限' in remark_str:
                old_count = 0
                old_keywords = ['仅限', '限', '要求', '需', '必须', '服务年限', '最低服务']
                for kw in old_keywords:
                    old_count += remark_str.count(kw)
                valid_count = max(0, old_count)
            
            return valid_count
        
        merged_df['性别要求'] = merged_df['备注'].apply(parse_gender_requirement)
        merged_df['备注限制数'] = merged_df['备注'].apply(count_restrictions)
    
    # 保存为Parquet
    cache_path = os.path.join(CACHE_DIR, f"positions_{year}.parquet")
    merged_df.to_parquet(cache_path, index=False)
    print(f"  ✓ 已保存: {cache_path} ({len(merged_df)} 行)")
    print(f"    - 新增列: {[col for col in merged_df.columns if col in ['学历映射', '政治面貌映射', '专业要求数', '专业要求数_大专', '专业要求数_本科', '专业要求数_研究生', '专业要求数_博士', '机构层级映射', '性别要求', '备注限制数']]}")

# ==================== 第三步：生成汇总Excel ====================
print("\n" + "=" * 60)
print("📋 正在生成Excel汇总大表...")
print("=" * 60)

all_years_data = []

for file in files:
    year = file.replace(".xlsx", "")
    cache_path = os.path.join(CACHE_DIR, f"positions_{year}.parquet")
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        df['年份'] = year
        all_years_data.append(df)
        print(f"  ✓ 加载 {year}年: {len(df)} 个岗位")

if all_years_data:
    all_data = pd.concat(all_years_data, ignore_index=True)
    print(f"\n  📊 总岗位数: {len(all_data)}")
    print(f"  📊 总列数: {len(all_data.columns)}")
    print(f"\n  📋 所有列: {all_data.columns.tolist()}")
    
    excel_path = os.path.join("data", "岗位表_汇总_2018-2026.xlsx")
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    all_data.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"\n  ✅ Excel汇总大表已生成！")
    print(f"  📁 文件: {os.path.abspath(excel_path)}")

print("\n" + "=" * 60)
print("🎉 全部完成！")
print(f"📁 缓存文件保存在: {os.path.abspath(CACHE_DIR)}")
print("=" * 60)
