import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="公务员考试岗位个性化推荐系统",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📋 公务员考试岗位个性化推荐系统")
st.markdown("---")

DATA_DIR = "data/raw/岗位表"
SCORE_DIR = "data/raw/进面分数线"
CACHE_DIR = "data/cache"

# 删除不需要的列
columns_to_drop = ["部门网站", "咨询电话1", "咨询电话2", "咨询电话3", "落户地点", "学位", "招录机关"]

@st.cache_data
def load_score_data():
    cache_path = os.path.join(CACHE_DIR, "scores.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    return None

@st.cache_data
def load_data(year, sheet_name=None, merge_all=False):
    # 如果是合并所有，从缓存读取
    if merge_all:
        cache_path = os.path.join(CACHE_DIR, f"positions_{year}.parquet")
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
        else:
            # 从Excel读取并合并
            file_path = os.path.join(DATA_DIR, f"{year}.xlsx")
            if not os.path.exists(file_path):
                return None
            
            xl = pd.ExcelFile(file_path)
            all_dfs = []
            for sheet in xl.sheet_names:
                df_sheet = pd.read_excel(file_path, sheet_name=sheet, header=1)
                all_dfs.append(df_sheet)
            df = pd.concat(all_dfs, ignore_index=True)
    elif sheet_name:
        # 单个工作表，直接从Excel读取
        file_path = os.path.join(DATA_DIR, f"{year}.xlsx")
        if not os.path.exists(file_path):
            return None
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
    else:
        return None
    
    # 删除不需要的列（如果从Excel读取的话）
    columns_to_drop = ["部门网站", "咨询电话1", "咨询电话2", "咨询电话3", "落户地点", "学位", "招录机关", "工作表"]
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 尝试合并进面分数
    score_df = load_score_data()
    if score_df is not None and '年份' in score_df.columns:
        year_scores = score_df[score_df['年份'] == year]
        if not year_scores.empty:
            # 尝试通过职位代码或部门代码合并
            merge_keys = []
            for key in ['职位代码', '部门代码']:
                if key in df.columns and key in year_scores.columns:
                    merge_keys.append(key)
            
            if merge_keys:
                # 只保留分数表中不在主表中的列（避免重复列）
                # 特别确保不要包含"招录机关"列
                score_cols = []
                for col in year_scores.columns:
                    if col not in ['年份'] + merge_keys and col not in df.columns and col != '招录机关':
                        score_cols.append(col)
                
                if score_cols:
                    merge_df = year_scores[merge_keys + score_cols].copy()
                    # 去重
                    merge_df = merge_df.drop_duplicates(subset=merge_keys)
                    # 合并
                    df = df.merge(merge_df, on=merge_keys, how='left')
    
    return df

@st.cache_data
def get_sheet_names(year):
    # 从Excel获取工作表名称（缓存中已删除工作表列）
    file_path = os.path.join(DATA_DIR, f"{year}.xlsx")
    if os.path.exists(file_path):
        xl = pd.ExcelFile(file_path)
        return xl.sheet_names
    return []

def get_available_years():
    # 首先尝试从缓存获取
    years = []
    if os.path.exists(CACHE_DIR):
        for file in os.listdir(CACHE_DIR):
            if file.startswith("positions_") and file.endswith(".parquet"):
                year = file.replace("positions_", "").replace(".parquet", "")
                if year.isdigit():
                    years.append(int(year))
        if years:
            return sorted(years)
    
    # 从原始数据获取
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            if file.endswith(".xlsx"):
                year = file.replace(".xlsx", "")
                if year.isdigit():
                    years.append(int(year))
    return sorted(years)

# ==================== 推荐算法函数 ====================

def calculate_recommendation_scores(df, user_edu=None, user_pol=None):
    """
    使用 CRITIC + 业务权重 + TOPSIS 计算推荐分数
    返回：(result_df, score_details, weights_info)
    """
    if len(df) < 2:
        # 岗位太少，无法计算，返回默认分
        result_df = df.copy()
        result_df['推荐分'] = 5.0
        return result_df, None, None
    
    # 复制数据
    df = df.copy().reset_index(drop=True)
    
    # ==================== 1. 提取指标数据 ====================
    indicators = pd.DataFrame(index=df.index)
    raw_indicators = pd.DataFrame(index=df.index)  # 保存原始值
    
    # 指标1: 进面分数（成本型：越小越好）
    if '最低面试分数' in df.columns:
        indicators['进面分数'] = pd.to_numeric(df['最低面试分数'], errors='coerce').fillna(100)
        raw_indicators['进面分数'] = indicators['进面分数']
    else:
        indicators['进面分数'] = 50  # 默认值
        raw_indicators['进面分数'] = 50
    
    # 指标2: 招考人数（效益型：越大越好）
    if '招考人数' in df.columns:
        indicators['招考人数'] = pd.to_numeric(df['招考人数'], errors='coerce').fillna(1)
        raw_indicators['招考人数'] = indicators['招考人数']
    else:
        indicators['招考人数'] = 1  # 默认值
        raw_indicators['招考人数'] = 1
    
    # 指标3: 专业要求数（成本型：越少越好）
    if '专业' in df.columns:
        def count_majors(major_str):
            if pd.isna(major_str):
                return 999
            major_str = str(major_str).strip()
            if not major_str or '不限' in major_str:
                return 999
            separators = [',', '，', '；', ';', '/', '、']
            count = 1
            for sep in separators:
                if sep in major_str:
                    parts = [p.strip() for p in major_str.split(sep) if p.strip()]
                    count = max(count, len(parts))
            return count
        indicators['专业要求数'] = df['专业'].apply(count_majors)
        raw_indicators['专业要求数'] = indicators['专业要求数']
    else:
        indicators['专业要求数'] = 5  # 默认值
        raw_indicators['专业要求数'] = 5
    
    # 指标4: 机构层级（效益型：越高越好）
    if '部门名称' in df.columns:
        def get_institution_level(name):
            if pd.isna(name):
                return 1
            name = str(name)
            if '中央' in name or '部委' in name:
                return 4
            elif '省' in name:
                return 3
            elif '市' in name:
                return 2
            else:
                return 1
        indicators['机构层级'] = df['部门名称'].apply(get_institution_level)
        raw_indicators['机构层级'] = indicators['机构层级']
    else:
        indicators['机构层级'] = 2  # 默认值
        raw_indicators['机构层级'] = 2
    
    # 指标5: 学历匹配度（适度型：越符合越好）
    if '学历' in df.columns and user_edu and user_edu != "请选择":
        edu_hierarchy = {
            "专科": 1, "大专": 1,
            "本科": 2,
            "硕士研究生": 3, "硕士": 3,
            "博士研究生": 4, "博士": 4
        }
        user_level = edu_hierarchy.get(user_edu, 2)
        
        def calculate_edu_match(edu_req):
            if pd.isna(edu_req):
                return 5
            edu_req = str(edu_req).strip()
            
            # 找出岗位要求的学历层级
            req_levels = []
            for edu_name, level in edu_hierarchy.items():
                if edu_name in edu_req:
                    req_levels.append(level)
            
            if not req_levels:
                return 5
            
            min_req = min(req_levels)
            max_req = max(req_levels) if '仅限' in edu_req else 4
            
            # 刚好匹配最好
            if min_req <= user_level <= max_req:
                if min_req == user_level and max_req == user_level:
                    return 10  # 仅限，完美匹配
                return 7  # 符合要求
            elif user_level < min_req:
                return 1  # 学历不够
            else:
                return 4  # 学历浪费
        
        indicators['学历匹配'] = df['学历'].apply(calculate_edu_match)
        raw_indicators['学历匹配'] = indicators['学历匹配']
    else:
        indicators['学历匹配'] = 5  # 默认值
        raw_indicators['学历匹配'] = 5
    
    # 指标6: 备注限制数（成本型：越少越好）
    if '备注' in df.columns:
        def count_restrictions(note):
            if pd.isna(note):
                return 0
            note = str(note)
            count = 0
            restrictions = ['仅限', '限', '要求', '需', '必须', '服务年限', '最低服务']
            for r in restrictions:
                count += note.count(r)
            return count
        indicators['备注限制数'] = df['备注'].apply(count_restrictions)
        raw_indicators['备注限制数'] = indicators['备注限制数']
    else:
        indicators['备注限制数'] = 0  # 默认值
        raw_indicators['备注限制数'] = 0
    
    # ==================== 2. 数据标准化 ====================
    def normalize_benefit(x, max_val, min_val):
        if max_val == min_val:
            return 0.5
        return (x - min_val) / (max_val - min_val)
    
    def normalize_cost(x, max_val, min_val):
        if max_val == min_val:
            return 0.5
        return (max_val - x) / (max_val - min_val)
    
    normalized = indicators.copy()
    
    # 进面分数（成本型）
    max_score = indicators['进面分数'].max()
    min_score = indicators['进面分数'].min()
    normalized['进面分数'] = indicators['进面分数'].apply(
        lambda x: normalize_cost(x, max_score, min_score)
    )
    
    # 招考人数（效益型）
    max_count = indicators['招考人数'].max()
    min_count = indicators['招考人数'].min()
    normalized['招考人数'] = indicators['招考人数'].apply(
        lambda x: normalize_benefit(x, max_count, min_count)
    )
    
    # 专业要求数（成本型）
    max_major = indicators['专业要求数'].max()
    min_major = indicators['专业要求数'].min()
    normalized['专业要求数'] = indicators['专业要求数'].apply(
        lambda x: normalize_cost(x, max_major, min_major)
    )
    
    # 机构层级（效益型）
    max_level = indicators['机构层级'].max()
    min_level = indicators['机构层级'].min()
    normalized['机构层级'] = indicators['机构层级'].apply(
        lambda x: normalize_benefit(x, max_level, min_level)
    )
    
    # 学历匹配（效益型，已经是0-10，标准化到0-1）
    max_edu = indicators['学历匹配'].max()
    min_edu = indicators['学历匹配'].min()
    normalized['学历匹配'] = indicators['学历匹配'].apply(
        lambda x: normalize_benefit(x, max_edu, min_edu)
    )
    
    # 备注限制数（成本型）
    max_restrict = indicators['备注限制数'].max()
    min_restrict = indicators['备注限制数'].min()
    normalized['备注限制数'] = indicators['备注限制数'].apply(
        lambda x: normalize_cost(x, max_restrict, min_restrict)
    )
    
    # ==================== 3. CRITIC 客观权重 ====================
    # 计算标准差
    std_devs = normalized.std()
    
    # 计算相关系数矩阵
    corr_matrix = normalized.corr()
    
    # 计算 (1 - 相关系数之和)
    corr_terms = {}
    for col in normalized.columns:
        corr_sum = corr_matrix[col].sum() - 1
        corr_terms[col] = 1 - corr_sum
    
    # 找到最小值，加偏移量保证为正
    min_corr_term = min(corr_terms.values())
    offset = 0
    if min_corr_term < 0:
        offset = abs(min_corr_term) + 0.1  # 加0.1避免等于0
    
    # 计算 CRITIC 值（加偏移量）
    critic_values = {}
    for col in normalized.columns:
        critic_values[col] = std_devs[col] * (corr_terms[col] + offset)
    
    # 归一化得到 CRITIC 权重
    total_critic = sum(critic_values.values())
    critic_weights = {col: val / total_critic for col, val in critic_values.items()}
    
    # ==================== 4. 业务权重 ====================
    business_weights = {
        '进面分数': 0.35,
        '招考人数': 0.25,
        '专业要求数': 0.15,
        '机构层级': 0.10,
        '学历匹配': 0.08,
        '备注限制数': 0.07
    }
    
    # ==================== 5. 混合权重（60% CRITIC + 40% 业务） ====================
    alpha = 0.6
    beta = 0.4
    final_weights = {}
    for col in critic_weights.keys():
        final_weights[col] = alpha * critic_weights[col] + beta * business_weights[col]
    
    # 归一化
    total_final = sum(final_weights.values())
    final_weights = {col: w / total_final for col, w in final_weights.items()}
    
    # ==================== 6. TOPSIS 排序 ====================
    # 加权标准化矩阵
    weighted = normalized.copy()
    for col in final_weights.keys():
        weighted[col] = weighted[col] * final_weights[col]
    
    # 正负理想解
    positive_ideal = weighted.max()
    negative_ideal = weighted.min()
    
    # 计算距离
    d_positive = np.sqrt(((weighted - positive_ideal) ** 2).sum(axis=1))
    d_negative = np.sqrt(((weighted - negative_ideal) ** 2).sum(axis=1))
    
    # 相对贴近度
    closeness = d_negative / (d_positive + d_negative)
    
    # 转换为 1-10 分
    df['推荐分'] = (closeness * 9 + 1).round(1)
    
    # 按推荐分排序
    df = df.sort_values('推荐分', ascending=False)
    
    # 添加排名
    df['排名'] = range(1, len(df) + 1)
    
    # ==================== 7. 准备评分详情 ====================
    score_details = []
    for idx, row in df.iterrows():
        detail = {
            '排名': row['排名'],
            '推荐分': row['推荐分']
        }
        
        # 添加各指标原始值和标准化得分（转换为0-10分）
        for col in indicators.columns:
            raw_val = raw_indicators.loc[idx, col]
            norm_score = normalized.loc[idx, col] * 10  # 0-10分
            weight = final_weights[col] * 100  # 转为百分比
            detail[f'{col}_原始值'] = raw_val
            detail[f'{col}_得分'] = round(norm_score, 1)
            detail[f'{col}_权重'] = round(weight, 1)
        
        score_details.append(detail)
    
    # 权重信息
    weights_info = {
        '指标': list(final_weights.keys()),
        'CRITIC权重': [round(critic_weights[col] * 100, 1) for col in final_weights.keys()],
        '业务权重': [round(business_weights[col] * 100, 1) for col in final_weights.keys()],
        '最终权重': [round(final_weights[col] * 100, 1) for col in final_weights.keys()]
    }
    
    return df, score_details, weights_info

available_years = get_available_years()

if not available_years:
    st.error("❌ 未找到岗位表数据文件！请确保岗位表文件夹存在并包含Excel文件。")
    st.stop()

st.sidebar.header("🔧 筛选条件")
selected_year = st.sidebar.selectbox("选择年份", available_years, index=len(available_years)-1)

sheet_names = get_sheet_names(selected_year)
merge_option = st.sidebar.checkbox("合并所有工作表", value=True)

if merge_option:
    df = load_data(selected_year, merge_all=True)
    selected_sheet = "全部合并"
else:
    selected_sheet = st.sidebar.selectbox("选择工作表", sheet_names, index=0)
    df = load_data(selected_year, sheet_name=selected_sheet)

if df is None:
    st.error(f"❌ 无法加载 {selected_year} 年的数据！")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("� 我的信息（智能匹配）")

# 学历层级映射
edu_hierarchy = {
    "专科": 1,
    "大专": 1,
    "本科": 2,
    "硕士研究生": 3,
    "硕士": 3,
    "博士研究生": 4,
    "博士": 4
}

# 用户学历选择
user_edu = st.sidebar.selectbox(
    "我的学历",
    options=["请选择", "专科", "本科", "硕士研究生", "博士研究生"],
    index=0,
    key="user_edu"
)

# 政治面貌层级映射
pol_hierarchy = {
    "群众": 1,
    "共青团员": 2,
    "中共预备党员": 3,
    "中共党员": 4
}

# 用户政治面貌选择
user_pol = st.sidebar.selectbox(
    "我的政治面貌",
    options=["请选择", "群众", "共青团员", "中共预备党员", "中共党员"],
    index=0,
    key="user_pol"
)

# 用户性别选择
user_gender = st.sidebar.selectbox(
    "我的性别",
    options=["请选择", "男", "女"],
    index=0,
    key="user_gender"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 数据概览")
st.sidebar.info(f"📈 总岗位数: {len(df)}")
st.sidebar.info(f"📊 数据列数: {len(df.columns)}")
if not merge_option:
    st.sidebar.info(f"📄 工作表: {selected_sheet}")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 筛选选项")

filtered_df = df.copy()

# 智能匹配：学历
if user_edu != "请选择" and "学历" in filtered_df.columns:
    user_edu_level = edu_hierarchy.get(user_edu, 2)
    
    def is_edu_qualified(edu_req):
        if pd.isna(edu_req):
            return True
        edu_req = str(edu_req).strip()
        
        # 不限
        if "不限" in edu_req:
            return True
        
        # 解析学历要求，找出最低和最高要求
        min_level = 1
        max_level = 4
        
        # 检查是否有"仅限"
        only = "仅限" in edu_req or "限" in edu_req and "及以上" not in edu_req
        
        # 检查具体学历
        found_levels = []
        for edu_name, level in edu_hierarchy.items():
            if edu_name in edu_req:
                found_levels.append(level)
        
        if found_levels:
            if only:
                # 仅限：必须完全匹配
                return user_edu_level in found_levels
            else:
                # 非仅限：用户学历 >= 最低要求
                return user_edu_level >= min(found_levels)
        
        # 没有找到具体学历，默认通过
        return True
    
    edu_mask = filtered_df["学历"].apply(is_edu_qualified)
    filtered_df = filtered_df[edu_mask]

# 智能匹配：政治面貌
if user_pol != "请选择" and "政治面貌" in filtered_df.columns:
    user_pol_level = pol_hierarchy.get(user_pol, 1)
    
    def is_pol_qualified(pol_req):
        if pd.isna(pol_req):
            return True
        pol_req = str(pol_req).strip()
        
        # 不限
        if "不限" in pol_req:
            return True
        
        # 检查是否匹配用户政治面貌或更高
        # 用户是共青团员，可以报：不限、共青团员
        # 用户是党员，可以报：不限、共青团员、预备党员、党员
        for pol_name, level in pol_hierarchy.items():
            if pol_name in pol_req and level <= user_pol_level:
                return True
        
        return False
    
    pol_mask = filtered_df["政治面貌"].apply(is_pol_qualified)
    filtered_df = filtered_df[pol_mask]

# 智能匹配：性别
if user_gender != "请选择" and "备注" in filtered_df.columns:
    
    def is_gender_qualified(remark):
        if pd.isna(remark):
            return True
        remark_str = str(remark).strip()
        
        # 不限性别（没有性别限制关键词）
        male_keywords = ["限男性", "仅限男性", "男性，", "，男性", "限男", "仅男"]
        female_keywords = ["限女性", "仅限女性", "女性，", "，女性", "限女", "仅女"]
        
        has_male_restrict = any(keyword in remark_str for keyword in male_keywords)
        has_female_restrict = any(keyword in remark_str for keyword in female_keywords)
        
        if user_gender == "男":
            # 用户是男性，只要不限女性或限男性都可以
            if has_female_restrict:
                return False
            return True
        else:  # 用户是女性
            # 用户是女性，只要不限男性或限女性都可以
            if has_male_restrict:
                return False
            return True
    
    gender_mask = filtered_df["备注"].apply(is_gender_qualified)
    filtered_df = filtered_df[gender_mask]

# 只保留以下筛选列（学历、政治面貌、性别已通过智能匹配处理）
allowed_columns = ["专业", "基层工作最低年限", "工作地点"]

# 文本搜索列（包含搜索）
text_search_columns = ["专业"]

for col in allowed_columns:
    if col not in df.columns:
        continue
    
    if col == "工作地点":
        # 工作地点三级筛选（支持多选）
        st.sidebar.markdown("### 📍 工作地点筛选")
        
        # 从工作地点中提取省份
        def extract_province(loc):
            if pd.isna(loc):
                return ""
            loc_str = str(loc)
            # 常见省份列表
            provinces = ["北京", "天津", "上海", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江", 
                        "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", 
                        "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海", "台湾", 
                        "内蒙古", "广西", "西藏", "宁夏", "新疆", "香港", "澳门"]
            for prov in provinces:
                if prov in loc_str:
                    return prov
            # 如果没有匹配到，返回前2-3个字符
            return loc_str[:3] if len(loc_str) >= 3 else loc_str
        
        # 提取省份选项
        all_locations = df[col].dropna()
        provinces = sorted(list(set([extract_province(loc) for loc in all_locations if extract_province(loc)])))
        
        selected_provinces = st.sidebar.multiselect(
            "选择省份（可多选）",
            options=provinces,
            default=[],
            key="province_select"
        )
        
        # 根据省份筛选
        if selected_provinces:
            province_mask = filtered_df[col].astype(str).apply(
                lambda x: any(prov in x for prov in selected_provinces)
            )
            filtered_df = filtered_df[province_mask]
        
        # 提取市（在省份筛选后）
        def extract_city(loc):
            if pd.isna(loc):
                return ""
            loc_str = str(loc)
            # 简单提取：找到"市"字
            if "市" in loc_str:
                city_end = loc_str.find("市") + 1
                return loc_str[:city_end]
            return ""
        
        # 获取市选项
        if selected_provinces:
            city_locations = filtered_df[col].dropna()
        else:
            city_locations = all_locations
        
        cities = sorted(list(set([extract_city(loc) for loc in city_locations if extract_city(loc)])))
        
        selected_cities = st.sidebar.multiselect(
            "选择市（可多选）",
            options=cities,
            default=[],
            key="city_select"
        )
        
        # 根据市筛选
        if selected_cities:
            city_mask = filtered_df[col].astype(str).apply(
                lambda x: any(city in x for city in selected_cities)
            )
            filtered_df = filtered_df[city_mask]
        
        # 提取区县（在省市筛选后）
        def extract_district(loc):
            if pd.isna(loc):
                return ""
            loc_str = str(loc)
            # 简单提取：找到"区"或"县"字
            if "区" in loc_str:
                district_end = loc_str.find("区") + 1
                return loc_str[:district_end]
            elif "县" in loc_str:
                district_end = loc_str.find("县") + 1
                return loc_str[:district_end]
            return ""
        
        # 获取区县选项
        if selected_cities:
            district_locations = filtered_df[col].dropna()
        elif selected_provinces:
            district_locations = filtered_df[col].dropna()
        else:
            district_locations = all_locations
        
        districts = sorted(list(set([extract_district(loc) for loc in district_locations if extract_district(loc)])))
        
        selected_districts = st.sidebar.multiselect(
            "选择县/区（可多选）",
            options=districts,
            default=[],
            key="district_select"
        )
        
        # 根据区县筛选
        if selected_districts:
            district_mask = filtered_df[col].astype(str).apply(
                lambda x: any(dist in x for dist in selected_districts)
            )
            filtered_df = filtered_df[district_mask]
        
        st.sidebar.markdown("---")
        
    elif col in text_search_columns:
        # 文本包含搜索
        search_value = st.sidebar.text_input(
            f"{col} (关键词搜索)",
            value="",
            key=f"text_search_{col}"
        )
        if search_value:
            # 包含搜索，不区分大小写
            filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(search_value, case=False, na=False)]
    else:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 50 and len(unique_vals) > 1:
            # 基层工作最低年限默认选择"无限制"
            default_vals = []
            if col == "基层工作最低年限":
                # 查找"无限制"相关选项
                for val in unique_vals:
                    val_str = str(val)
                    if "无限制" in val_str or "不限" in val_str or "不限制" in val_str:
                        default_vals = [val]
                        break
            
            selected = st.sidebar.multiselect(
                f"{col}",
                options=sorted(unique_vals),
                default=default_vals,
                key=f"filter_{col}"
            )
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]
        elif df[col].dtype in ['int64', 'float64']:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            if min_val < max_val:
                val_range = st.sidebar.slider(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"slider_{col}"
                )
                filtered_df = filtered_df[(filtered_df[col] >= val_range[0]) & (filtered_df[col] <= val_range[1])]

if merge_option:
    st.header(f"📋 {selected_year}年岗位数据")
else:
    st.header(f"📋 {selected_year}年岗位数据 - {selected_sheet}")
st.info(f"🔍 筛选结果: {len(filtered_df)} 个岗位")

# 初始化session state来跟踪删除的行
if 'deleted_rows' not in st.session_state:
    st.session_state.deleted_rows = set()

# 如果筛选条件改变，重置删除的行
if 'current_year' not in st.session_state or st.session_state.current_year != selected_year:
    st.session_state.deleted_rows = set()
    st.session_state.current_year = selected_year

if 'current_sheet' not in st.session_state or st.session_state.current_sheet != selected_sheet:
    st.session_state.deleted_rows = set()
    st.session_state.current_sheet = selected_sheet

# 添加选择列
display_df = filtered_df.copy()
display_df = display_df.reset_index(drop=True)
display_df['选择删除'] = False

# 使用数据编辑器让用户选择要删除的行
st.subheader("✂️ 手动删除岗位")
st.caption("勾选要删除的岗位，然后点击'删除选中岗位'按钮")

edited_df = st.data_editor(
    display_df,
    use_container_width=True,
    column_config={
        "选择删除": st.column_config.CheckboxColumn(
            "选择删除",
            help="勾选要删除的岗位",
            default=False,
        )
    },
    hide_index=True,
)

# 删除按钮
col1, col2 = st.columns([1, 3])
with col1:
    delete_clicked = st.button("🗑️ 删除选中岗位", type="primary")
    if delete_clicked:
        # 获取要删除的行索引
        rows_to_delete = edited_df[edited_df['选择删除']].index.tolist()
        if rows_to_delete:
            # 添加到已删除行集合
            for idx in rows_to_delete:
                st.session_state.deleted_rows.add(idx)
            st.success(f"已删除 {len(rows_to_delete)} 个岗位！")
            # 使用experimental_rerun来兼容旧版本Streamlit
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        else:
            st.warning("请先勾选要删除的岗位！")

# 显示最终结果（排除已删除的行）
final_df = display_df[~display_df.index.isin(st.session_state.deleted_rows)].drop(columns=['选择删除'])

# 计算推荐分数
score_details = None
weights_info = None
if len(final_df) > 0:
    final_df, score_details, weights_info = calculate_recommendation_scores(final_df, user_edu, user_pol)
    
    # 将排名和推荐分列移到前面
    cols = ['排名', '推荐分'] + [col for col in final_df.columns if col not in ['排名', '推荐分']]
    final_df = final_df[cols]

st.markdown("---")
st.subheader(f"📊 最终结果 ({len(final_df)} 个岗位)")
st.dataframe(final_df, use_container_width=True)

# 显示评分详情
if score_details is not None and weights_info is not None:
    st.markdown("---")
    
    # 选项卡：权重信息 / 评分详情
    tab1, tab2 = st.tabs(["📊 权重信息", "📋 各岗位评分详情"])
    
    with tab1:
        st.subheader("各指标权重")
        weights_df = pd.DataFrame(weights_info)
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
        
        st.caption("💡 说明：")
        st.caption("- CRITIC权重：基于数据离散度和相关性的客观权重")
        st.caption("- 业务权重：基于公务员报考常识的主观权重")
        st.caption("- 最终权重：60% CRITIC + 40% 业务权重")
    
    with tab2:
        st.subheader("各岗位评分详情")
        
        # 创建评分详情表格
        detail_rows = []
        for detail in score_details:
            row = {
                '排名': detail['排名'],
                '推荐分': detail['推荐分']
            }
            
            # 添加各指标信息
            for col in ['进面分数', '招考人数', '专业要求数', '机构层级', '学历匹配', '备注限制数']:
                row[f'{col}_原始'] = detail[f'{col}_原始值']
                row[f'{col}_得分'] = detail[f'{col}_得分']
                row[f'{col}_权重(%)'] = detail[f'{col}_权重']
            
            detail_rows.append(row)
        
        detail_df = pd.DataFrame(detail_rows)
        
        # 重新排列列顺序
        detail_cols = ['排名', '推荐分']
        for col in ['进面分数', '招考人数', '专业要求数', '机构层级', '学历匹配', '备注限制数']:
            detail_cols.extend([f'{col}_原始', f'{col}_得分', f'{col}_权重(%)'])
        
        detail_df = detail_df[detail_cols]
        
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        
        st.caption("💡 说明：")
        st.caption("- 得分：0-10分，越高越好")
        st.caption("- 权重(%)：该指标在总分中的占比")
        st.caption("- 推荐分 = 各指标得分 × 权重，再通过TOPSIS计算")

csv = final_df.to_csv(index=False).encode('utf-8-sig')
if merge_option:
    file_name = f"岗位筛选结果_{selected_year}.csv"
else:
    file_name = f"岗位筛选结果_{selected_year}_{selected_sheet}.csv"
st.download_button(
    label="📥 下载最终结果 (CSV)",
    data=csv,
    file_name=file_name,
    mime="text/csv"
)

# 重置按钮
reset_clicked = st.button("🔄 重置所有删除")
if reset_clicked:
    st.session_state.deleted_rows = set()
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

st.markdown("---")
st.caption("💡 提示：在左侧边栏选择年份、工作表和筛选条件，然后可以手动勾选删除不符合要求的岗位！")
