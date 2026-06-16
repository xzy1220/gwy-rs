import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import re

st.set_page_config(
    page_title="公务员考试岗位个性化推荐系统",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📋 公务员考试岗位个性化推荐系统")
st.markdown("---")

DATA_DIR = "data/raw/岗位表"
CACHE_DIR = "data/cache"

edu_hierarchy = {
    "大专": 1,
    "本科": 2,
    "硕士研究生": 3,
    "博士研究生": 4
}

pol_hierarchy = {
    "群众": 1,
    "共青团员": 2,
    "中共党员": 3
}

@st.cache_data(ttl=3600, max_entries=20, show_spinner="正在加载数据...")
def load_data(year, sheet_name=None, merge_all=True):
    """
    加载数据，优先从Parquet缓存读取
    """
    cache_path = os.path.join(CACHE_DIR, f"positions_{year}.parquet")
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        if merge_all:
            return df
        elif sheet_name and '工作表' in df.columns:
            return df[df['工作表'] == sheet_name].copy()
        return df
    else:
        # 如果没有缓存，从Excel读取（备用方案）
        file_path = os.path.join(DATA_DIR, f"{year}.xlsx")
        if not os.path.exists(file_path):
            return None
        
        if merge_all:
            xl = pd.ExcelFile(file_path)
            all_dfs = []
            for sheet in xl.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet, header=1)
                df['工作表'] = sheet
                all_dfs.append(df)
            return pd.concat(all_dfs, ignore_index=True)
        elif sheet_name:
            return pd.read_excel(file_path, sheet_name=sheet_name, header=1)
        else:
            return None

@st.cache_data(ttl=3600)
def get_sheet_names(year):
    """
    获取指定年份的所有工作表名称
    """
    file_path = os.path.join(DATA_DIR, f"{year}.xlsx")
    if os.path.exists(file_path):
        xl = pd.ExcelFile(file_path)
        return xl.sheet_names
    return []

@st.cache_data(ttl=3600)
def get_available_years():
    """
    获取所有可用的年份
    """
    years = []
    # 先从缓存目录获取
    if os.path.exists(CACHE_DIR):
        for file in os.listdir(CACHE_DIR):
            if file.startswith("positions_") and file.endswith(".parquet"):
                year = file.replace("positions_", "").replace(".parquet", "")
                if year.isdigit():
                    years.append(int(year))
        if years:
            return sorted(years)
    
    # 再从原始目录获取
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            if file.endswith(".xlsx"):
                year = file.replace(".xlsx", "")
                if year.isdigit():
                    years.append(int(year))
    return sorted(years)

@st.cache_data(ttl=3600)
def build_major_category_map(df):
    mapping = {}
    majors = df['专业'].dropna().astype(str)
    for m in majors:
        parts = re.split(r'[，,；;、/\n\r]+', m)
        parts = [p.strip() for p in parts if p.strip()]
        cats = [p for p in parts if p.endswith('类') and len(p) <= 10]
        specs = [p for p in parts if not p.endswith('类') and len(p) <= 20]
        for cat in cats:
            for spec in specs:
                mapping.setdefault(spec, set()).add(cat)
    return mapping

@st.cache_data(ttl=1800, show_spinner="正在计算推荐分数...")
def calculate_recommendation_scores(df, user_edu=None, user_pol=None, user_gender=None, selected_region=None, selected_major=None, selected_work_years=None):
    """
    使用 CRITIC + TOPSIS 计算推荐分数
    返回：result_df, score_details, weights_info, calculation_data
    其中 calculation_data 包含所有中间计算数据用于导出Excel
    """
    if len(df) < 2:
        result_df = df.copy()
        result_df['推荐分'] = 5.0
        result_df['排名'] = 1
        return result_df, None, None, None
    
    df = df.copy().reset_index(drop=True)
    
    # 初始化指标
    indicators = pd.DataFrame(index=df.index)
    raw_indicators = pd.DataFrame(index=df.index)
    
    # 1. 进面分数 - 成本型指标（越小越好）
    if '最低面试分数' in df.columns:
        indicators['进面分数'] = pd.to_numeric(df['最低面试分数'], errors='coerce').fillna(100)
        raw_indicators['进面分数'] = indicators['进面分数']
    else:
        indicators['进面分数'] = 50
        raw_indicators['进面分数'] = 50
    
    # 2. 招考人数 - 效益型指标（越大越好）
    if '招考人数' in df.columns:
        indicators['招考人数'] = pd.to_numeric(df['招考人数'], errors='coerce').fillna(1)
        raw_indicators['招考人数'] = indicators['招考人数']
    else:
        indicators['招考人数'] = 1
        raw_indicators['招考人数'] = 1
    
    # 3. 专业要求数 - 成本型指标（越小越好）
    # 根据用户学历选择对应的专业要求数列
    major_col = '专业要求数'
    if user_edu and user_edu != "请选择":
        if user_edu == "大专" and '专业要求数_大专' in df.columns:
            major_col = '专业要求数_大专'
        elif user_edu == "本科" and '专业要求数_本科' in df.columns:
            major_col = '专业要求数_本科'
        elif user_edu == "硕士研究生" and '专业要求数_研究生' in df.columns:
            major_col = '专业要求数_研究生'
        elif user_edu == "博士研究生" and '专业要求数_博士' in df.columns:
            major_col = '专业要求数_博士'
    
    if major_col in df.columns:
        indicators['专业要求数'] = pd.to_numeric(df[major_col], errors='coerce').fillna(999)
        # 处理999表示无限制
        indicators['专业要求数'] = indicators['专业要求数'].replace(999, 0)
        raw_indicators['专业要求数'] = indicators['专业要求数']
    else:
        indicators['专业要求数'] = 0
        raw_indicators['专业要求数'] = 0
    
    # 4. 机构层级 - 效益型指标（越大越好）
    if '机构层级映射' in df.columns:
        indicators['机构层级'] = pd.to_numeric(df['机构层级映射'], errors='coerce').fillna(1)
        raw_indicators['机构层级'] = indicators['机构层级']
    else:
        indicators['机构层级'] = 1
        raw_indicators['机构层级'] = 1
    
    # 5. 学历匹配度 - 适度型指标（越符合越好）
    if '学历映射' in df.columns and user_edu and user_edu != "请选择":
        user_level = edu_hierarchy.get(user_edu, 2)
        
        def calculate_edu_match_score(edu_json):
            if pd.isna(edu_json):
                return 5
            try:
                allowed_list = json.loads(edu_json)
            except:
                return 5
            
            if not allowed_list:
                return 5
            
            count = len(allowed_list)
            # 根据学历映射列长度算分：1个学历→10分；2个→8分；3个→6分；4个→4分
            if count == 1:
                return 10
            elif count == 2:
                return 8
            elif count == 3:
                return 6
            elif count == 4:
                return 4
            else:
                return 5
        
        indicators['学历匹配度'] = df['学历映射'].apply(calculate_edu_match_score)
        raw_indicators['学历匹配度'] = indicators['学历匹配度']
    else:
        indicators['学历匹配度'] = 5
        raw_indicators['学历匹配度'] = 5
    
    # 6. 备注限制数 - 成本型指标（越小越好）
    if '备注限制数' in df.columns:
        indicators['备注限制数'] = pd.to_numeric(df['备注限制数'], errors='coerce').fillna(0)
        raw_indicators['备注限制数'] = indicators['备注限制数']
    else:
        indicators['备注限制数'] = 0
        raw_indicators['备注限制数'] = 0
    
    # ==================== 数据归一化 ====================
    normalized = pd.DataFrame(index=df.index)
    
    # 效益型指标（越大越好）：(x - min) / (max - min)
    benefit_cols = ['招考人数', '机构层级', '学历匹配度']
    # 成本型指标（越小越好）：(max - x) / (max - min)
    cost_cols = ['进面分数', '专业要求数', '备注限制数']
    
    for col in benefit_cols:
        max_val = indicators[col].max()
        min_val = indicators[col].min()
        if max_val == min_val:
            normalized[col] = 0.5
        else:
            normalized[col] = (indicators[col] - min_val) / (max_val - min_val)
    
    for col in cost_cols:
        max_val = indicators[col].max()
        min_val = indicators[col].min()
        if max_val == min_val:
            normalized[col] = 0.5
        else:
            normalized[col] = (max_val - indicators[col]) / (max_val - min_val)
    
    # ==================== CRITIC 客观权重 ====================
    # 计算标准差
    std_devs = normalized.std()
    
    # 计算相关系数矩阵
    corr_matrix = normalized.corr()
    
    # 计算冲突性指标：对每一个指标，计算它和其他所有指标相关系数之和，然后用1减去这个和
    corr_terms = {}
    for col in normalized.columns:
        corr_sum = corr_matrix[col].sum() - 1  # 减去自身
        corr_terms[col] = 1 - corr_sum
    
    # 确保冲突性指标为正，添加偏移量
    min_corr_term = min(corr_terms.values())
    offset = 0
    if min_corr_term < 0:
        offset = abs(min_corr_term) + 0.1
    
    # 计算 CRITIC 值
    critic_values = {}
    for col in normalized.columns:
        critic_values[col] = std_devs[col] * (corr_terms[col] + offset)
    
    # 归一化得到权重
    total_critic = sum(critic_values.values())
    final_weights = {col: val / total_critic for col, val in critic_values.items()}
    
    # ==================== TOPSIS 排序 ====================
    # 加权标准化矩阵
    weighted = normalized.copy()
    for col in final_weights.keys():
        weighted[col] = weighted[col] * final_weights[col]
    
    # 确定正理想解和负理想解
    positive_ideal = weighted.max()
    negative_ideal = weighted.min()
    
    # 计算距离
    d_positive = np.sqrt(((weighted - positive_ideal) ** 2).sum(axis=1))
    d_negative = np.sqrt(((weighted - negative_ideal) ** 2).sum(axis=1))
    
    # 计算相对贴近度
    closeness = d_negative / (d_positive + d_negative)
    
    # 转化为 1-10 分
    df['推荐分'] = (closeness * 9 + 1).round(1)
    
    # 排序
    sorted_indices = df.sort_values('推荐分', ascending=False).index
    df = df.loc[sorted_indices].reset_index(drop=True)
    
    indicators = indicators.loc[sorted_indices].reset_index(drop=True)
    raw_indicators = raw_indicators.loc[sorted_indices].reset_index(drop=True)
    normalized = normalized.loc[sorted_indices].reset_index(drop=True)
    weighted = weighted.loc[sorted_indices].reset_index(drop=True)
    d_positive = d_positive.loc[sorted_indices].reset_index(drop=True)
    d_negative = d_negative.loc[sorted_indices].reset_index(drop=True)
    closeness = closeness.loc[sorted_indices].reset_index(drop=True)
    
    df['排名'] = range(1, len(df) + 1)
    
    # 准备详细评分信息
    score_details = []
    all_cols = ['进面分数', '招考人数', '专业要求数', '机构层级', '学历匹配度', '备注限制数']
    
    for i in range(len(df)):
        row = df.iloc[i]
        detail = {
            '排名': row['排名'],
            '推荐分': row['推荐分']
        }
        
        for col in all_cols:
            if col in indicators.columns:
                raw_val = raw_indicators.iloc[i, indicators.columns.get_loc(col)]
                norm_score = normalized.iloc[i, normalized.columns.get_loc(col)] * 10
                weight = final_weights[col] * 100 if col in final_weights else 0
            else:
                if col == '进面分数':
                    raw_val = 50
                elif col == '招考人数':
                    raw_val = 1
                elif col == '专业要求数':
                    raw_val = 0
                elif col == '机构层级':
                    raw_val = 1
                elif col == '学历匹配度':
                    raw_val = 5
                elif col == '备注限制数':
                    raw_val = 0
                norm_score = 5
                weight = 0
            
            detail[f'{col}_原始值'] = raw_val
            detail[f'{col}_得分'] = round(norm_score, 1)
            detail[f'{col}_权重'] = round(weight, 1)
        
        score_details.append(detail)
    
    weights_info = {
        '指标': all_cols,
        'CRITIC权重': [round(final_weights[col] * 100, 1) if col in final_weights else 0 for col in all_cols]
    }
    
    # 准备所有计算数据用于导出
    calculation_data = {
        'user_info': {
            'Gender': user_gender if user_gender and user_gender != "请选择" else "未选择",
            'Education': user_edu if user_edu and user_edu != "请选择" else "未选择",
            'Political': user_pol if user_pol and user_pol != "请选择" else "未选择",
            'Region': selected_region if selected_region else "未选择",
            'Major': selected_major if selected_major else "未选择",
            'WorkYears': selected_work_years if selected_work_years else "未选择",
            'Positions': len(df)
        },
        'raw_indicators': raw_indicators.copy(),
        'normalized': normalized.copy(),
        'weighted': weighted.copy(),
        'final_weights': final_weights.copy(),
        'std_devs': std_devs.copy(),
        'corr_terms': corr_terms.copy(),
        'critic_values': critic_values.copy(),
        'positive_ideal': positive_ideal.copy(),
        'negative_ideal': negative_ideal.copy(),
        'd_positive': d_positive.copy(),
        'd_negative': d_negative.copy(),
        'closeness': closeness.copy(),
        'original_df': df.copy()
    }
    
    return df, score_details, weights_info, calculation_data

# ==================== 主程序开始 ====================

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
    df = load_data(selected_year, sheet_name=selected_sheet, merge_all=False)

if df is None:
    st.error(f"❌ 无法加载 {selected_year}年的数据！")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("👤 我的信息（智能匹配）")

# 用户信息 - 学历
user_edu = st.sidebar.selectbox(
    "我的学历",
    options=["请选择", "大专", "本科", "硕士研究生", "博士研究生"],
    index=0,
    key="user_edu"
)

# 用户信息 - 政治面貌
user_pol = st.sidebar.selectbox(
    "我的政治面貌",
    options=["请选择", "群众", "共青团员", "中共党员"],
    index=0,
    key="user_pol"
)

# 用户信息 - 性别
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

# 重置筛选按钮
reset_col1, reset_col2 = st.sidebar.columns([1, 1])
with reset_col1:
    if st.button("🔄 重置筛选", use_container_width=True):
        # 清除所有筛选相关的 session_state
        for key in list(st.session_state.keys()):
            if key.startswith(('filter_', 'text_search_', 'province_', 'city_', 'district_', 'slider_')):
                del st.session_state[key]
        st.session_state.current_page = 0
        st.session_state.result_page = 0
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 筛选选项")

filtered_df = df.copy()

# ==================== 智能匹配模块 ====================

# 1. 学历匹配
if user_edu != "请选择" and '学历映射' in filtered_df.columns:
    user_edu_level = edu_hierarchy.get(user_edu, 2)
    
    def is_edu_qualified(edu_json):
        if pd.isna(edu_json):
            return True
        try:
            allowed_list = json.loads(edu_json)
            return user_edu_level in allowed_list
        except:
            return True
    
    edu_mask = filtered_df['学历映射'].apply(is_edu_qualified)
    filtered_df = filtered_df[edu_mask]

# 2. 政治面貌匹配
if user_pol != "请选择" and '政治面貌映射' in filtered_df.columns:
    user_pol_level = pol_hierarchy.get(user_pol, 1)
    
    def is_pol_qualified(pol_code):
        if pd.isna(pol_code):
            return True
        # 政治面貌映射：1=不限，2=中共党员或共青团员，3=中共党员
        if pol_code == 1:  # 不限
            return True
        elif pol_code == 2:  # 中共党员或共青团员
            return user_pol_level >= 2
        elif pol_code == 3:  # 中共党员
            return user_pol_level >= 3
        return True
    
    pol_mask = filtered_df['政治面貌映射'].apply(is_pol_qualified)
    filtered_df = filtered_df[pol_mask]

# 3. 性别匹配
if user_gender != "请选择" and '性别要求' in filtered_df.columns:
    def is_gender_qualified(gender_req):
        if pd.isna(gender_req):
            return True
        if gender_req is None:
            return True
        if user_gender == '男':
            return gender_req == '男' or gender_req is None
        else:  # 女
            return gender_req == '女' or gender_req is None
    
    gender_mask = filtered_df['性别要求'].apply(is_gender_qualified)
    filtered_df = filtered_df[gender_mask]

# ==================== 其他筛选模块 ====================

# 存储用户筛选信息用于导出
selected_region = ""
selected_major = ""
selected_work_years = ""

allowed_columns = ["专业", "基层工作最低年限", "工作地点"]
text_search_columns = ["专业"]

for col in allowed_columns:
    if col not in df.columns:
        continue
    
    if col == "工作地点":
        # 工作地点三级筛选
        st.sidebar.markdown("### 📍 工作地点筛选")

        # ---- 工具函数：剥离省/自治区/直辖市前缀 ----
        def _strip_province_prefix(loc_str):
            """去掉「XX省」「XX自治区」「北京市」等前缀，返回剩余部分"""
            # 按从长到短匹配，避免 "壮族自治区" 被 "自治区" 抢先匹配
            for suffix in ['壮族自治区', '回族自治区', '维吾尔自治区', '自治区', '省']:
                idx = loc_str.find(suffix)
                if idx >= 0:
                    return loc_str[idx + len(suffix):]
            # 直辖市
            for muni in ['北京市', '上海市', '天津市', '重庆市']:
                if loc_str.startswith(muni):
                    return loc_str[len(muni):]
            return loc_str  # 无省级前缀

        def extract_province(loc):
            if pd.isna(loc):
                return ""
            loc_str = str(loc)
            provinces = ["北京", "天津", "上海", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江",
                        "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
                        "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海", "台湾",
                        "内蒙古", "广西", "西藏", "宁夏", "新疆", "香港", "澳门"]
            for prov in provinces:
                if prov in loc_str:
                    return prov
            return loc_str[:3] if len(loc_str) >= 3 else loc_str

        def extract_city(loc):
            """提取城市名（不含省级前缀），如 '秦皇岛市'"""
            if pd.isna(loc):
                return ""
            loc_str = str(loc)
            # 直辖市直接返回市名
            for muni in ['北京市', '上海市', '天津市', '重庆市']:
                if loc_str.startswith(muni):
                    return muni
            rest = _strip_province_prefix(loc_str)
            if not rest:
                return ""
            # 自治州优先（因为有的自治州名下含"市"字，如地级市名）
            idx = rest.find('自治州')
            if idx >= 0:
                return rest[:idx + 3]
            # 市 / 地区 / 盟
            for term in ['市', '地区', '盟']:
                idx = rest.find(term)
                if idx >= 0:
                    return rest[:idx + len(term)]
            return ""

        def extract_district(loc):
            """提取区/县/旗名（不含省、市前缀），如 '海港区'"""
            if pd.isna(loc):
                return ""
            loc_str = str(loc)
            # 去掉省级前缀
            for muni in ['北京市', '上海市', '天津市', '重庆市']:
                if loc_str.startswith(muni):
                    rest = loc_str[len(muni):]
                    break
            else:
                rest = _strip_province_prefix(loc_str)
            # 再去掉地级市/自治州前缀
            for term in ['自治州', '地区', '市', '盟']:
                idx = rest.find(term)
                if idx >= 0:
                    rest = rest[idx + len(term):]
                    break
            if not rest:
                return ""
            # 提取区 / 县 / 旗 / 县级市
            for term in ['区', '县', '市', '旗']:
                idx = rest.find(term)
                if idx >= 0:
                    return rest[:idx + 1]
            return ""

        all_locations = df[col].dropna()
        provinces = sorted(list(set([extract_province(loc) for loc in all_locations if extract_province(loc)])))

        selected_provinces = st.sidebar.multiselect(
            "选择省份（可多选）",
            options=provinces,
            default=[],
            key="province_select"
        )

        if selected_provinces:
            def match_province(x):
                x_str = str(x)
                return any(prov in x_str for prov in selected_provinces)
            province_mask = filtered_df[col].apply(match_province)
            filtered_df = filtered_df[province_mask]
            selected_region = "、".join(selected_provinces)

        # ---- 城市选择 ----
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

        if selected_cities:
            def match_city(x):
                x_str = str(x)
                return any(city in x_str for city in selected_cities)
            city_mask = filtered_df[col].apply(match_city)
            filtered_df = filtered_df[city_mask]
            cities_str = "、".join(selected_cities)
            if selected_region:
                selected_region += " " + cities_str
            else:
                selected_region = cities_str

        # ---- 区县选择 ----
        if selected_cities:
            district_locations = filtered_df[col].dropna()
        elif selected_provinces:
            district_locations = filtered_df[col].dropna()
        else:
            district_locations = all_locations

        # 构建区县选项：未选城市时加城市前缀消除歧义（如 "石家庄市-长安区" vs "西安市-长安区"）
        raw_districts = [extract_district(loc) for loc in district_locations]
        raw_districts = [d for d in raw_districts if d]
        if selected_cities:
            # 已选城市 → 直接用区县名
            district_options = sorted(list(set(raw_districts)))
            district_label = {d: d for d in district_options}
        else:
            # 未选城市 → 附加上级城市名，防止同名区县混淆
            district_with_city = {}
            for loc in district_locations:
                d = extract_district(loc)
                c = extract_city(loc)
                if d and c:
                    key = f"{c}-{d}"
                    district_with_city[key] = (c, d)
            district_options = sorted(district_with_city.keys())
            district_label = {k: k for k in district_options}

        selected_districts = st.sidebar.multiselect(
            "选择县/区（可多选）",
            options=district_options,
            default=[],
            key="district_select"
        )

        if selected_districts:
            def match_district(x):
                x_str = str(x)
                for sel in selected_districts:
                    if selected_cities:
                        # 城市已选，区县名直接匹配 + 确认城市也匹配
                        if sel in x_str and any(c in x_str for c in selected_cities):
                            return True
                    elif selected_provinces:
                        # 有省无市，用 "市-区" 格式中的区县名匹配 + 确认省份匹配
                        if '-' in sel:
                            _, dist_name = sel.rsplit('-', 1)
                        else:
                            dist_name = sel
                        if dist_name in x_str and any(p in x_str for p in selected_provinces):
                            return True
                    else:
                        # 无省无市，用 "市-区" 格式中的城市和区县名同时匹配
                        if '-' in sel:
                            city_name, dist_name = sel.rsplit('-', 1)
                            if city_name in x_str and dist_name in x_str:
                                return True
                        else:
                            if sel in x_str:
                                return True
                return False

            district_mask = filtered_df[col].apply(match_district)
            filtered_df = filtered_df[district_mask]
            # 显示时去掉城市前缀
            display_districts = []
            for sel in selected_districts:
                display_districts.append(sel.rsplit('-', 1)[-1] if '-' in sel else sel)
            if selected_region:
                selected_region += " " + "、".join(display_districts)
            else:
                selected_region = "、".join(display_districts)

        st.sidebar.markdown("---")
    
    elif col in text_search_columns:
        # ---- 专业关键词搜索（可切换大类扩展） ----
        st.sidebar.markdown(f"### 🔍 {col}筛选")
        search_value = st.sidebar.text_input(
            f"输入{col}关键词",
            value="",
            placeholder=f"例如：计算机、会计、法学...",
            key=f"text_search_{col}",
            help=f"输入{col}名称中的关键词，支持模糊匹配"
        )
        expand_cats = st.sidebar.checkbox(
            "同时搜索专业大类",
            value=True,
            key=f"expand_cats_{col}",
            help="开启后，搜索具体专业名时也会匹配写了该专业所属大类的岗位"
        )
        if search_value:
            pre_search_df = filtered_df.copy()
            # 1) 直接子串匹配
            filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(search_value, case=False, na=False, regex=False)]
            selected_major = search_value

            # 2) 自动扩展到大类
            if expand_cats:
                major_map = build_major_category_map(df)
                expanded_cats = set()
                for spec_name, cats in major_map.items():
                    if search_value in spec_name:
                        expanded_cats.update(cats)
                if expanded_cats:
                    cat_match = pre_search_df[col].astype(str).apply(
                        lambda x: any(cat in x for cat in expanded_cats)
                    )
                    new_indices = cat_match[cat_match].index.difference(filtered_df.index)
                    if len(new_indices) > 0:
                        filtered_df = pd.concat([filtered_df, pre_search_df.loc[new_indices]])
                        st.sidebar.caption(f"🔗 扩展到：{'、'.join(sorted(expanded_cats))}（+{len(new_indices)} 个岗位）")

        # 显示常用专业供参考
        with st.sidebar.expander(f"📋 常用{col}参考"):
            common_majors = df[col].dropna().unique()
            common_majors = sorted([m for m in common_majors if len(str(m)) <= 20])[:30]
            st.caption("点击可复制到搜索框：")
            for m in common_majors:
                if st.button(str(m), key=f"major_chip_{m}"):
                    st.session_state[f"text_search_{col}"] = str(m)
                    st.rerun()
    
    else:
        # 基层工作最低年限等
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 50 and len(unique_vals) > 1:
            default_vals = []
            if col == "基层工作最低年限":
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
                if col == "基层工作最低年限":
                    selected_work_years = "、".join(selected)
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

# ==================== 显示与操作 ====================

if merge_option:
    st.header(f"📋 {selected_year}年岗位数据")
else:
    st.header(f"📋 {selected_year}年岗位数据 - {selected_sheet}")
st.info(f"🔍 筛选结果: {len(filtered_df)} 个岗位")

# ==================== 手动删除岗位模块 ====================
# 使用稳定唯一ID（职位代码+部门代码）替代整数索引，防止筛选变化后删错行
if 'deleted_ids' not in st.session_state:
    st.session_state.deleted_ids = set()

if 'current_year' not in st.session_state or st.session_state.current_year != selected_year:
    st.session_state.deleted_ids = set()
    st.session_state.current_year = selected_year

if 'current_sheet' not in st.session_state or st.session_state.current_sheet != selected_sheet:
    st.session_state.deleted_ids = set()
    st.session_state.current_sheet = selected_sheet

display_df = filtered_df.copy()
display_df = display_df.reset_index(drop=True)

# 生成唯一行ID（优先使用职位代码+部门代码，否则用行号）
def make_row_id(row):
    parts = []
    if '职位代码' in row.index and pd.notna(row['职位代码']):
        parts.append(str(row['职位代码']))
    if '部门代码' in row.index and pd.notna(row['部门代码']):
        parts.append(str(row['部门代码']))
    if parts:
        return '_'.join(parts)
    return None

display_df['_row_id'] = display_df.apply(make_row_id, axis=1)
# 对于无法生成唯一ID的行，用"行号:"前缀区分
null_id_mask = display_df['_row_id'].isna()
display_df.loc[null_id_mask, '_row_id'] = 'row:' + display_df.loc[null_id_mask].index.astype(str)

# 隐藏的列（加上内部ID列）
hide_columns = ['学历映射', '政治面貌映射', '专业要求数', '专业要求数_大专', '专业要求数_本科',
                '专业要求数_研究生', '专业要求数_博士', '机构层级映射', '备注限制数', '性别要求', '_row_id']
if '工作表' in display_df.columns:
    hide_columns.append('工作表')

# ==================== 分页控制 ====================
PAGE_SIZE_OPTIONS = [20, 50, 100, 200]
if 'page_size' not in st.session_state:
    st.session_state.page_size = 50
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

st.subheader("✂️ 手动删除岗位")
col_info, col_page, col_size = st.columns([2, 2, 1])
with col_info:
    st.caption("勾选要删除的岗位，然后点击'删除选中岗位'按钮")
with col_size:
    st.session_state.page_size = st.selectbox(
        "每页显示", PAGE_SIZE_OPTIONS,
        index=PAGE_SIZE_OPTIONS.index(st.session_state.page_size) if st.session_state.page_size in PAGE_SIZE_OPTIONS else 1,
        key="page_size_select",
        label_visibility="collapsed"
    )

# 计算分页
total_rows = len(display_df)
total_pages = max(1, (total_rows + st.session_state.page_size - 1) // st.session_state.page_size)
if st.session_state.current_page >= total_pages:
    st.session_state.current_page = 0

start_idx = st.session_state.current_page * st.session_state.page_size
end_idx = min(start_idx + st.session_state.page_size, total_rows)

# 准备要显示的列
display_columns = [col for col in display_df.columns if col not in hide_columns]
editor_df = display_df[display_columns].iloc[start_idx:end_idx].copy()
editor_df['选择删除'] = False

# 预勾选已删除的行
editor_df['选择删除'] = editor_df.index.map(lambda i: display_df.iloc[i]['_row_id'] in st.session_state.deleted_ids)

edited_df = st.data_editor(
    editor_df,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "选择删除": st.column_config.CheckboxColumn(
            "选择删除",
            help="勾选要删除的岗位",
            default=False,
        )
    },
    hide_index=True,
    key=f"editor_{start_idx}"
)

# 分页导航 + 操作按钮
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])
with nav_col1:
    if st.button("◀ 上一页", disabled=(st.session_state.current_page == 0)):
        st.session_state.current_page = max(0, st.session_state.current_page - 1)
        st.rerun()
with nav_col2:
    if st.button("下一页 ▶", disabled=(st.session_state.current_page >= total_pages - 1)):
        st.session_state.current_page = min(total_pages - 1, st.session_state.current_page + 1)
        st.rerun()
with nav_col3:
    st.caption(f"第 {st.session_state.current_page + 1}/{total_pages} 页，共 {total_rows} 条")
with nav_col4:
    delete_clicked = st.button("🗑️ 删除选中", type="primary")
    if delete_clicked:
        ids_to_delete = set()
        for _, row in edited_df.iterrows():
            if row['选择删除']:
                orig_idx = row.name  # editor_df 保留了 display_df 的 index
                ids_to_delete.add(display_df.iloc[orig_idx]['_row_id'])
        if ids_to_delete:
            st.session_state.deleted_ids.update(ids_to_delete)
            st.success(f"已删除 {len(ids_to_delete)} 个岗位！")
            st.rerun()
        else:
            st.warning("请先勾选要删除的岗位！")
with nav_col5:
    reset_delete = st.button("🔄 重置删除")
    if reset_delete:
        st.session_state.deleted_ids = set()
        st.session_state.current_page = 0
        st.rerun()

# 应用删除：用 row_id 过滤
final_df = display_df[~display_df['_row_id'].isin(st.session_state.deleted_ids)].copy()
final_df = final_df.drop(columns=['_row_id'])

# 计算推荐分数
score_details = None
weights_info = None
calculation_data = None
if len(final_df) > 0:
    final_df, score_details, weights_info, calculation_data = calculate_recommendation_scores(
        final_df, user_edu, user_pol, user_gender, selected_region, selected_major, selected_work_years
    )
    cols = ['排名', '推荐分'] + [col for col in final_df.columns if col not in ['排名', '推荐分']]
    final_df = final_df[cols]

st.markdown("---")
st.subheader(f"📊 最终结果（{len(final_df)} 个岗位）")
final_display_columns = [col for col in final_df.columns if col not in hide_columns]

# 分页显示最终结果
if len(final_df) > 0:
    result_page_size = st.session_state.page_size
    result_total_pages = max(1, (len(final_df) + result_page_size - 1) // result_page_size)
    if 'result_page' not in st.session_state:
        st.session_state.result_page = 0
    if st.session_state.result_page >= result_total_pages:
        st.session_state.result_page = 0

    r_start = st.session_state.result_page * result_page_size
    r_end = min(r_start + result_page_size, len(final_df))

    st.dataframe(final_df[final_display_columns].iloc[r_start:r_end], use_container_width=True)

    rnav1, rnav2, rnav3 = st.columns([1, 1, 3])
    with rnav1:
        if st.button("◀ 上一页", key="result_prev", disabled=(st.session_state.result_page == 0)):
            st.session_state.result_page = max(0, st.session_state.result_page - 1)
            st.rerun()
    with rnav2:
        if st.button("下一页 ▶", key="result_next", disabled=(st.session_state.result_page >= result_total_pages - 1)):
            st.session_state.result_page = min(result_total_pages - 1, st.session_state.result_page + 1)
            st.rerun()
    with rnav3:
        st.caption(f"第 {st.session_state.result_page + 1}/{result_total_pages} 页")
else:
    st.warning("没有符合条件的岗位！")

# 显示评分详情
if score_details is not None and weights_info is not None:
    st.markdown("---")
    tab1, tab2 = st.tabs(["📊 权重信息", "📋 各岗位评分详情"])
    
    with tab1:
        st.subheader("各指标CRITIC客观权重")
        weights_df = pd.DataFrame(weights_info)
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
        st.caption("💡 说明：")
        st.caption("- CRITIC权重：基于数据离散度和相关性的客观权重")
    
    with tab2:
        st.subheader("各岗位评分详情")
        detail_rows = []
        for detail in score_details:
            row = {
                '排名': detail['排名'],
                '推荐分': detail['推荐分']
            }
            for col in ['进面分数', '招考人数', '专业要求数', '机构层级', '学历匹配度', '备注限制数']:
                row[f'{col}_原始值'] = detail[f'{col}_原始值']
                row[f'{col}_得分'] = detail[f'{col}_得分']
                row[f'{col}_权重'] = detail[f'{col}_权重']
            detail_rows.append(row)

        detail_df = pd.DataFrame(detail_rows)
        detail_cols = ['排名', '推荐分']
        for col in ['进面分数', '招考人数', '专业要求数', '机构层级', '学历匹配度', '备注限制数']:
            detail_cols.extend([f'{col}_原始值', f'{col}_得分', f'{col}_权重'])
        detail_df = detail_df[detail_cols]

        # 分页显示评分详情
        d_start = st.session_state.result_page * result_page_size
        d_end = min(d_start + result_page_size, len(detail_df))
        st.dataframe(detail_df.iloc[d_start:d_end], use_container_width=True, hide_index=True)
        st.caption(f"显示第 {d_start + 1}-{d_end} 条（共 {len(detail_df)} 条）")
        st.caption("💡 说明：")
        st.caption("- 得分：0-10分，越高越好")
        st.caption("- 权重：该指标在总分中的占比（CRITIC客观权重）")
        st.caption("- 推荐分：通过TOPSIS计算得出")

# ==================== 导出结果 ====================
if len(final_df) > 0:
    from io import BytesIO

    # ---- 简洁导出：推荐岗位列表 (Excel) ----
    simple_output = BytesIO()
    with pd.ExcelWriter(simple_output, engine='openpyxl') as writer:
        final_df[final_display_columns].to_excel(writer, sheet_name='推荐岗位列表', index=False)

    simple_output.seek(0)
    simple_name = f"岗位推荐结果_{selected_year}.xlsx" if merge_option else f"岗位推荐结果_{selected_year}_{selected_sheet}.xlsx"

    st.download_button(
        label="📥 下载推荐结果（Excel）",
        data=simple_output,
        file_name=simple_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ---- CSV 导出 ----
    csv_data = final_df[final_display_columns].to_csv(index=False).encode('utf-8-sig')
    csv_name = f"岗位筛选结果_{selected_year}.csv" if merge_option else f"岗位筛选结果_{selected_year}_{selected_sheet}.csv"
    st.download_button(
        label="📥 下载结果（CSV）",
        data=csv_data,
        file_name=csv_name,
        mime='text/csv'
    )

    # ---- 高级导出：完整计算过程（折叠在 expander 中） ----
    if calculation_data is not None:
        with st.expander("📊 高级导出：完整 CRITIC-TOPSIS 计算数据", expanded=False):
            st.caption("包含所有计算中间步骤，适合论文/研究分析")
            all_cols = ['进面分数', '招考人数', '专业要求数', '机构层级', '学历匹配度', '备注限制数']

            full_output = BytesIO()
            with pd.ExcelWriter(full_output, engine='openpyxl') as writer:
                # 用户信息
                pd.DataFrame({
                    'Item': ['Gender', 'Education', 'Political', 'Region', 'Major', 'WorkYears', 'Positions'],
                    'Value': [calculation_data['user_info'][k] for k in ['Gender', 'Education', 'Political', 'Region', 'Major', 'WorkYears', 'Positions']]
                }).to_excel(writer, sheet_name='用户信息', index=False)

                # 推荐岗位列表
                final_df[final_display_columns].to_excel(writer, sheet_name='推荐岗位列表', index=False)

                # CRITIC权重
                pd.DataFrame({
                    'Indicator': all_cols,
                    'StdDev': [round(calculation_data['std_devs'].get(col, 0), 4) for col in all_cols],
                    '1-CorrSum': [round(calculation_data['corr_terms'].get(col, 0), 4) for col in all_cols],
                    'CRITIC': [round(calculation_data['critic_values'].get(col, 0), 4) for col in all_cols],
                    'Weight': [round(calculation_data['final_weights'].get(col, 0), 4) for col in all_cols],
                    'Weight%': [round(calculation_data['final_weights'].get(col, 0) * 100, 4) for col in all_cols]
                }).to_excel(writer, sheet_name='CRITIC权重计算', index=False)

                # 标准化指标
                norm_df = calculation_data['normalized'].copy().round(4)
                norm_df.insert(0, '排名', range(1, len(norm_df) + 1))
                norm_df.to_excel(writer, sheet_name='标准化指标', index=False)

                # TOPSIS
                pd.DataFrame({
                    '排名': range(1, len(final_df) + 1),
                    'D+': calculation_data['d_positive'].round(4),
                    'D-': calculation_data['d_negative'].round(4),
                    'C': calculation_data['closeness'].round(4),
                    '推荐分': final_df['推荐分'].values
                }).to_excel(writer, sheet_name='TOPSIS计算', index=False)

                # 正负理想解
                pd.DataFrame({
                    'Indicator': all_cols,
                    'PositiveIdeal': [round(calculation_data['positive_ideal'].get(col, 0), 4) for col in all_cols],
                    'NegativeIdeal': [round(calculation_data['negative_ideal'].get(col, 0), 4) for col in all_cols]
                }).to_excel(writer, sheet_name='正负理想解', index=False)

            full_output.seek(0)
            full_name = f"岗位推荐完整计算_{selected_year}.xlsx" if merge_option else f"岗位推荐完整计算_{selected_year}_{selected_sheet}.xlsx"

            st.download_button(
                label="📥 下载完整计算数据（含所有中间步骤）",
                data=full_output,
                file_name=full_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

st.markdown("---")
st.caption("💡 提示：在左侧边栏设置筛选条件和个人信息，系统会自动匹配并推荐最合适的岗位！")
