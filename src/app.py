import streamlit as st
import pandas as pd
import numpy as np
import os
import json

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
        
        # 提取市
        def extract_city(loc):
            if pd.isna(loc):
                return ""
            loc_str = str(loc)
            if "市" in loc_str:
                city_end = loc_str.find("市") + 1
                return loc_str[:city_end]
            return ""
        
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
            if selected_region:
                selected_region += " " + "、".join(selected_cities)
            else:
                selected_region = "、".join(selected_cities)
        
        # 提取区县
        def extract_district(loc):
            if pd.isna(loc):
                return ""
            loc_str = str(loc)
            if "区" in loc_str:
                district_end = loc_str.find("区") + 1
                return loc_str[:district_end]
            elif "县" in loc_str:
                district_end = loc_str.find("县") + 1
                return loc_str[:district_end]
            return ""
        
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
        
        if selected_districts:
            def match_district(x):
                x_str = str(x)
                return any(dist in x_str for dist in selected_districts)
            district_mask = filtered_df[col].apply(match_district)
            filtered_df = filtered_df[district_mask]
            if selected_region:
                selected_region += " " + "、".join(selected_districts)
            else:
                selected_region = "、".join(selected_districts)
        
        st.sidebar.markdown("---")
    
    elif col in text_search_columns:
        # 专业关键词搜索
        search_value = st.sidebar.text_input(
            f"{col}（关键词搜索）",
            value="",
            key=f"text_search_{col}"
        )
        if search_value:
            filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(search_value, case=False, na=False)]
            selected_major = search_value
    
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

# 手动删除岗位模块
if 'deleted_rows' not in st.session_state:
    st.session_state.deleted_rows = set()

if 'current_year' not in st.session_state or st.session_state.current_year != selected_year:
    st.session_state.deleted_rows = set()
    st.session_state.current_year = selected_year

if 'current_sheet' not in st.session_state or st.session_state.current_sheet != selected_sheet:
    st.session_state.deleted_rows = set()
    st.session_state.current_sheet = selected_sheet

display_df = filtered_df.copy()
display_df = display_df.reset_index(drop=True)

# 隐藏的列
hide_columns = ['学历映射', '政治面貌映射', '专业要求数', '专业要求数_大专', '专业要求数_本科', '专业要求数_研究生', '专业要求数_博士', '机构层级映射', '备注限制数', '性别要求']
if '工作表' in display_df.columns:
    hide_columns.append('工作表')

st.subheader("✂️ 手动删除岗位")
st.caption("勾选要删除的岗位，然后点击'删除选中岗位'按钮")

# 准备要显示的列
display_columns = [col for col in display_df.columns if col not in hide_columns]
# 添加选择删除列
editor_df = display_df[display_columns].copy()
editor_df['选择删除'] = False

edited_df = st.data_editor(
    editor_df,
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

col1, col2 = st.columns([1, 3])
with col1:
    delete_clicked = st.button("🗑️ 删除选中岗位", type="primary")
    if delete_clicked:
        rows_to_delete = edited_df[edited_df['选择删除']].index.tolist()
        if rows_to_delete:
            for idx in rows_to_delete:
                st.session_state.deleted_rows.add(idx)
            st.success(f"已删除 {len(rows_to_delete)} 个岗位！")
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        else:
            st.warning("请先勾选要删除的岗位！")

# 最终数据
final_df = display_df[~display_df.index.isin(st.session_state.deleted_rows)]

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

# 直接显示全部数据
st.dataframe(final_df[final_display_columns], use_container_width=True)

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
        
        # 直接显示全部评分详情
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        st.caption("💡 说明：")
        st.caption("- 得分：0-10分，越高越好")
        st.caption("- 权重：该指标在总分中的占比（CRITIC客观权重）")
        st.caption("- 推荐分：通过TOPSIS计算得出")

# 导出结果
if len(final_df) > 0 and calculation_data is not None:
    # 生成详细的Excel文件
    import io
    from io import BytesIO
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. 用户信息
        user_info_df = pd.DataFrame({
            'Item': ['Gender', 'Education', 'Political', 'Region', 'Major', 'WorkYears', 'Positions'],
            'Value': [
                calculation_data['user_info']['Gender'],
                calculation_data['user_info']['Education'],
                calculation_data['user_info']['Political'],
                calculation_data['user_info']['Region'],
                calculation_data['user_info']['Major'],
                calculation_data['user_info']['WorkYears'],
                calculation_data['user_info']['Positions']
            ]
        })
        user_info_df.to_excel(writer, sheet_name='用户信息', index=False)
        
        # 2. 推荐岗位列表
        position_list_df = final_df[final_display_columns].copy()
        position_list_df.to_excel(writer, sheet_name='推荐岗位列表', index=False)
        
        # 3. CRITIC权重计算
        all_cols = ['进面分数', '招考人数', '专业要求数', '机构层级', '学历匹配度', '备注限制数']
        critic_df = pd.DataFrame({
            'Indicator': all_cols,
            'StdDev': [calculation_data['std_devs'].get(col, 0) for col in all_cols],
            '1-CorrSum': [calculation_data['corr_terms'].get(col, 0) for col in all_cols],
            'CRITIC': [calculation_data['critic_values'].get(col, 0) for col in all_cols],
            'Weight': [calculation_data['final_weights'].get(col, 0) for col in all_cols],
            'Weight%': [round(calculation_data['final_weights'].get(col, 0) * 100, 2) for col in all_cols]
        })
        critic_df.to_excel(writer, sheet_name='CRITIC权重计算', index=False)
        
        # 4. 原始指标
        raw_df = calculation_data['raw_indicators'].copy()
        raw_df.insert(0, '排名', range(1, len(raw_df) + 1))
        raw_df.to_excel(writer, sheet_name='原始指标', index=False)
        
        # 5. 标准化指标
        norm_df = calculation_data['normalized'].copy()
        norm_df.insert(0, '排名', range(1, len(norm_df) + 1))
        norm_df.to_excel(writer, sheet_name='标准化指标', index=False)
        
        # 6. 加权指标
        weighted_df = calculation_data['weighted'].copy()
        weighted_df.insert(0, '排名', range(1, len(weighted_df) + 1))
        weighted_df.to_excel(writer, sheet_name='加权指标', index=False)
        
        # 7. TOPSIS计算
        topsis_df = pd.DataFrame({
            '排名': range(1, len(final_df) + 1),
            'D+': calculation_data['d_positive'],
            'D-': calculation_data['d_negative'],
            'C': calculation_data['closeness'],
            '推荐分': final_df['推荐分'].values
        })
        topsis_df.to_excel(writer, sheet_name='TOPSIS计算', index=False)
        
        # 8. 正负理想解
        ideal_df = pd.DataFrame({
            'Indicator': all_cols,
            'PositiveIdeal': [calculation_data['positive_ideal'].get(col, 0) for col in all_cols],
            'NegativeIdeal': [calculation_data['negative_ideal'].get(col, 0) for col in all_cols]
        })
        ideal_df.to_excel(writer, sheet_name='正负理想解', index=False)
    
    output.seek(0)
    
    if merge_option:
        excel_file_name = f"岗位推荐详细计算_{selected_year}.xlsx"
    else:
        excel_file_name = f"岗位推荐详细计算_{selected_year}_{selected_sheet}.xlsx"
    
    st.download_button(
        label="📥 下载详细计算结果（Excel）",
        data=output,
        file_name=excel_file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# 同时保留简单的CSV导出
csv = final_df[final_display_columns].to_csv(index=False).encode('utf-8-sig')
if merge_option:
    file_name = f"岗位筛选结果_{selected_year}.csv"
else:
    file_name = f"岗位筛选结果_{selected_year}_{selected_sheet}.csv"
st.download_button(
    label="📥 下载最终结果（CSV）",
    data=csv,
    file_name=file_name,
    mime='text/csv'
)

# 重置删除
reset_clicked = st.button("🔄 重置所有删除")
if reset_clicked:
    st.session_state.deleted_rows = set()
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

st.markdown("---")
st.caption("💡 提示：在左侧边栏选择年份、工作表和筛选条件，然后可以手动勾选删除不符合要求的岗位！")
