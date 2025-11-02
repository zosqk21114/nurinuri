# streamlit_app.py
# 실행: streamlit run streamlit_app.py
# 필요 패키지: streamlit, pandas, numpy, pydeck, openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime, timedelta, time as dtime
from io import BytesIO
from zoneinfo import ZoneInfo
import os, re

KST = ZoneInfo("Asia/Seoul")

# =============================
# 전역 UI (큰 글씨/큰 버튼)
# =============================
st.set_page_config(page_title="🧡 독거노인 지원 웹앱 (Prototype)", page_icon="🧡", layout="wide")
st.markdown("""
<style>
:root { --base-font: 20px; }
html, body, [class*="css"]  { font-size: var(--base-font); }
button, .stButton>button { font-size: 1.1rem !important; padding: 0.6rem 1.1rem !important; border-radius: 12px !important; }
input, select, textarea, .stTextInput>div>div>input { font-size: 1.05rem !important; }
thead tr th { font-size: 1.05rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("🧡 독거노인 지원 웹앱 (Prototype)")

# =============================
# 파일 경로 / 상수
# =============================
CHECKIN_CSV = "checkins.csv"
MEDS_CSV = "meds.csv"
MEDLOG_CSV = "med_log.csv"
INSTITUTIONS_CSV = "institutions.csv"      # 내부 표준 캐시
REGIONAL_CSV = "regional_factors.csv"      # 내부 표준 캐시
HOME_JSON = "home_location.json"           # 집 위치 저장

# 사용자가 미리 올린 원본 파일 경로(있으면 자동 반영)
USER_INST_CANDIDATES = [
    "/mnt/data/전국의료기관 표준데이터.csv",
    "전국의료기관 표준데이터.csv"
]
USER_REG_CANDIDATES = [
    "/mnt/data/독거노인가구비율_시도_시_군_구__20251029204458.xlsx",
    "독거노인가구비율_시도_시_군_구__20251029204458.xlsx"
]

# =============================
# 유틸
# =============================
def now_kst():
    return datetime.now(KST)

def load_csv(path, dtype=None, parse_dates=None):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, dtype=dtype, parse_dates=parse_dates)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_csv(df, path):
    if isinstance(df, pd.DataFrame):
        df.to_csv(path, index=False)

def try_read_first_exists(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def make_alarm_wav(seconds=2, freq=880, sr=16000):
    import wave, struct
    t = np.linspace(0, seconds, int(sr*seconds), False)
    tone = (0.5*np.sin(2*np.pi*freq*t)).astype(np.float32)
    buf = BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        for s in tone:
            w.writeframes(struct.pack('<h', int(s*32767)))
    buf.seek(0)
    return buf

ALARM_WAV = make_alarm_wav()

def parse_time_str(tstr):
    try:
        h, m = map(int, str(tstr).split(":"))
        return dtime(hour=h, minute=m)
    except Exception:
        return None

# --- 집 위치 저장/불러오기 ---
def load_home():
    import json
    if os.path.exists(HOME_JSON):
        try:
            with open(HOME_JSON, "r", encoding="utf-8") as f:
                return json.load(f)  # {"label": str, "lat": float, "lon": float}
        except Exception:
            return None
    return None

def save_home(lat: float, lon: float, label: str = "우리 집"):
    import json
    data = {"label": label, "lat": float(lat), "lon": float(lon)}
    with open(HOME_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data

def delete_home():
    try:
        os.remove(HOME_JSON)
    except FileNotFoundError:
        pass

# =============================
# 사용자 원본 -> 내부 표준 변환
# institutions: name,type,lat,lon,address,region_name
# regional: region_name, solo_ratio(0~1), accessibility_score(optional)
# =============================
def normalize_institutions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["name","type","lat","lon","address","region_name"])
    d = df.copy()
    colmap = {}
    for c in d.columns:
        lc = c.lower()
        if lc in ["요양기관명","기관명","name","inst_name","명칭"]: colmap[c] = "name"
        elif lc in ["종별코드명","종별코드","종별","유형","type","category"]: colmap[c] = "type"
        elif lc in ["위도","lat","latitude","y","좌표y","좌표_y"]: colmap[c] = "lat"
        elif lc in ["경도","lon","lng","longitude","x","좌표x","좌표_x"]: colmap[c] = "lon"
        elif any(k in lc for k in ["도로명주소","지번주소","주소","address"]): colmap[c] = "address"
        elif lc in ["시도명","시도","광역시도","시도코드명"]: colmap[c] = "sido"
        elif lc in ["시군구명","시군구","시군구코드명"]: colmap[c] = "sigungu"
    if colmap:
        d = d.rename(columns=colmap)

    for c in ["lat","lon"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if "type" not in d.columns: d["type"] = ""
    if "address" not in d.columns:
        if "sido" in d.columns or "sigungu" in d.columns:
            d["address"] = (d.get("sido","").astype(str).fillna("") + " " + d.get("sigungu","").astype(str).fillna("")).str.strip()
        else:
            d["address"] = ""

    if "sido" in d.columns:
        d["region_name"] = d["sido"].astype(str)
    else:
        def guess_sido(addr: str):
            if not isinstance(addr, str): return ""
            m = re.match(r"^(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)", addr)
            return m.group(0) if m else ""
        d["region_name"] = d["address"].astype(str).apply(guess_sido)

    def norm_type(t):
        t = str(t)
        if "약국" in t: return "약국"
        if any(k in t for k in ["병원","의원","한의원","치과"]): return "병원"
        return t if t else "기타"
    d["type"] = d["type"].apply(norm_type)

    d = d[pd.notna(d["lat"]) & pd.notna(d["lon"])]
    return d[["name","type","lat","lon","address","region_name"]].reset_index(drop=True)

def load_user_institutions():
    p = try_read_first_exists(USER_INST_CANDIDATES)
    if p:
        try:
            raw = pd.read_csv(p)
            return normalize_institutions(raw)
        except Exception as e:
            st.warning(f"전국의료기관 CSV 읽기 오류: {e}")
    return pd.DataFrame(columns=["name","type","lat","lon","address","region_name"])

def normalize_regional(df_or_path) -> pd.DataFrame:
    if df_or_path is None:
        return pd.DataFrame(columns=["region_name","solo_ratio","accessibility_score"])
    try:
        if isinstance(df_or_path, str):
            if df_or_path.lower().endswith(".xlsx"):
                d = pd.read_excel(df_or_path, engine="openpyxl")
            else:
                d = pd.read_csv(df_or_path)
        else:
            d = df_or_path.copy()
    except Exception as e:
        st.warning(f"지역 엑셀/CSV 읽기 오류: {e}")
        return pd.DataFrame(columns=["region_name","solo_ratio","accessibility_score"])

    if d is None or d.empty:
        return pd.DataFrame(columns=["region_name","solo_ratio","accessibility_score"])

    cols = [c for c in d.columns]
    region_col = None
    for c in cols:
        lc = str(c).lower()
        if any(k in lc for k in ["시군구","시·군·구","시군","시도","광역","행정구역","지역","지역명"]):
            region_col = c; break
    if region_col is None:
        if d.dtypes.iloc[0] == object: region_col = d.columns[0]
        else:
            d["region_name"] = "알수없음"; region_col = "region_name"

    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(d[c])]
    value_col = numeric_cols[0] if numeric_cols else None

    out = pd.DataFrame()
    out["region_name"] = d[region_col].astype(str)
    if value_col is not None:
        vals = d[value_col].astype(float)
        out["solo_ratio"] = vals/100.0 if vals.max() > 1.5 else vals
    else:
        out["solo_ratio"] = 0.0
    out = out.groupby("region_name", as_index=False)["solo_ratio"].mean()
    out["accessibility_score"] = np.nan
    return out

def load_user_regional():
    p = try_read_first_exists(USER_REG_CANDIDATES)
    if p: return normalize_regional(p)
    return pd.DataFrame(columns=["region_name","solo_ratio","accessibility_score"])

# =============================
# 체크인/복약/위험도
# =============================
def checkin_stats(df: pd.DataFrame, lookback_days=30):
    if df.empty:
        return {"missing_days": [], "z_outliers_idx": [], "mean_min": None, "std_min": None}
    df_recent = df[df["timestamp"] >= (now_kst() - timedelta(days=lookback_days))]
    if df_recent.empty:
        return {"missing_days": [], "z_outliers_idx": [], "mean_min": None, "std_min": None}
    daily = (df_recent
             .assign(date=lambda x: x["timestamp"].dt.date,
                     minutes=lambda x: x["timestamp"].dt.hour*60 + x["timestamp"].dt.minute)
             .sort_values("timestamp")
             .groupby("date", as_index=False).first())
    days = [(now_kst().date() - timedelta(days=i)) for i in range(lookback_days)]
    existing = set(daily["date"].tolist())
    missing = [d for d in days if d not in existing]
    if len(daily) >= 5:
        mins = daily["minutes"].to_numpy()
        mu = float(np.mean(mins))
        sd = float(np.std(mins)) if np.std(mins) > 0 else 1.0
        zscores = (mins - mu) / sd
        out_idx = list(np.where(np.abs(zscores) > 2)[0])
        return {"missing_days": missing, "z_outliers_idx": out_idx, "mean_min": mu, "std_min": sd, "daily": daily}
    return {"missing_days": missing, "z_outliers_idx": [], "mean_min": None, "std_min": None, "daily": daily}

def enumerate_due_times(start_clock: dtime, interval_hours: int, from_dt: datetime, to_dt: datetime):
    start_at = datetime.combine(from_dt.date(), start_clock, tzinfo=KST)
    while start_at > from_dt:
        start_at -= timedelta(hours=interval_hours)
    while start_at + timedelta(hours=interval_hours) < from_dt:
        start_at += timedelta(hours=interval_hours)
    times, cur = [], start_at
    while cur <= to_dt:
        if cur >= from_dt: times.append(cur)
        cur += timedelta(hours=interval_hours)
    return times

def estimate_adherence(meds_df, med_log_df, days=7, window_minutes=60):
    to_dt = now_kst(); from_dt = to_dt - timedelta(days=days)
    due_list = []
    taken_list = med_log_df[(med_log_df["taken_at"]>=from_dt) & (med_log_df["taken_at"]<=to_dt)].copy()
    for _, row in meds_df.iterrows():
        name = row["name"]; iv = int(row["interval_hours"]); sc = parse_time_str(str(row["start_time"]))
        if not sc: continue
        for d in enumerate_due_times(sc, iv, from_dt, to_dt):
            due_list.append({"name": name, "due_time": d})
    due_df = pd.DataFrame(due_list)
    if due_df.empty: return 0, 0
    taken_on_time, window = 0, timedelta(minutes=window_minutes)
    for _, due in due_df.iterrows():
        name = due["name"]; dtime_ = due["due_time"]
        cand = taken_list[(taken_list["name"]==name) & (taken_list["taken_at"].between(dtime_-window, dtime_+window))]
        if len(cand):
            taken_on_time += 1
            taken_list = taken_list.drop(cand.index[0])
    return len(due_df), taken_on_time

def due_now_list(meds_df, within_minutes=15, overdue_minutes=90):
    now = now_kst(); due_items = []
    for _, row in meds_df.iterrows():
        name = row["name"]; iv = int(row["interval_hours"]); sc = parse_time_str(str(row["start_time"]))
        if not sc: continue
        dues = enumerate_due_times(sc, iv, now - timedelta(days=2), now + timedelta(days=1))
        if dues:
            closest = min(dues, key=lambda d: abs((d - now).total_seconds()))
            diff_min = (closest - now).total_seconds()/60.0
            status = "due" if abs(diff_min)<=within_minutes else ("overdue" if diff_min<0 and abs(diff_min)<=overdue_minutes else None)
            if status: due_items.append({"name": name, "due_time": closest, "status": status})
    return due_items

def risk_score(checkins_df, med_log_df, meds_df):
    cs = checkin_stats(checkins_df, lookback_days=14)
    missing_last3 = [d for d in cs.get("missing_days", []) if (now_kst().date() - d).days <= 3]
    n_missing3 = len(missing_last3); n_out7 = 0
    if "daily" in cs and len(cs["daily"])>0 and cs.get("mean_min") is not None and cs.get("std_min",0)>0:
        last7 = cs["daily"][cs["daily"]["date"] >= (now_kst().date()-timedelta(days=7))]
        if len(last7) >= 5:
            mins = last7["minutes"].to_numpy()
            z = (mins - cs["mean_min"]) / cs["std_min"]
            n_out7 = int(np.sum(np.abs(z)>2))
    adherence = 1.0
    if not meds_df.empty:
        due_total, taken_on_time = estimate_adherence(meds_df, med_log_df, days=7, window_minutes=60)
        adherence = (taken_on_time / due_total) if due_total>0 else 1.0
    score = min(n_missing3, 3)/3*40 + min(n_out7, 5)/5*20 + (1.0 - adherence)*40
    return round(max(0, min(100, score)), 1), {
        "missing_last3": n_missing3, "outliers_last7": n_out7, "adherence_7d": round(adherence*100,1)
    }

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2-lat1); dlambda = np.radians(lon2-lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

# =============================
# 기본 데이터 로드 + 5번 자료 자동 반영
# =============================
checkins = load_csv(CHECKIN_CSV, parse_dates=["timestamp"])
if checkins.empty:
    checkins = pd.DataFrame(columns=["timestamp"]); save_csv(checkins, CHECKIN_CSV)

meds = load_csv(MEDS_CSV)
if meds.empty:
    meds = pd.DataFrame(columns=["name","interval_hours","start_time","notes"]); save_csv(meds, MEDS_CSV)

med_log = load_csv(MEDLOG_CSV, parse_dates=["taken_at"])
if med_log.empty:
    med_log = pd.DataFrame(columns=["name","due_time","taken_at"]); save_csv(med_log, MEDLOG_CSV)

_user_inst = load_user_institutions()
if len(_user_inst): save_csv(_user_inst, INSTITUTIONS_CSV)
_user_reg = load_user_regional()
if len(_user_reg): save_csv(_user_reg, REGIONAL_CSV)

institutions = load_csv(INSTITUTIONS_CSV)
regional = load_csv(REGIONAL_CSV)

# =============================
# 탭
# =============================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["① 체크인", "② 위험도/119 시나리오", "③ 복약 스케줄러", "④ 주변 의료기관 찾기", "⑤ 데이터/설정"])

# ① 체크인
with tab1:
    st.header("① 매일 체크인 (일어남 버튼)")
    c1, c2 = st.columns([1,2])
    with c1:
        if st.button("🌞 일어남(체크인)", use_container_width=True):
            checkins = pd.concat([checkins, pd.DataFrame([{"timestamp": now_kst()}])], ignore_index=True)
            save_csv(checkins, CHECKIN_CSV)
            st.success(f"체크인 완료: {now_kst().strftime('%Y-%m-%d %H:%M:%S')}")
    with c2:
        st.info("체크인은 하루 1회 이상 권장합니다. 아래 표에서 기록을 확인하세요.")

    if not checkins.empty:
        st.subheader("최근 체크인 기록")
        st.dataframe(checkins.sort_values("timestamp", ascending=False).head(50), use_container_width=True)
        df_plot = (checkins.assign(date=lambda x: x["timestamp"].dt.date,
                                   minutes=lambda x: x["timestamp"].dt.hour*60 + x["timestamp"].dt.minute)
                           .groupby("date", as_index=False)["minutes"].min()
                           .sort_values("date"))
        st.caption("날짜별 첫 체크인 시각(분)")
        st.line_chart(df_plot.set_index("date")["minutes"])
        cs = checkin_stats(checkins, lookback_days=30)
        st.markdown("**최근 30일 결측일**")
        if cs.get("missing_days"):
            st.warning(", ".join(sorted([d.strftime("%Y-%m-%d") for d in cs["missing_days"]])))
        else:
            st.success("결측 없음")
        st.markdown("**이상치(|z|>2) 의심(일별 첫 체크인 기준)**")
        daily = cs.get("daily", pd.DataFrame()); out_idx = cs.get("z_outliers_idx", [])
        if len(out_idx) and len(daily)>0:
            st.error(daily.iloc[out_idx])
        else:
            st.success("이상치 없음")

# ② 위험도 / 119
with tab2:
    st.header("② 위험도 예측 및 자동 알림(시뮬레이션)")
    thr, info = st.columns([1,3])
    with thr:
        risk_thr = st.slider("119/보호자 연락(가상) 발동 기준(%)", 10, 100, 60, 5)
    with info:
        st.info("실제 전화 발신은 하지 않으며, 임계치 초과 시 경보음과 시나리오 안내를 표시합니다.")

    score, detail = risk_score(checkins, med_log, meds)
    st.subheader(f"현재 위험도: {score}%")
    st.progress(min(1.0, score/100.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("최근 3일 결측(일)", detail["missing_last3"])
    c2.metric("최근 7일 이상치(일)", detail["outliers_last7"])
    c3.metric("복약 준수(7일)", f"{detail['adherence_7d']}%")

    if score >= risk_thr:
        st.error("⚠️ 위험도 임계치 초과! (가상 경보/연락 시나리오)")
        st.audio(ALARM_WAV)
        st.markdown("""
**시뮬레이션: 자동 연락 절차**
1) 보호자 1차 연락 시도  
2) 미응답 시 119 연계 안내 음성 송출  
3) 위치/최근 체크인/복약정보 요약 전송(가상)
""")
    else:
        st.success("현재는 임계치 미만입니다.")

# ③ 복약
with tab3:
    st.header("③ 복약 스케줄러 / 리마인더")
    st.caption("앱이 열려 있을 때에만 리마인더가 화면에 표시됩니다(프로토타입 한계).")

    with st.form("add_med", clear_on_submit=True):
        st.subheader("약 추가")
        cx, cy, cz = st.columns([2,1,2])
        name = cx.text_input("약 이름", placeholder="예: 고혈압약A")
        interval = cy.number_input("복용 간격(시간)", 4, 48, 12, 1)
        start_t = cz.text_input("첫 복용 시각(HH:MM)", "08:00")
        notes = st.text_input("메모(선택)", "")
        submit = st.form_submit_button("추가")
        if submit and name and parse_time_str(start_t):
            meds = pd.concat([meds, pd.DataFrame([{
                "name": name, "interval_hours": int(interval), "start_time": start_t, "notes": notes
            }])], ignore_index=True)
            save_csv(meds, MEDS_CSV)
            st.success(f"추가됨: {name} / {interval}시간 간격 / 시작 {start_t}")
        elif submit:
            st.error("입력을 확인하세요. (시각 형식 HH:MM)")

    if len(meds):
        st.subheader("등록된 약")
        st.dataframe(meds, use_container_width=True)
    else:
        st.info("등록된 약이 없습니다.")

    if len(meds):
        st.subheader("리마인더")
        due_items = due_now_list(meds, within_minutes=15, overdue_minutes=90)
        if due_items:
            for item in due_items:
                name = item["name"]; due = item["due_time"].strftime("%Y-%m-%d %H:%M")
                status = "🕒 곧 복약" if item["status"]=="due" else "⏰ 연체"
                st.warning(f"{status}: {name} / 예정시각 {due}")
                b1, b2, _ = st.columns([1,1,3])
                with b1:
                    if st.button(f"✅ {name} 복용 기록", key=f"take_{name}_{due}"):
                        med_log = pd.concat([med_log, pd.DataFrame([{
                            "name": name, "due_time": item["due_time"], "taken_at": now_kst()
                        }])], ignore_index=True)
                        save_csv(med_log, MEDLOG_CSV)
                        st.success(f"{name} 복용 기록 완료")
                with b2:
                    st.audio(ALARM_WAV)
        else:
            st.success("현재 15분 이내 예정/연체 항목 없음")

    if len(meds):
        total7, ok7 = estimate_adherence(meds, med_log, days=7, window_minutes=60)
        if total7>0:
            st.metric("최근 7일 준수율", f"{round(ok7/total7*100,1)}% ({ok7}/{total7})")
        else:
            st.info("최근 7일 예정 스케줄 없음")

    if len(med_log):
        st.subheader("복용 기록")
        st.dataframe(med_log.sort_values("taken_at", ascending=False).head(100), use_container_width=True)

# ④ 주변 의료기관(집 위치 저장/사용)
with tab4:
    st.header("④ 주변 약국/병원 찾기 및 추천")
    st.caption("※ 5번 탭 자료(전국의료기관 표준데이터, 독거노인가구 비율)를 자동 반영. 필요 시 아래에서 교체 업로드 가능.")

    up1, up2 = st.columns(2)
    with up1:
        inst_file = st.file_uploader("전국 의료기관 표준데이터 CSV 업로드", type=["csv"])
        if inst_file is not None:
            institutions = normalize_institutions(pd.read_csv(inst_file))
            save_csv(institutions, INSTITUTIONS_CSV)
            st.success(f"업로드 완료: {len(institutions)}개 기관")
    with up2:
        reg_file = st.file_uploader("독거노인가구 비율 파일 업로드 (xlsx/csv)", type=["xlsx","csv"])
        if reg_file is not None:
            regional = normalize_regional(reg_file)
            save_csv(regional, REGIONAL_CSV)
            st.success(f"업로드 완료: 시도 단위 {len(regional)}개")

    if institutions.empty:
        st.info("의료기관 데이터가 없습니다. 5번 자료가 감지되지 않으면 여기서 업로드하세요.")
    else:
        left, right = st.columns([2,1])
        with left:
            tsel = st.selectbox("기관 유형", ["약국","병원","전체"], index=0)
        with right:
            radius_km = st.slider("검색 반경(km)", 1, 20, 3)

        # === 집 위치 ===
        st.subheader("내 위치(위도/경도)")
        home = load_home()
        use_home = st.checkbox("저장된 집 위치 사용", value=(home is not None))

        if use_home and home is not None:
            st.success(f"집 위치: {home['label']} (lat: {home['lat']:.6f}, lon: {home['lon']:.6f})")
            lat = float(home["lat"]); lon = float(home["lon"])
            cA, cB, cC = st.columns([1,1,2])
            with cA:
                if st.button("집 위치로 검색", use_container_width=True):
                    st.toast("집 위치를 기준으로 검색합니다.", icon="🏠")
            with cB:
                if st.button("집 위치 삭제", use_container_width=True):
                    delete_home()
                    st.experimental_rerun()
            with cC:
                st.caption("집 위치는 로컬 파일(home_location.json)에 저장됩니다.")
        else:
            lat = st.number_input("위도(lat)", value=37.5665, format="%.6f")
            lon = st.number_input("경도(lon)", value=126.9780, format="%.6f")
            with st.expander("➕ 이 위치를 '집'으로 저장"):
                home_label = st.text_input("표시 이름", value="우리 집")
                if st.button("이 위치를 집으로 저장", use_container_width=True):
                    save_home(lat, lon, home_label)
                    st.success(f"저장 완료: {home_label} (lat: {lat:.6f}, lon: {lon:.6f})")
                    st.experimental_rerun()

        # 필터/거리계산
        df = institutions.copy()
        if tsel != "전체": df = df[df["type"]==tsel]

        if {"lat","lon"}.issubset(df.columns) and len(df):
            df["distance_km"] = haversine_km(lat, lon, df["lat"].astype(float), df["lon"].astype(float))
            df = df[df["distance_km"]<=radius_km].sort_values("distance_km").reset_index(drop=True)

            # 지역 취약도 결합(시도명 기준)
            if not regional.empty and "region_name" in df.columns:
                r = regional.copy()
                if "solo_ratio" in r.columns:
                    r["solo_ratio_norm"] = r["solo_ratio"].astype(float).clip(0,1)
                if "accessibility_score" in r.columns and r["accessibility_score"].notna().any():
                    vals = r["accessibility_score"].astype(float)
                    r["accessibility_score_norm"] = 1.0 - (vals - vals.min())/(vals.max()-vals.min()+1e-9)
                r["regional_need"] = 0.0
                if "solo_ratio_norm" in r.columns: r["regional_need"] += 0.6*r["solo_ratio_norm"]
                if "accessibility_score_norm" in r.columns: r["regional_need"] += 0.4*r["accessibility_score_norm"]
                rr = r[["region_name","regional_need"]].drop_duplicates()
                df = df.merge(rr, on="region_name", how="left")
            else:
                df["regional_need"] = np.nan

            # 최종 추천 점수
            if len(df):
                df["proximity"] = 1.0 - (df["distance_km"] / (radius_km+1e-9))
                df["proximity"] = df["proximity"].clip(0,1)
                if df["regional_need"].notna().any():
                    df["rec_score"] = 0.6*df["proximity"] + 0.4*df["regional_need"].fillna(df["regional_need"].median())
                else:
                    df["rec_score"] = df["proximity"]

                # 지도(집 마커 + 기관 레이어)
                layers = []
                # 집 마커
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=pd.DataFrame([{"name":"집","lat":lat,"lon":lon}]),
                    get_position='[lon, lat]',
                    get_radius=80,
                    pickable=True,
                    get_fill_color=[255, 0, 0, 200],
                ))
                # 기관 마커
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position='[lon, lat]',
                    get_radius=50,
                    pickable=True,
                    get_fill_color=[0, 128, 255, 160],
                ))
                view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=13)
                tooltip = {"text": "{name}\n거리: {distance_km}km\n추천점수: {rec_score}"}
                st.subheader("지도")
                st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip))

                st.subheader("가까운 순 추천 리스트")
                show_cols = [c for c in ["name","type","address","region_name","distance_km","rec_score"] if c in df.columns]
                st.dataframe(df[show_cols].head(50), use_container_width=True)
            else:
                st.info("반경 내 결과 없음.")
        else:
            st.error("의료기관 데이터에 lat/lon(위도/경도) 컬럼이 필요합니다.")

# ⑤ 데이터/설정 (자료 관리)
with tab5:
    st.header("⑤ 데이터/설정 (자료 관리)")
    st.markdown("5번 탭 자료(전국의료기관 표준데이터, 독거노인가구 비율)를 자동 인식하여 표준 포맷으로 변환·캐시합니다.")
    st.markdown("- **의료기관 표준 CSV**: `institutions.csv`  \n- **지역요인 표준 CSV**: `regional_factors.csv`")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("체크인 CSV", data=checkins.to_csv(index=False).encode("utf-8"), file_name="checkins.csv")
    with c2:
        st.download_button("약 목록 CSV", data=meds.to_csv(index=False).encode("utf-8"), file_name="meds.csv")
    with c3:
        st.download_button("복약 기록 CSV", data=med_log.to_csv(index=False).encode("utf-8"), file_name="med_log.csv")
    with c4:
        if not institutions.empty:
            st.download_button("의료기관 CSV", data=institutions.to_csv(index=False).encode("utf-8"), file_name="institutions.csv")
        else:
            st.write("의료기관 CSV: (없음)")

    st.markdown("#### 자동 로드 상태 미리보기")
    ic, rc = st.columns(2)
    with ic:
        if len(_user_inst):
            st.success(f"전국의료기관 원본 감지됨 ✅  (행 {len(_user_inst)}) → 표준 변환 저장 완료")
        else:
            st.info("전국의료기관 원본 미감지. 탭4에서 업로드 가능.")
        if not institutions.empty:
            st.dataframe(institutions.head(10), use_container_width=True)
    with rc:
        if len(_user_reg):
            st.success(f"독거노인가구 비율 원본 감지됨 ✅  (행 {len(_user_reg)}) → 표준 변환 저장 완료")
        else:
            st.info("독거노인가구 비율 원본 미감지. 탭4에서 업로드 가능.")
        if not regional.empty:
            st.dataframe(regional.head(10), use_container_width=True)

    st.markdown("#### 위험도 계산식(요약)")
    st.code("""
# score = 0
# score += min(n_missing3, 3) / 3 * 40      # 최근 3일 결측
# score += min(n_out7, 5) / 5 * 20          # 최근 7일 이상치(체크인 시각)
# score += (1.0 - adherence) * 40           # 7일 복약 준수율 역가중
# => 0~100 점수
""", language="python")

# =============================
# 상태 저장
# =============================
save_csv(checkins, CHECKIN_CSV)
save_csv(meds, MEDS_CSV)
save_csv(med_log, MEDLOG_CSV)
if len(institutions): save_csv(institutions, INSTITUTIONS_CSV)
if len(regional): save_csv(regional, REGIONAL_CSV)
