# -*- coding: utf-8 -*-
"""
스마트폰 이용행태 조사(2025) 데이터 분석 웹앱
- 실행: streamlit run main.py
- 업로드한 CSV의 실제 열 이름을 읽어 '이용 시간 구간별 비율 열', '평균 이용시간 열' 등을 자동으로 찾아 사용합니다.
"""

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="스마트폰 이용행태 조사 분석", layout="wide")

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------
ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
TOTAL_WORDS = {"총인원", "전체", "합계", "총계", "계", "전체합계", "total", "all"}
NONE_OPT = "(없음)"

# 분석용으로 통일한 열 이름
C_DIM = "구분"          # 대분류 (예: 연령, 성별)
C_GRP = "집단"          # 세부 집단 (예: 10대, 20대)
C_N = "사례수"
C_MEAN = "평균 이용시간(분)"
C_RATE = "이용률"
C_TOTAL = "집계행"

DOWNLOAD_NAME = "스마트폰_이용행태_조사_2025_전처리결과.csv"
DEFAULT_CSV = "스마트폰_이용행태_조사_2025_전처리완료.csv"  # main.py와 같은 폴더에 둘 기본 데이터 파일


def _version():
    try:
        p = st.__version__.split(".")
        return int(p[0]), int(p[1])
    except Exception:
        return (0, 0)


NEW_API = _version() >= (1, 50)


# ------------------------------------------------------------------
# 기본 데이터 파일 찾기
# ------------------------------------------------------------------
def find_default_csv():
    """main.py와 같은 폴더의 기본 CSV를 찾는다. 이름이 조금 달라도 폴더 안의 CSV를 찾아 쓴다."""
    try:
        folders = [Path(__file__).resolve().parent, Path.cwd()]
    except NameError:
        folders = [Path.cwd()]
    for folder in folders:
        p = folder / DEFAULT_CSV
        if p.exists():
            return p
    for folder in folders:
        cands = sorted(folder.glob("*.csv"))
        if cands:
            return cands[0]
    return None


# ------------------------------------------------------------------
# 화면 출력 도우미 (Streamlit 버전 차이 대응)
# ------------------------------------------------------------------
def show_chart(fig, key=None):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(size=14),
    )
    if NEW_API:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(df):
    if NEW_API:
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def fmt_min(m):
    """분 → '2시간 10분' 형태"""
    if m is None or pd.isna(m):
        return "-"
    total = int(round(float(m)))
    h, mm = divmod(total, 60)
    return f"{mm}분" if h == 0 else f"{h}시간 {mm}분"


def korean_dtype(s):
    if pd.api.types.is_bool_dtype(s):
        return "참/거짓"
    if pd.api.types.is_integer_dtype(s):
        return "숫자(정수)"
    if pd.api.types.is_numeric_dtype(s):
        return "숫자(실수)"
    return "문자"


def josa(word, with_batchim, without_batchim):
    """받침 유무에 따라 조사를 붙인다. 예) josa('20대', '이', '가') → '20대가'"""
    w = str(word).strip()
    if w:
        ch = w[-1]
        if "가" <= ch <= "힣":
            return w + (with_batchim if (ord(ch) - 0xAC00) % 28 else without_batchim)
    return w + without_batchim


def josa_ro(word):
    w = str(word).strip()
    if w and "가" <= w[-1] <= "힣":
        b = (ord(w[-1]) - 0xAC00) % 28
        return w + ("로" if b in (0, 8) else "으로")
    return w + "로"


def strength_word(r):
    a = abs(r)
    if a >= 0.8:
        return "매우 강한"
    if a >= 0.6:
        return "강한"
    if a >= 0.4:
        return "뚜렷한"
    if a >= 0.2:
        return "약한"
    return "거의 없는"


# ------------------------------------------------------------------
# 파일 읽기 · 전처리
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes):
    """한글 인코딩(UTF-8, UTF-8 BOM, CP949, EUC-KR)을 차례로 시도해 CSV를 읽는다."""
    last_error = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            if df.shape[1] == 1:  # 구분자가 쉼표가 아닌 경우 대비
                df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
            return df, enc
        except UnicodeDecodeError as e:
            last_error = e
        except pd.errors.EmptyDataError:
            raise
        except Exception as e:  # noqa: BLE001
            last_error = e
    raise ValueError(f"파일을 읽을 수 없습니다. ({last_error})")


def _clean_text(v):
    if isinstance(v, str):
        v2 = re.sub(r"\s+", " ", v.replace("\ufeff", "")).strip()
        return np.nan if v2 == "" else v2
    return v


def basic_clean(raw: pd.DataFrame):
    """열 이름 정리, 문자열 공백 정리, 숫자형 변환."""
    df = raw.copy()

    names, seen = [], {}
    for c in df.columns:
        name = re.sub(r"\s+", " ", str(c).replace("\ufeff", "")).strip()
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    df.columns = names

    stripped_cells = 0
    converted_cols = []
    unconvertible = 0

    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            continue
        cleaned = pd.Series([_clean_text(v) for v in s], index=s.index, dtype=object)
        stripped_cells += sum(1 for a, b in zip(s, cleaned) if isinstance(a, str) and a != b)

        nn = cleaned.notna()
        if nn.sum() > 0:
            t = (
                cleaned[nn]
                .astype(str)
                .str.replace(r"[,%\s]", "", regex=True)
                .str.replace("−", "-", regex=False)
            )
            conv = pd.to_numeric(t, errors="coerce")
            if conv.notna().sum() >= 0.8 * nn.sum():
                full = pd.Series(np.nan, index=s.index, dtype=float)
                full[nn] = conv.values
                unconvertible += int(conv.isna().sum())
                df[c] = full
                converted_cols.append(c)
                continue
        df[c] = cleaned

    info = {
        "stripped_cells": stripped_cells,
        "converted_cols": converted_cols,
        "unconvertible": unconvertible,
        "missing_total": int(df.isna().sum().sum()),
    }
    return df, info


def parse_hours(name):
    """열 이름에서 이용 시간 구간(시간 단위)을 읽는다. 예) '1-2시간 미만' → (1, 2)"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*시간", name)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*시간\s*미만", name)
    if m:
        return 0.0, float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*시간\s*이상", name)
    if m:
        n = float(m.group(1))
        return n, n + 1.0
    return None


def short_label(name):
    s = re.sub(r"\(.*?\)", "", name)
    s = s.replace("이용시간", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s or name


def detect_columns(df: pd.DataFrame):
    """실제 열 이름을 보고 분석에 쓸 열을 찾는다. (없으면 None)"""
    cols = list(df.columns)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    text_cols = [c for c in cols if c not in num_cols]

    g1 = "구분1" if "구분1" in cols else (text_cols[0] if text_cols else None)
    g2 = "구분2" if "구분2" in cols else (text_cols[1] if len(text_cols) > 1 else None)

    mean_col = next((c for c in num_cols if "평균" in c and "시간" in c), None)
    cnt_col = next((c for c in num_cols if "사례수" in c), None)
    rate_col = next((c for c in num_cols if "이용률" in c), None)

    exclude = ("요금", "만원", "TV", "OTT", "뉴스", "SNS", "시청", "검색", "평균")
    bins = [
        c for c in num_cols
        if parse_hours(c) is not None and not any(w in c for w in exclude)
    ]
    bins.sort(key=lambda c: parse_hours(c)[0])
    return dict(g1=g1, g2=g2, mean=mean_col, cnt=cnt_col, rate=rate_col, bins=bins)


def handle_missing_dup(df: pd.DataFrame, key_cols):
    """중복 제거 → 핵심 열 결측 행 제거 → 그 밖의 숫자 열 결측은 중앙값으로 대체."""
    info = {"원본 행 수": len(df)}
    dup_n = int(df.duplicated().sum())
    d = df.drop_duplicates().reset_index(drop=True)
    info["중복 제거 후 행 수"] = len(d)

    keep = ~d[key_cols].isna().any(axis=1)
    removed_rows = int((~keep).sum())
    d = d[keep].reset_index(drop=True)
    info["결측 처리 후 행 수(최종)"] = len(d)

    filled = 0
    for c in d.columns:
        if c in key_cols:
            continue
        if pd.api.types.is_numeric_dtype(d[c]) and d[c].isna().any():
            filled += int(d[c].isna().sum())
            d[c] = d[c].fillna(d[c].median())

    return d, dict(counts=info, dup=dup_n, removed_rows=removed_rows, filled=filled)


def is_total_row(df, g1, g2):
    m = pd.Series(False, index=df.index)
    for c in (g1, g2):
        if c:
            m |= df[c].astype(str).str.strip().str.lower().isin(TOTAL_WORDS)
    return m


def build_std(df, g1, g2, mean_col, cnt_col, rate_col, bins, labels, total_mask):
    """분석에 쓰기 편하도록 열 이름을 통일한 표를 만든다."""
    out = pd.DataFrame(index=df.index)
    if g2:
        out[C_DIM] = df[g1].astype(str)
        out[C_GRP] = df[g2].astype(str)
    else:
        out[C_DIM] = "전체"
        out[C_GRP] = df[g1].astype(str)
    out[C_N] = df[cnt_col] if cnt_col else np.nan
    out[C_RATE] = df[rate_col] if rate_col else np.nan
    out[C_MEAN] = df[mean_col]
    for c, lab in zip(bins, labels):
        out[lab] = df[c]
    out[C_TOTAL] = total_mask.values
    return out


# ------------------------------------------------------------------
# 탭 1: 데이터 확인
# ------------------------------------------------------------------
def render_overview(ctx):
    raw, clean0, enc = ctx["raw"], ctx["clean0"], ctx["enc"]
    ci = ctx["clean_info"]

    st.header("데이터 확인과 전처리")
    st.markdown(
        "업로드한 파일이 어떤 모양인지 먼저 살펴보고, 분석하기 좋게 정리한 과정을 보여 줍니다. "
        "**전처리**란 분석 전에 자료의 공백, 자료형, 빠진 값, 겹치는 값을 정리하는 일입니다."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("행(집단) 개수", f"{raw.shape[0]:,}")
    c2.metric("열 개수", f"{raw.shape[1]:,}")
    c3.metric("인코딩(글자 저장 방식)", enc)
    c4.metric("원본 결측치 개수", f"{ci['missing_total']:,}")

    st.subheader("1) 원본 데이터 미리보기")
    show_table(raw.head(10))
    st.caption("처음 10행입니다. 이 자료는 응답자 한 사람 한 사람의 답이 아니라, 집단별로 요약한 표입니다.")

    st.subheader("2) 열 이름과 데이터 유형")
    type_rows = []
    for rc, cc in zip(raw.columns, clean0.columns):
        type_rows.append({
            "열 이름": cc,
            "처리 전 유형": korean_dtype(raw[rc]),
            "처리 후 유형": korean_dtype(clean0[cc]),
            "결측치 수": int(clean0[cc].isna().sum()),
            "서로 다른 값 수": int(clean0[cc].nunique(dropna=True)),
        })
    show_table(pd.DataFrame(type_rows))

    st.subheader("3) 정리한 내용")
    st.markdown(
        f"- 문자 값의 앞뒤·중복 공백을 정리한 칸: **{ci['stripped_cells']:,}개**\n"
        f"- 숫자형으로 변환한 열: **{len(ci['converted_cols'])}개**"
        + (f" ({', '.join(ci['converted_cols'][:5])}{' 등' if len(ci['converted_cols']) > 5 else ''})"
           if ci["converted_cols"] else "")
        + f"\n- 숫자로 바꿀 수 없어 빈 값이 된 칸: **{ci['unconvertible']:,}개**"
    )

    st.subheader("4) 중복·결측치 처리 전후 비교")
    proc = ctx["proc"]
    counts = proc["counts"]
    show_table(pd.DataFrame({"단계": list(counts.keys()), "행 개수": list(counts.values())}))
    msgs = []
    msgs.append(
        f"완전히 똑같은 중복 행 **{proc['dup']}개**를 제거했습니다." if proc["dup"]
        else "완전히 똑같은 중복 행은 없었습니다."
    )
    msgs.append(
        f"핵심 열(구분, 평균 이용시간, 시간 구간 비율)에 빈 값이 있던 행 **{proc['removed_rows']}개**를 제거했습니다."
        if proc["removed_rows"] else "핵심 열에 빈 값이 있는 행은 없었습니다."
    )
    if proc["filled"]:
        msgs.append(f"그 밖의 숫자 열의 빈 칸 **{proc['filled']}개**는 열의 중앙값으로 채웠습니다.")
    st.info(" ".join(msgs))

    dup_labels = ctx["std_all"].duplicated(subset=[C_DIM, C_GRP]).sum()
    if dup_labels:
        st.warning(f"같은 '구분·집단' 이름이 겹치는 행이 {int(dup_labels)}개 있습니다. 해석할 때 주의하세요.")

    st.subheader("5) 전체 집계 행(총인원) 확인")
    n_total = int(ctx["std_all"][C_TOTAL].sum())
    if n_total:
        names = ", ".join(ctx["std_all"].loc[ctx["std_all"][C_TOTAL], C_GRP].tolist())
        st.success(
            f"전체를 한꺼번에 나타내는 집계 행 **{n_total}개**({names})를 찾았습니다. "
            "이 행은 하나의 '집단'이 아니라 모든 집단을 합친 값이므로 **집단 간 비교 분석에서는 제외**했습니다. "
            "(그래프의 '전체 평균' 기준선으로만 참고합니다.)"
        )
    else:
        st.info("'총인원', '전체' 같은 집계 행은 발견되지 않아 모든 행을 비교에 사용합니다.")

    st.subheader("6) 비교에 사용하는 집단")
    a = ctx["std"]
    grp_tbl = (
        a.groupby(C_DIM, sort=False)[C_GRP]
        .agg(["count", lambda x: ", ".join(x)])
        .reset_index()
    )
    grp_tbl.columns = ["구분", "집단 수", "집단 이름"]
    show_table(grp_tbl)
    st.caption(
        "성별·연령·지역 등은 같은 사람을 서로 다른 기준으로 나눈 것입니다. "
        "그래서 모든 구분을 한꺼번에 놓고 볼 때는 같은 사람이 여러 집단에 겹쳐 들어 있다는 점을 기억해야 합니다."
    )

    st.subheader("7) 이용 시간 구간 비율 검증")
    bins = ctx["bins"]
    sums = ctx["std_all"][bins].sum(axis=1)
    st.markdown(
        f"사용하는 구간 열: **{', '.join(bins)}**  \n"
        f"각 집단의 구간 비율 합계는 **{sums.min():.1f}% ~ {sums.max():.1f}%** 입니다."
    )
    if (sums - 100).abs().max() > 1.5:
        st.warning("일부 집단의 비율 합계가 100%에서 많이 벗어납니다. 열 선택이 올바른지 확인해 주세요.")
    else:
        st.caption("반올림 때문에 생기는 0.1%p 안팎의 차이를 빼면 합계가 100%이므로, 구간들이 전체를 나눈 자료임을 알 수 있습니다.")


# ------------------------------------------------------------------
# 탭 2: 탐구 질문 1
# ------------------------------------------------------------------
def render_q1(ctx):
    st.header("탐구 질문 1. 연령대별 하루 평균 스마트폰 이용 시간에는 어떤 차이가 있을까?")
    st.markdown(
        "연령대마다 하루에 스마트폰을 쓰는 시간이 얼마나 다른지 **평균 이용 시간(분)** 으로 비교합니다. "
        "막대가 길수록 그 연령대가 스마트폰을 더 오래 쓴다는 뜻입니다."
    )

    a = ctx["std"]
    std_all = ctx["std_all"]
    dims = list(dict.fromkeys(a[C_DIM]))
    age_dim = next((v for v in dims if "연령" in str(v)), None)
    if age_dim is None:
        st.warning("'연령' 구분을 찾지 못했습니다. 비교할 구분을 직접 골라 주세요.")
        age_dim = st.selectbox("비교할 구분", dims, key="q1_dim")
    sub = a[a[C_DIM] == age_dim].dropna(subset=[C_MEAN]).copy()
    if len(sub) < 2:
        st.warning("비교할 집단이 2개 미만이라 분석할 수 없습니다.")
        return

    vals = sub[C_MEAN].to_numpy(dtype=float)
    groups = sub[C_GRP].tolist()
    i_max, i_min = int(np.argmax(vals)), int(np.argmin(vals))
    mx, mn = vals[i_max], vals[i_min]

    tot = std_all[std_all[C_TOTAL]]
    total_mean = float(tot[C_MEAN].iloc[0]) if len(tot) else None

    c1, c2, c3 = st.columns(3)
    c1.metric("가장 오래 쓰는 집단", groups[i_max], fmt_min(mx))
    c2.metric("가장 적게 쓰는 집단", groups[i_min], fmt_min(mn))
    c3.metric("두 집단의 차이", fmt_min(mx - mn), f"약 {mx / mn:.1f}배" if mn > 0 else None, delta_color="off")

    colors = []
    for i in range(len(vals)):
        if i == i_max:
            colors.append("#1f4e9c")
        elif i == i_min:
            colors.append("#e07b39")
        else:
            colors.append("#9db7d9")
    fig = go.Figure(
        go.Bar(
            x=groups,
            y=vals,
            marker_color=colors,
            text=[f"{v:.1f}분" for v in vals],
            textposition="outside",
            hovertemplate="%{x}<br>평균 %{y:.1f}분<extra></extra>",
        )
    )
    if total_mean is not None:
        fig.add_hline(
            y=total_mean, line_dash="dash", line_color="gray",
            annotation_text=f"전체 평균 {total_mean:.1f}분", annotation_position="top right",
        )
    fig.update_layout(
        title="연령대별 하루 평균 스마트폰 이용 시간",
        xaxis_title="연령대",
        yaxis_title="하루 평균 이용 시간(분)",
        yaxis_range=[0, float(vals.max()) * 1.2],
        showlegend=False,
    )
    show_chart(fig, key="q1_bar")
    st.caption("진한 파란색: 가장 긴 집단 / 주황색: 가장 짧은 집단 / 점선: 전체 평균(비교 대상이 아닌 기준선)")

    # ---- 표
    tbl = pd.DataFrame({"연령대": groups})
    if sub[C_N].notna().any():
        tbl["사례수(명)"] = sub[C_N].round().astype("Int64").to_numpy()
    tbl["평균 이용시간(분)"] = np.round(vals, 1)
    tbl["시간·분으로 표시"] = [fmt_min(v) for v in vals]
    if total_mean is not None:
        tbl["전체 평균과의 차이(분)"] = [f"{v - total_mean:+.1f}" for v in vals]
    prev = [np.nan] + list(np.diff(vals))
    tbl["바로 앞 연령대와의 차이(분)"] = ["-" if pd.isna(v) else f"{v:+.1f}" for v in prev]
    show_table(tbl)

    # ---- 자동 해석
    st.subheader("핵심 해석")
    lines = [
        f"하루 평균 스마트폰 이용 시간이 가장 긴 연령대는 **{groups[i_max]}**{josa_ro(groups[i_max])[len(groups[i_max]):]} **{fmt_min(mx)}({mx:.1f}분)**, "
        f"가장 짧은 연령대는 **{groups[i_min]}**{josa_ro(groups[i_min])[len(groups[i_min]):]} **{fmt_min(mn)}({mn:.1f}분)** 입니다. "
        f"두 집단의 차이는 약 **{mx - mn:.1f}분**"
        + (f"으로, {josa(groups[i_max], '이', '가')} {groups[i_min]}의 약 **{mx / mn:.1f}배**입니다." if mn > 0 else "입니다.")
    ]

    after = vals[i_max:]
    before = vals[: i_max + 1]
    if len(after) >= 3 and np.all(np.diff(after) <= 0) and i_max > 0:
        lines.append(
            f"**{groups[i_max]}** 때 이용 시간이 가장 길고, 그 이후에는 연령대가 높아질수록 이용 시간이 계속 줄어드는 모습입니다."
        )
    elif len(after) >= 3 and np.all(np.diff(after) <= 0):
        lines.append("가장 어린 연령대부터 연령대가 높아질수록 이용 시간이 계속 줄어드는 모습입니다.")
    if len(before) >= 3 and np.all(np.diff(before) >= 0) and i_max > 0:
        lines.append(f"{groups[i_max]}까지는 연령대가 올라갈수록 이용 시간이 늘어납니다.")

    d = np.diff(vals)
    k = int(np.argmax(np.abs(d)))
    lines.append(
        f"바로 이웃한 연령대끼리 비교하면 **{groups[k]} → {groups[k + 1]}** 구간에서 "
        f"{'늘어난' if d[k] > 0 else '줄어든'} 폭이 **{abs(d[k]):.1f}분**으로 가장 큽니다."
    )

    if total_mean is not None:
        above = [g for g, v in zip(groups, vals) if v > total_mean]
        below = [g for g, v in zip(groups, vals) if v < total_mean]
        if above:
            lines.append(f"전체 평균({total_mean:.1f}분)보다 오래 쓰는 연령대: **{', '.join(above)}**.")
        if below:
            lines.append(f"전체 평균보다 적게 쓰는 연령대: **{', '.join(below)}**.")

    if sub[C_RATE].notna().any():
        low = sub.loc[sub[C_RATE].astype(float).idxmin()]
        if float(low[C_RATE]) < 90:
            lines.append(
                f"참고로 **{low[C_GRP]}**{josa(low[C_GRP], '은', '는')[len(str(low[C_GRP]).strip()):]} 스마트폰 이용률이 {float(low[C_RATE]):.1f}%로 다른 집단보다 낮아, "
                "스마트폰을 쓰는 사람 자체가 상대적으로 적다는 점도 함께 생각해 볼 수 있습니다."
            )

    st.info("\n\n".join(lines))
    st.caption(
        "이 자료에는 응답자 개인의 값이 없고 집단별 평균만 있습니다. 그래서 ANOVA 같은 통계 검정은 할 수 없고, "
        "집단 평균의 격차 크기로 차이를 비교했습니다. 사례수(표본 크기)가 작은 집단은 평균이 조금 더 흔들릴 수 있습니다."
    )


# ------------------------------------------------------------------
# 탭 3: 탐구 질문 2
# ------------------------------------------------------------------
def corr_table(d, bins):
    rows = []
    y = d[C_MEAN].astype(float)
    for b in bins:
        x = d[b].astype(float)
        if x.nunique() < 2 or y.nunique() < 2 or len(d) < 3:
            rows.append((b, np.nan, np.nan, np.nan))
            continue
        r, p = stats.pearsonr(x, y)
        rho, _ = stats.spearmanr(x, y)
        rows.append((b, r, rho, p))
    return pd.DataFrame(rows, columns=["구간", "r", "rho", "p"])


def render_q2(ctx):
    st.header("탐구 질문 2. 이용 시간대별 비율과 평균 이용 시간 사이에는 어떤 관계가 있을까?")
    bins = ctx["bins"]
    a = ctx["std"]

    st.markdown(
        "이 자료에서 **이용 시간대**는 '오전·오후' 같은 시각이 아니라, **하루 이용 시간의 길이 구간**"
        f"({', '.join(bins)})을 뜻합니다. 각 집단에서 이 구간에 속한 사람의 비율이 얼마나 되는지, "
        "그 비율이 높을수록 집단의 **평균 이용 시간**이 어떻게 달라지는지 살펴봅니다."
    )
    st.info(
        "**꼭 알아둘 점:** 구간별 비율을 모두 더하면 100%입니다. 한 구간의 비율이 늘어나면 다른 구간의 비율은 "
        "줄어들 수밖에 없습니다. 그래서 긴 시간 구간과 짧은 시간 구간의 결과가 서로 **반대 방향**으로 나오는 것은 자연스러운 일입니다."
    )

    ALL = "전체 집단 (모든 구분을 함께)"
    dims = list(dict.fromkeys(a[C_DIM]))
    scope = st.selectbox("분석할 집단 범위", [ALL] + dims, key="q2_scope")
    d = a if scope == ALL else a[a[C_DIM] == scope]
    d = d.dropna(subset=[C_MEAN] + bins).copy()
    d["표시이름"] = (d[C_DIM] + " · " + d[C_GRP]) if scope == ALL else d[C_GRP]
    n = len(d)
    if n < 4:
        st.warning(f"선택한 범위의 집단이 {n}개뿐이라 관계를 분석하기 어렵습니다. 다른 범위를 골라 주세요.")
        return
    if n < 8:
        st.warning(f"집단이 {n}개로 적어서 결과는 '참고용'으로만 보세요. 점 몇 개가 달라져도 결과가 크게 바뀔 수 있습니다.")
    if scope == ALL:
        st.caption(
            f"총 {n}개 집단을 사용합니다. 같은 사람이 성별·연령·직업 등 여러 집단에 겹쳐 있으므로, "
            "아래의 p값은 정확한 검정값이 아니라 참고용입니다."
        )

    # ---------------- (1) 상관계수 막대그래프
    st.subheader("1) 구간별 비율과 평균 이용 시간의 상관관계")
    st.markdown(
        "**상관계수(r)** 는 두 값이 함께 움직이는 정도를 -1~+1로 나타낸 값입니다. "
        "+1에 가까우면 '비율이 높을수록 평균 이용 시간도 길다', -1에 가까우면 '비율이 높을수록 평균 이용 시간은 짧다', "
        "0에 가까우면 뚜렷한 관계가 없다는 뜻입니다."
    )
    ct = corr_table(d, bins)
    figc = go.Figure(
        go.Bar(
            x=ct["구간"],
            y=ct["r"],
            marker_color=["#1f4e9c" if (pd.notna(v) and v >= 0) else "#d65f5f" for v in ct["r"]],
            text=[("-" if pd.isna(v) else f"{v:+.2f}") for v in ct["r"]],
            textposition="outside",
            hovertemplate="%{x}<br>r = %{y:.2f}<extra></extra>",
        )
    )
    figc.update_layout(
        title="이용 시간 구간별 비율과 평균 이용 시간의 상관계수",
        xaxis_title="이용 시간 구간",
        yaxis_title="상관계수(r)",
        yaxis_range=[-1.2, 1.2],
        showlegend=False,
    )
    figc.add_hline(y=0, line_color="gray")
    show_chart(figc, key="q2_corr")
    st.caption("파란색: 비율이 높을수록 평균 이용 시간이 긴 경향(+) / 빨간색: 비율이 높을수록 평균 이용 시간이 짧은 경향(-)")

    valid = ct.dropna(subset=["r"])
    lines = []
    if len(valid):
        best = valid.loc[valid["r"].idxmax()]
        worst = valid.loc[valid["r"].idxmin()]
        if best["r"] > 0.2:
            lines.append(
                f"**'{best['구간']}'** 비율이 높은 집단일수록 평균 이용 시간이 길어지는 경향이 있습니다. "
                f"(r = {best['r']:+.2f}, {strength_word(best['r'])} 양(+)의 상관)"
            )
        if worst["r"] < -0.2:
            lines.append(
                f"반대로 **'{worst['구간']}'** 비율이 높은 집단일수록 평균 이용 시간이 짧아지는 경향이 있습니다. "
                f"(r = {worst['r']:+.2f}, {strength_word(worst['r'])} 음(-)의 상관)"
            )
        weak = valid[valid["r"].abs() < 0.4]["구간"].tolist()
        if weak:
            lines.append(
                f"{', '.join(weak)} 구간은 상관이 약해서(|r| < 0.4), 이 구간의 비율만으로는 평균 이용 시간을 설명하기 어렵습니다."
            )
        lines.append(
            "비율의 합이 100%이기 때문에, 긴 시간 구간(예: 가장 오른쪽 구간)에 사람이 많이 몰린 집단은 "
            "짧은 시간 구간에는 사람이 적을 수밖에 없습니다. 그래서 양의 상관과 음의 상관이 양쪽 끝에서 대칭처럼 나타납니다."
        )
    if lines:
        st.subheader("핵심 해석")
        st.info("\n\n".join(lines))
    with st.expander("상관계수 표로 보기"):
        t = ct.copy()
        t["관계 세기"] = [("-" if pd.isna(r) else strength_word(r)) for r in t["r"]]
        t["방향"] = [("-" if pd.isna(r) else ("양(+)" if r >= 0 else "음(-)")) for r in t["r"]]
        t = t.rename(columns={"r": "피어슨 r", "rho": "스피어만 ρ(순위 상관)", "p": "p값(참고용)"})
        for c in ["피어슨 r", "스피어만 ρ(순위 상관)"]:
            t[c] = t[c].round(3)
        t["p값(참고용)"] = t["p값(참고용)"].map(lambda v: "-" if pd.isna(v) else f"{v:.4f}")
        show_table(t)

    # ---------------- (2) 산점도
    st.subheader("2) 구간 하나를 골라 산점도로 확인하기")
    pick = st.selectbox("살펴볼 이용 시간 구간", bins, index=len(bins) - 1, key="q2_bin")
    x = d[pick].astype(float)
    y = d[C_MEAN].astype(float)
    color_col = C_DIM if scope == ALL else "표시이름"
    size_col = None
    if d[C_N].notna().all():
        size_col = C_N
    figs = px.scatter(
        d, x=pick, y=C_MEAN, color=color_col, size=size_col, size_max=22,
        hover_name="표시이름",
        labels={pick: f"'{pick}' 비율(%)", C_MEAN: "평균 이용 시간(분)", C_DIM: "구분"},
        title=f"'{pick}' 비율과 평균 이용 시간",
    )
    if x.nunique() > 1:
        res = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        figs.add_trace(
            go.Scatter(
                x=xs, y=res.intercept + res.slope * xs, mode="lines",
                line=dict(color="black", dash="dash"), name="추세선", hoverinfo="skip",
            )
        )
    figs.update_layout(xaxis_title=f"'{pick}' 비율(%)", yaxis_title="평균 이용 시간(분)")
    show_chart(figs, key="q2_scatter")
    st.caption("점 하나는 집단 하나입니다." + (" 점이 클수록 사례수(조사 인원)가 많은 집단입니다." if size_col else "") + " 점선은 전체 경향을 나타내는 추세선입니다.")

    if x.nunique() > 1:
        res = stats.linregress(x, y)
        r = res.rvalue
        direction = "길어지는" if res.slope > 0 else "짧아지는"
        txt = (
            f"**'{pick}'** 비율이 1%p(퍼센트포인트) 높아질 때 평균 이용 시간은 대략 **{abs(res.slope):.1f}분 {direction}** 경향입니다. "
            f"상관계수는 r = {r:+.2f}({strength_word(r)} 상관)이고, 이 구간의 비율이 평균 이용 시간 차이의 약 "
            f"**{r ** 2 * 100:.0f}%** 를 설명합니다(결정계수 R² = {r ** 2:.2f}). "
        )
        if abs(r) >= 0.6:
            txt += "점들이 추세선 가까이에 모여 있어 관계가 비교적 분명합니다."
        elif abs(r) >= 0.4:
            txt += "점들이 어느 정도 흩어져 있어, 경향은 보이지만 예외도 꽤 있습니다."
        else:
            txt += "점들이 넓게 흩어져 있어 뚜렷한 관계라고 말하기 어렵습니다."
        txt += " 다만 상관관계가 있다고 해서 '원인과 결과'라고 단정할 수는 없습니다."
        st.info(txt)
    else:
        st.warning("선택한 구간의 비율이 모든 집단에서 같아 추세선을 그릴 수 없습니다.")

    # ---------------- (3) 100% 누적 막대
    st.subheader("3) 집단별 시간대 구성(100%)과 평균 이용 시간")
    ds = d.sort_values(C_MEAN).copy()
    note = ""
    if len(ds) > 20:
        ds = pd.concat([ds.head(10), ds.tail(10)])
        note = f" (집단이 많아 평균 이용 시간이 가장 짧은 10개와 가장 긴 10개 집단만 표시했습니다.)"
    ds["표시"] = ds["표시이름"] + " (" + ds[C_MEAN].round(0).astype(int).astype(str) + "분)"
    long = ds.melt(id_vars=["표시"], value_vars=bins, var_name="이용 시간 구간", value_name="비율(%)")
    try:
        k = len(bins)
        palette = px.colors.sample_colorscale("Blues", [0.15 + 0.8 * i / max(k - 1, 1) for i in range(k)])
    except Exception:
        palette = px.colors.qualitative.Set2
    figb = px.bar(
        long, x="비율(%)", y="표시", color="이용 시간 구간", orientation="h",
        category_orders={"이용 시간 구간": bins, "표시": ds["표시"].tolist()},
        color_discrete_sequence=palette,
        title="집단별 이용 시간 구간 비율 (아래쪽일수록 평균 이용 시간이 짧음 → 위쪽일수록 김)",
    )
    figb.update_layout(
        barmode="stack", xaxis_title="비율(%)", yaxis_title="",
        height=max(380, 26 * len(ds) + 140), legend_title_text="이용 시간 구간",
    )
    show_chart(figb, key="q2_stack")
    st.caption("막대 전체 길이는 100%입니다. 이름 옆 괄호는 그 집단의 평균 이용 시간입니다." + note)

    kk = max(1, len(d) // 3)
    ranked = d.sort_values(C_MEAN)
    low_g, high_g = ranked.head(kk), ranked.tail(kk)
    first_b, last_b = bins[0], bins[-1]
    st.info(
        f"평균 이용 시간이 **긴 상위 {kk}개 집단**은 '{last_b}' 비율이 평균 **{high_g[last_b].mean():.1f}%**, "
        f"'{first_b}' 비율이 평균 **{high_g[first_b].mean():.1f}%** 입니다. "
        f"반대로 평균 이용 시간이 **짧은 하위 {kk}개 집단**은 '{last_b}' 비율이 **{low_g[last_b].mean():.1f}%**, "
        f"'{first_b}' 비율이 **{low_g[first_b].mean():.1f}%** 입니다. "
        "즉, 오래 쓰는 집단은 긴 시간 구간 쪽으로 사람이 쏠려 있고, 적게 쓰는 집단은 짧은 시간 구간에 사람이 많이 모여 있습니다."
    )

    # ---------------- (4) 한 걸음 더
    with st.expander("한 걸음 더: 비율만으로 평균을 얼마나 맞힐 수 있을까?"):
        mids = ctx["bin_mid"]
        if all(b in mids for b in bins):
            w = d[bins].astype(float)
            est = (w * np.array([mids[b] for b in bins])).sum(axis=1) / w.sum(axis=1)
            actual = d[C_MEAN].astype(float)
            r_est = np.corrcoef(est, actual)[0, 1] if est.nunique() > 1 else np.nan
            mae = float(np.mean(np.abs(est - actual)))
            lo, hi = float(min(est.min(), actual.min())), float(max(est.max(), actual.max()))
            fige = px.scatter(
                x=est, y=actual, hover_name=d["표시이름"],
                labels={"x": "구간 비율로 계산한 추정 평균(분)", "y": "실제 평균 이용 시간(분)"},
                title="구간 비율로 계산한 추정 평균 vs 실제 평균",
            )
            fige.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                      line=dict(color="gray", dash="dash"), name="일치선", hoverinfo="skip"))
            show_chart(fige, key="q2_est")
            mid_txt = ", ".join(f"{b} → {mids[b]:.0f}분" for b in bins)
            st.markdown(
                f"각 구간의 가운데 값({mid_txt}; 마지막 구간은 한 시간 더 긴 값으로 가정)에 구간 비율을 곱해 "
                "평균을 직접 계산해 보았습니다."
            )
            if pd.notna(r_est):
                st.info(
                    f"계산한 추정 평균과 실제 평균의 상관계수는 **{r_est:.2f}**, 평균적인 오차는 약 **{mae:.1f}분**입니다. "
                    "점이 점선에 가까울수록 '시간대별 비율이 평균 이용 시간을 거의 결정한다'는 뜻입니다. "
                    "평균은 결국 구간별 비율로 만들어지는 값이기 때문에 둘의 관계가 강하게 나오는 것은 자연스럽습니다."
                )
        else:
            st.caption("구간 열 이름에서 시간 범위를 읽지 못해 이 계산은 생략했습니다.")

        st.markdown("---")
        st.markdown(
            f"**여러 구간을 동시에 고려한 선형 회귀(scikit-learn)** — 비율의 합이 100%이므로 "
            f"가장 짧은 구간 **'{bins[0]}'** 을 기준으로 두고 나머지 구간만 사용합니다."
        )
        if len(bins) >= 3 and len(d) >= len(bins) + 2:
            Xr = d[bins[1:]].astype(float).to_numpy()
            yr = d[C_MEAN].astype(float).to_numpy()
            lr = LinearRegression().fit(Xr, yr)
            r2 = lr.score(Xr, yr)
            coef = pd.DataFrame({
                "구간": bins[1:],
                f"'{bins[0]}' 대신 1%p 늘면 평균 변화(분)": np.round(lr.coef_, 2),
            })
            show_table(coef)
            st.info(
                f"이 식은 평균 이용 시간 차이의 약 **{r2 * 100:.0f}%**(R² = {r2:.2f})를 설명합니다. "
                f"예를 들어 '{bins[0]}' 비율이 줄고 '{bins[-1]}' 비율이 1%p 늘면 평균 이용 시간은 약 "
                f"**{lr.coef_[-1]:+.1f}분** 달라지는 것으로 계산됩니다. 집단 수가 적고 구간들이 서로 얽혀 있어 "
                "계수 하나하나를 정확한 값으로 믿기보다 '방향과 대략적인 크기'로 참고하세요."
            )
        else:
            st.caption("집단 수가 부족해 회귀 분석은 생략했습니다.")


# ------------------------------------------------------------------
# 탭 4: 다운로드
# ------------------------------------------------------------------
def render_download(ctx):
    st.header("전처리된 데이터 내려받기")
    st.markdown(
        "공백 정리, 숫자형 변환, 중복·결측치 처리를 마친 데이터를 CSV 파일로 저장할 수 있습니다. "
        "엑셀에서도 한글이 깨지지 않도록 UTF-8(BOM) 형식으로 저장합니다."
    )
    cleaned = ctx["cleaned"]
    mask = ctx["total_mask"]
    has_total = bool(mask.any())
    exclude = False
    if has_total:
        exclude = st.checkbox("전체 집계 행(총인원)을 제외하고 저장", value=False, key="dl_excl")
    out = cleaned[~mask] if exclude else cleaned
    st.caption(f"저장될 데이터: {out.shape[0]:,}행 × {out.shape[1]:,}열")
    show_table(out.head(20))
    csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "전처리된 CSV 다운로드", data=csv_bytes, file_name=DOWNLOAD_NAME, mime="text/csv",
    )


# ------------------------------------------------------------------
# 시작 화면 · 사이드바
# ------------------------------------------------------------------
def idx_of(options, value, default=0):
    try:
        return options.index(value)
    except ValueError:
        return default


def sidebar_settings(clean0, det):
    cols = list(clean0.columns)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(clean0[c])]
    with st.sidebar.expander("분석에 사용할 열 (자동 인식 결과)", expanded=False):
        st.caption("자동으로 찾은 열입니다. 잘못 인식되었다면 직접 바꿀 수 있습니다.")
        g1 = st.selectbox("집단 구분(대분류) 열", cols, index=idx_of(cols, det["g1"]), key="sel_g1")
        g2_opts = [NONE_OPT] + cols
        g2 = st.selectbox("집단 이름(세부) 열", g2_opts, index=idx_of(g2_opts, det["g2"]), key="sel_g2")
        mean_default = det["mean"] if det["mean"] else (num_cols[0] if num_cols else cols[0])
        mean_col = st.selectbox("평균 이용시간 열", cols, index=idx_of(cols, mean_default), key="sel_mean")
        cnt_opts = [NONE_OPT] + cols
        cnt = st.selectbox("사례수 열", cnt_opts, index=idx_of(cnt_opts, det["cnt"]), key="sel_cnt")
        rate = st.selectbox("이용률 열", cnt_opts, index=idx_of(cnt_opts, det["rate"]), key="sel_rate")
        bins = st.multiselect("이용 시간 구간별 비율 열", cols, default=det["bins"], key="sel_bins")
    return dict(
        g1=g1,
        g2=None if g2 == NONE_OPT else g2,
        mean=mean_col,
        cnt=None if cnt == NONE_OPT else cnt,
        rate=None if rate == NONE_OPT else rate,
        bins=bins,
    )


def validate_selection(clean0, sel):
    problems = []
    if sel["g2"] and sel["g2"] == sel["g1"]:
        problems.append("집단 구분 열과 집단 이름 열이 같습니다.")
    if not pd.api.types.is_numeric_dtype(clean0[sel["mean"]]):
        problems.append(f"'{sel['mean']}' 열이 숫자형이 아니라 평균 이용 시간으로 쓸 수 없습니다.")
    if len(sel["bins"]) < 3:
        problems.append("이용 시간 구간별 비율 열이 3개 이상 필요합니다. 사이드바에서 열을 지정해 주세요.")
    for b in sel["bins"]:
        if not pd.api.types.is_numeric_dtype(clean0[b]):
            problems.append(f"'{b}' 열이 숫자형이 아닙니다.")
    for k in ("cnt", "rate"):
        c = sel[k]
        if c and not pd.api.types.is_numeric_dtype(clean0[c]):
            problems.append(f"'{c}' 열이 숫자형이 아닙니다. (사례수/이용률 열을 '(없음)'으로 바꿔 보세요.)")
    return problems


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------
def main():
    st.title("스마트폰 이용행태 조사 2025 데이터 분석")
    st.caption("스마트폰 이용행태 조사 자료를 자동으로 정리하고, 두 가지 탐구 질문에 대한 그래프와 해석을 보여 줍니다.")

    uploaded = st.sidebar.file_uploader("다른 CSV로 분석하려면 업로드 (선택)", type=["csv"])
    default_path = find_default_csv()

    if uploaded is not None:
        data, source = uploaded.getvalue(), f"업로드한 파일: {uploaded.name}"
    elif default_path is not None:
        try:
            data, source = default_path.read_bytes(), f"기본 데이터: {default_path.name}"
        except Exception as e:  # noqa: BLE001
            st.error(f"기본 데이터 파일을 읽지 못했습니다. ({e})")
            return
    else:
        st.error(
            f"기본 데이터 파일(`{DEFAULT_CSV}`)을 찾지 못했습니다. "
            "이 파일을 main.py와 같은 폴더에 넣거나, 왼쪽 사이드바에서 CSV를 업로드해 주세요."
        )
        return
    st.sidebar.caption(source)

    try:
        raw, enc = read_csv_bytes(data)
    except pd.errors.EmptyDataError:
        st.error("파일이 비어 있습니다. 내용이 있는 CSV 파일을 확인해 주세요.")
        return
    except Exception as e:  # noqa: BLE001
        st.error(f"파일을 읽는 중 문제가 생겼습니다. CSV 형식이 맞는지 확인해 주세요. ({e})")
        return

    if raw.shape[0] == 0 or raw.shape[1] < 3:
        st.error("행이나 열이 너무 적어 분석할 수 없습니다. 파일 내용을 확인해 주세요.")
        return

    clean0, clean_info = basic_clean(raw)
    det = detect_columns(clean0)
    sel = sidebar_settings(clean0, det)
    problems = validate_selection(clean0, sel)

    tab_names = [
        "데이터 확인·전처리",
        "탐구 질문 1: 연령대별 이용 시간",
        "탐구 질문 2: 시간대 비율과 평균",
        "전처리 데이터 다운로드",
    ]
    tabs = st.tabs(tab_names)

    if problems:
        with tabs[0]:
            st.header("데이터 확인과 전처리")
            c1, c2, c3 = st.columns(3)
            c1.metric("행 개수", f"{raw.shape[0]:,}")
            c2.metric("열 개수", f"{raw.shape[1]:,}")
            c3.metric("인코딩", enc)
            show_table(raw.head(10))
            st.error("분석에 필요한 열을 정하는 중 문제가 발견되었습니다.")
            for p in problems:
                st.markdown(f"- {p}")
            st.markdown("**현재 인식된 열 이름:** " + ", ".join(clean0.columns))
        for t in tabs[1:]:
            with t:
                st.warning("먼저 '데이터 확인·전처리' 탭의 안내에 따라 열 설정을 바로잡아 주세요.")
        return

    g1, g2, mean_col = sel["g1"], sel["g2"], sel["mean"]
    bins_raw = sorted(sel["bins"], key=lambda c: (parse_hours(c)[0] if parse_hours(c) else 1e9))
    key_cols = [g1] + ([g2] if g2 else []) + [mean_col] + bins_raw

    cleaned, proc = handle_missing_dup(clean0, key_cols)
    if cleaned.empty:
        st.error("전처리 후 남은 데이터가 없습니다. 열 설정을 확인해 주세요.")
        return

    total_mask = is_total_row(cleaned, g1, g2)

    labels = []
    for c in bins_raw:
        lab = short_label(c)
        if lab in labels:
            lab = f"{lab}_{len(labels) + 1}"
        labels.append(lab)

    std_all = build_std(cleaned, g1, g2, mean_col, sel["cnt"], sel["rate"], bins_raw, labels, total_mask)
    std = std_all[~std_all[C_TOTAL]].copy()
    if std.empty:
        st.error("집계 행을 제외하고 나니 비교할 집단이 남지 않았습니다.")
        return

    bin_mid = {}
    for c, lab in zip(bins_raw, labels):
        ph = parse_hours(c)
        if ph:
            bin_mid[lab] = (ph[0] + ph[1]) / 2 * 60

    ctx = dict(
        raw=raw, clean0=clean0, enc=enc, clean_info=clean_info, proc=proc,
        cleaned=cleaned, total_mask=total_mask, std_all=std_all, std=std,
        bins=labels, bin_mid=bin_mid,
    )

    with tabs[0]:
        render_overview(ctx)
    with tabs[1]:
        try:
            render_q1(ctx)
        except Exception as e:  # noqa: BLE001
            st.error(f"분석 중 문제가 발생했습니다: {e}")
    with tabs[2]:
        try:
            render_q2(ctx)
        except Exception as e:  # noqa: BLE001
            st.error(f"분석 중 문제가 발생했습니다: {e}")
    with tabs[3]:
        render_download(ctx)


main()
