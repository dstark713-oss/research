# regime_model/regime_based/state_builder.py
# ============================================================
# Построение S_t на 185 акциях + WF PCA/KMeans.
# ============================================================
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

WINDOW = 252
STEP = 20
MIN_STOCKS = 30

def _dfa_hurst_1d(x, order=1, max_lag=50):
    """Быстрый DFA Hurst для 1D массива."""
    x = np.asarray(x)
    n = len(x)
    if n < 50: return np.nan
    y = np.cumsum(x - x.mean())
    max_lag = min(max_lag, n // 4)
    if max_lag < 4: return np.nan
    lags = np.unique(np.floor(np.logspace(np.log10(4), np.log10(max_lag), 10)).astype(int))
    F = []
    for lag in lags:
        if lag < 2: continue
        nc = n // lag
        if nc < 2: continue
        segs = y[:nc*lag].reshape(nc, lag)
        t = np.arange(lag)
        t_mean = t.mean()
        t_var = ((t - t_mean) ** 2).sum()
        if t_var == 0: continue
        seg_mean = segs.mean(axis=1, keepdims=True)
        cov = ((t[None, :] - t_mean) * (segs - seg_mean)).sum(axis=1)
        slope = cov / t_var
        intercept = seg_mean.squeeze(1) - slope * t_mean
        trend = intercept[:, None] + slope[:, None] * t[None, :]
        rms = np.sqrt(((segs - trend) ** 2).mean(axis=1))
        F.append((lag, rms.mean()))
    
    if len(F) < 3: return np.nan
    la = np.log(np.array([f[0] for f in F]))
    Fa = np.log(np.array([f[1] for f in F]))
    if np.isnan(la).any() or np.isnan(Fa).any(): return np.nan
    return np.polyfit(la, Fa, 1)[0]

def build_state_matrix_from_prices(close: pd.DataFrame,
                                   window: int = WINDOW,
                                   step: int = STEP) -> pd.DataFrame:
    """
    S_t на 185 акциях. Считаем признаки по каждой акции в окне, 
    затем берем медиану по акциям.
    """
    returns = close.pct_change()
    dates = returns.index
    n = len(dates)
    starts = list(range(0, n - window + 1, step))
    if not starts:
        return pd.DataFrame()

    # Ускоряем rolling статистики через pandas
    ret1 = returns
    abs_ret = returns.abs()
    
    roll_mean = ret1.rolling(window, min_periods=int(window*0.7)).mean()
    roll_std = ret1.rolling(window, min_periods=int(window*0.7)).std()
    roll_skew = ret1.rolling(window, min_periods=int(window*0.7)).skew()
    roll_kurt = ret1.rolling(window, min_periods=int(window*0.7)).kurt()
    
    # ACF proxy через rolling corr
    acf_ret_1 = ret1.rolling(window, min_periods=int(window*0.7)).corr(ret1.shift(1))
    acf_abs_1 = abs_ret.rolling(window, min_periods=int(window*0.7)).corr(abs_ret.shift(1))
    acf_abs_5 = abs_ret.rolling(window, min_periods=int(window*0.7)).corr(abs_ret.shift(5))
    
    # Downside/Upside vol
    down_ret = ret1.where(ret1 < 0)
    up_ret = ret1.where(ret1 > 0)
    down_vol = down_ret.rolling(window, min_periods=int(window*0.7)).std()
    up_vol = up_ret.rolling(window, min_periods=int(window*0.7)).std()
    vol_asym = down_vol / (up_vol + 1e-9)
    
    rows = []
    for s in starts:
        e = s + window
        dt = dates[e - 1]
        
        # Валидация
        valid = roll_std.loc[dt].notna()
        if valid.sum() < MIN_STOCKS:
            continue
            
        # Для Hurst и VR оставим цикл по валидным акциям
        valid_tickers = valid[valid].index
        hurst_list = []
        vr5_list = []
        vr20_list = []
        
        for t in valid_tickers:
            x = ret1[t].iloc[s:e].dropna().values
            if len(x) < window * 0.7: continue
            hurst_list.append(_dfa_hurst_1d(x))
            # VR
            v1 = np.var(x, ddof=1)
            if v1 > 0:
                xq5 = pd.Series(x).rolling(5).sum().dropna().values
                vr5_list.append(np.var(xq5, ddof=1) / 5 / v1)
                xq20 = pd.Series(x).rolling(20).sum().dropna().values
                vr20_list.append(np.var(xq20, ddof=1) / 20 / v1)

        row = {
            "window_end": dt,
            "mean": roll_mean.loc[dt, valid_tickers].median(),
            "std": roll_std.loc[dt, valid_tickers].median(),
            "skew": roll_skew.loc[dt, valid_tickers].median(),
            "kurt": roll_kurt.loc[dt, valid_tickers].median(),
            "acf_ret_1": acf_ret_1.loc[dt, valid_tickers].median(),
            "acf_abs_1": acf_abs_1.loc[dt, valid_tickers].median(),
            "acf_abs_5": acf_abs_5.loc[dt, valid_tickers].median(),
            "downside_vol": down_vol.loc[dt, valid_tickers].median(),
            "upside_vol": up_vol.loc[dt, valid_tickers].median(),
            "vol_asym": vol_asym.loc[dt, valid_tickers].median(),
            "hurst": np.nanmedian(hurst_list) if hurst_list else np.nan,
            "vr_5": np.nanmedian(vr5_list) if vr5_list else np.nan,
            "vr_20": np.nanmedian(vr20_list) if vr20_list else np.nan,
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("window_end")
    return df.sort_index()

def build_pc_and_cluster(states: pd.DataFrame, n_components: int = 3,
                         n_clusters: int = 4) -> pd.DataFrame:
    """PCA + KMeans на полном S_t (для сравнения)."""
    feat_cols = [c for c in states.columns]
    X = states[feat_cols].fillna(states[feat_cols].median())
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(Xs)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(Z)
    return pd.DataFrame({
        "PC1": Z[:, 0], "PC2": Z[:, 1] if Z.shape[1] > 1 else np.nan,
        "PC3": Z[:, 2] if Z.shape[1] > 2 else np.nan,
        "cluster": labels, "hurst": states["hurst"].values,
    }, index=states.index)

def build_wf_pc_and_cluster(states: pd.DataFrame, rebal_dates: list,
                            n_components: int = 3, n_clusters: int = 4,
                            min_train_windows: int = 20) -> pd.DataFrame:
    """Expanding WF PCA/KMeans (без look-ahead, с безопасным imputation)."""
    out_rows = []
    rebal_dates = sorted(rebal_dates)
    feat_cols = [c for c in states.columns]
    X_all = states[feat_cols].values
    dates_all = states.index

    for rd in rebal_dates:
        rd = pd.Timestamp(rd)
        mask_train = dates_all < rd
        if mask_train.sum() < min_train_windows:
            continue
        X_train = X_all[mask_train]

        mask_rd = dates_all <= rd
        if not mask_rd.any():
            continue
        idx_rd = np.where(mask_rd)[0][-1]
        X_rd = X_all[idx_rd:idx_rd + 1]

        # --- безопасный imputation ---
        # медиана по колонке train
        medians = np.nanmedian(X_train, axis=0)
        # если вся колонка NaN — медиана NaN, заменяем на 0
        medians = np.where(np.isfinite(medians), medians, 0.0)

        X_train_imp = np.where(np.isfinite(X_train), X_train,
                                np.broadcast_to(medians, X_train.shape))
        X_rd_imp = np.where(np.isfinite(X_rd), X_rd,
                             np.broadcast_to(medians, X_rd.shape))

        # --- проверка: нет ли оставшихся NaN ---
        if not np.isfinite(X_train_imp).all():
            # fallback: заменить все оставшиеся NaN на 0
            X_train_imp = np.nan_to_num(X_train_imp, nan=0.0,
                                         posinf=0.0, neginf=0.0)
        if not np.isfinite(X_rd_imp).all():
            X_rd_imp = np.nan_to_num(X_rd_imp, nan=0.0,
                                      posinf=0.0, neginf=0.0)

        # --- standardize (fit на train) ---
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_imp)
        X_rd_s = scaler.transform(X_rd_imp)

        # --- PCA (fit на train) ---
        pca = PCA(n_components=n_components)
        Z_train = pca.fit_transform(X_train_s)
        Z_rd = pca.transform(X_rd_s)[0]

        # --- KMeans (fit на train) ---
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        km.fit(Z_train)
        cluster = int(km.predict(Z_rd.reshape(1, -1))[0])

        hurst = float(states.iloc[idx_rd]["hurst"]) if "hurst" in states.columns else np.nan

        out_rows.append({
            "date": rd, "PC1": Z_rd[0],
            "PC2": Z_rd[1] if len(Z_rd) > 1 else np.nan,
            "PC3": Z_rd[2] if len(Z_rd) > 2 else np.nan,
            "cluster": cluster, "hurst": hurst,
        })

    return pd.DataFrame(out_rows).set_index("date").sort_index()