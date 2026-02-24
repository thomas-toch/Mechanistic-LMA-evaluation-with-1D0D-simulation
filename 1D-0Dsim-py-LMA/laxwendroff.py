import numpy as np
import math
from numba import njit, prange

pi = math.pi
sqrt = math.sqrt
pow = math.pow

@njit(fastmath=True)
def LaxWendroff(nartery, imaxtree, visc_kr, Atree, Atreem1, A0, inv_A0mid, Utree, Utreem1,
                    tbeta, tbetamid, p0, Atreem, Ptreem, Qtreem, Utreem, dt, dxi, roi, fr, cow_geo, exclude_artery):

    half_dt_dxi = 0.5 * dt * dxi
    half_dt = 0.5 * dt
    vkr_pi_fr = visc_kr * pi * fr

    for itree in range(1, nartery + 1):

        if (cow_geo != 0 and exclude_artery[itree] == 1):  # skip excluded arteries
            continue

        imax = imaxtree[itree]

        for j in range(imax):
            # --- Step 1 ---
            abw = Atree[itree, j]
            afw = Atree[itree,j + 1]
            ubw = Utree[itree,j]
            ufw = Utree[itree,j + 1]
            a0bw = A0[itree, j]
            a0fw = A0[itree, j + 1]

            inv_a0bw = 1.0 / a0bw
            inv_a0fw = 1.0 / a0fw
            sqrt_abw_a0bw = sqrt(abw * inv_a0bw)
            sqrt_afw_a0fw = sqrt(afw * inv_a0fw)

            qbw = abw * ubw
            qfw = afw * ufw

            ehrbw = tbeta[itree, j]
            ehrfw = tbeta[itree, j + 1]
            pbw = ehrbw * (sqrt_abw_a0bw - 1.0) + p0
            pfw = ehrfw * (sqrt_afw_a0fw - 1.0) + p0

            fbw = 0.5 * ubw * ubw + pbw * roi
            ffw = 0.5 * ufw * ufw + pfw * roi

            inv_abw = 1.0 / abw
            inv_afw = 1.0 / afw
            sbw = -vkr_pi_fr * ubw * inv_abw
            sfw = -vkr_pi_fr * ufw * inv_afw

            Atreem1[itree,j] = 0.5 * (abw + afw) - half_dt_dxi * (qfw - qbw)
            Utreem1[itree,j] = 0.5 * (ufw + ubw) - half_dt_dxi * (ffw - fbw) + half_dt * 0.5 * (sbw + sfw)

            if j == 0:
                continue

            # --- Step 2 ---
            abhw = Atreem1[itree,j - 1]
            afhw = Atreem1[itree,j]
            ubhw = Utreem1[itree,j - 1]
            ufhw = Utreem1[itree,j]

            # a0bhw = 0.5 * (A0[itree,j - 1] + A0[itree,j])
            # a0fhw = 0.5 * (A0[itree,j + 1] + A0[itree,j])
            # inv_a0bhw = 1.0 / a0bhw
            # inv_a0fhw = 1.0 / a0fhw
            inv_a0bhw = inv_A0mid[itree,j]
            inv_a0fhw = inv_A0mid[itree,j+1]
            sqrt_abhw = sqrt(abhw * inv_a0bhw)
            sqrt_afhw = sqrt(afhw * inv_a0fhw)

            qbhw = abhw * ubhw
            qfhw = afhw * ufhw

            # ehrbhw = 0.5 * (tbeta[itree, j - 1] + tbeta[itree, j])
            # ehrfhw = 0.5 * (tbeta[itree, j + 1] + tbeta[itree, j])
            ehrbhw = tbetamid[itree,j]
            ehrfhw = tbetamid[itree,j+1]
            pbhw = ehrbhw * (sqrt_abhw - 1.0) + p0
            pfhw = ehrfhw * (sqrt_afhw - 1.0) + p0

            fbhw = 0.5 * ubhw * ubhw + pbhw * roi
            ffhw = 0.5 * ufhw * ufhw + pfhw * roi

            inv_abhw = 1.0 / abhw
            inv_afhw = 1.0 / afhw
            sbhw = -vkr_pi_fr * ubhw * inv_abhw
            sfhw = -vkr_pi_fr * ufhw * inv_afhw

            Atreem_j = abw - dt * dxi * (qfhw - qbhw)
            Atreem[itree,j] = Atreem_j

            inv_a0bw = 1.0 / a0bw  # 再計算しても良い
            sqrt_atreem_a0 = sqrt(Atreem_j * inv_a0bw)
            Ptreem[itree,j] = ehrbw * (sqrt_atreem_a0 - 1.0) + p0

            Utreem_j = ubw - dt * dxi * (ffhw - fbhw) + half_dt * (sbhw + sfhw)
            Utreem[itree,j] = Utreem_j
            Qtreem[itree, j] = Atreem_j * Utreem_j

    return [Atreem, Ptreem, Utreem, Qtreem]

@njit(fastmath=True, parallel=True)
def LaxWendroff_opt(nartery, imaxtree, visc_kr, Atree, Atreem1, A0, inv_A0mid, Utree, Utreem1,
                    tbeta, tbetamid, p0, Atreem, Ptreem, Qtreem, Utreem, dt, dxi, roi, fr, cow_geo, exclude_artery):

    half_dt_dxi = 0.5 * dt * dxi
    half_dt = 0.5 * dt
    dt_dxi = dt * dxi
    vkr_pi_fr = visc_kr * pi * fr

    # 外側のループを並列化
    for itree in prange(1, nartery + 1):

        if cow_geo != 0 and exclude_artery[itree] == 1:
            continue

        imax = imaxtree[itree]

        # 【最適化】2次元配列の行への参照を事前に取得（内側のループでのインデックス計算コストを削減）
        row_Atree = Atree[itree]
        row_Utree = Utree[itree]
        row_A0 = A0[itree]
        row_tbeta = tbeta[itree]
        
        row_Atreem1 = Atreem1[itree]
        row_Utreem1 = Utreem1[itree]
        row_inv_A0mid = inv_A0mid[itree]
        row_tbetamid = tbetamid[itree]
        
        row_Atreem = Atreem[itree]
        row_Ptreem = Ptreem[itree]
        row_Qtreem = Qtreem[itree]
        row_Utreem = Utreem[itree]

        # 【最適化】レジスタローテーション用の変数
        # 前回のイテレーション(j-1)で計算された「ハーフステップ」の値を保持する変数
        # Step 2で再計算せずにこれを使用する
        qbhw = 0.0
        fbhw = 0.0
        sbhw = 0.0

        for j in range(imax):
            # --- Step 1 ---
            # 配列アクセスの簡略化
            abw = row_Atree[j]
            afw = row_Atree[j + 1]
            ubw = row_Utree[j]
            ufw = row_Utree[j + 1]
            a0bw = row_A0[j]
            a0fw = row_A0[j + 1]

            # 共通部分の計算
            inv_a0bw = 1.0 / a0bw
            sqrt_abw_a0bw = sqrt(abw * inv_a0bw)
            
            # ※ここで計算した inv_a0bw や sqrt 値の一部は Step 2 (Ptreem計算) で再利用可能だが、
            # フロー依存性を複雑にしないため、主要な flux/source 計算の持ち越しに集中する。

            inv_a0fw = 1.0 / a0fw
            sqrt_afw_a0fw = sqrt(afw * inv_a0fw)

            qbw_step1 = abw * ubw
            qfw_step1 = afw * ufw

            ehrbw = row_tbeta[j]
            ehrfw = row_tbeta[j + 1]
            pbw = ehrbw * (sqrt_abw_a0bw - 1.0) + p0
            pfw = ehrfw * (sqrt_afw_a0fw - 1.0) + p0

            fbw_step1 = 0.5 * ubw * ubw + pbw * roi
            ffw_step1 = 0.5 * ufw * ufw + pfw * roi

            inv_abw = 1.0 / abw
            inv_afw = 1.0 / afw
            sbw_step1 = -vkr_pi_fr * ubw * inv_abw
            sfw_step1 = -vkr_pi_fr * ufw * inv_afw

            # Step 1 の結果 (j+1/2 地点での値)
            # これが今回の "Forward Half Wait" (fhw) となり、次のループ(j+1)での "Backward Half Wait" (bhw) となる
            afhw_val = 0.5 * (abw + afw) - half_dt_dxi * (qfw_step1 - qbw_step1)
            ufhw_val = 0.5 * (ufw + ubw) - half_dt_dxi * (ffw_step1 - fbw_step1) + half_dt * 0.5 * (sbw_step1 + sfw_step1)

            # 結果を配列に書き込む（デバッグや他関数での利用のため）
            row_Atreem1[j] = afhw_val
            row_Utreem1[j] = ufhw_val

            # --- Step 2 用の事前計算（次のループでの再計算を防ぐ） ---
            # 現在のハーフステップ(j+1/2)の Flux, Source, Q を計算
            inv_a0fhw_mid = row_inv_A0mid[j+1] # j+1に対応する中間点
            sqrt_afhw_mid = sqrt(afhw_val * inv_a0fhw_mid)
            
            qfhw_val = afhw_val * ufhw_val
            
            ehrfhw_mid = row_tbetamid[j+1]
            pfhw_mid = ehrfhw_mid * (sqrt_afhw_mid - 1.0) + p0
            
            ffhw_val = 0.5 * ufhw_val * ufhw_val + pfhw_mid * roi
            sfhw_val = -vkr_pi_fr * ufhw_val / afhw_val

            # j=0 の場合は左側の情報(bhw)がないため、Step 2は実行できない（境界条件扱い）
            # ただし、今回の計算結果(fhw)は、次(j=1)のために保存しておく必要がある
            if j == 0:
                # 変数の持ち越し: Current Forward -> Next Backward
                qbhw = qfhw_val
                fbhw = ffhw_val
                sbhw = sfhw_val
                continue

            # --- Step 2 ---
            # j > 0 の場合、前のループからの持ち越し値 (bhw) と 今回計算した値 (fhw) を使用
            # ここでの qbhw, fbhw, sbhw は、前のイテレーションで計算した qfhw_val 等が入っている
            
            # 質量保存の更新
            # Atreem_j = abw - dt_dxi * (qfhw - qbhw) 
            # 注意: ここでの abw は Atree[j] (ループ冒頭でロード済み)
            Atreem_j = abw - dt_dxi * (qfhw_val - qbhw)
            row_Atreem[j] = Atreem_j

            # 圧力(Ptreem)の計算
            # inv_a0bw はループ冒頭で計算済み (1.0/A0[j])
            sqrt_atreem_a0 = sqrt(Atreem_j * inv_a0bw)
            row_Ptreem[j] = ehrbw * (sqrt_atreem_a0 - 1.0) + p0

            # 運動量保存の更新
            Utreem_j = ubw - dt_dxi * (ffhw_val - fbhw) + half_dt * (sbhw + sfhw_val)
            row_Utreem[j] = Utreem_j
            row_Qtreem[j] = Atreem_j * Utreem_j

            # 次のループのために変数を更新（スライドさせる）
            qbhw = qfhw_val
            fbhw = ffhw_val
            sbhw = sfhw_val

    return [Atreem, Ptreem, Utreem, Qtreem]

@njit(fastmath=True)
def parammid_cal(nartery,imaxtree,A0,tbeta,inv_A0mid,tbetamid,cow_geo,exclude_artery):

    for itree in range(1,nartery+1):
        if (cow_geo != 0 and exclude_artery[itree] == 1):
            continue
        imax = imaxtree[itree]
        for j in range(1,imax+1):
            temp_a = (A0[itree,j - 1] + A0[itree,j]) * 0.5
            inv_A0mid[itree,j] = 1 / temp_a
            tbetamid[itree,j] = (tbeta[itree,j - 1] + tbeta[itree,j]) * 0.5

    return inv_A0mid, tbetamid
