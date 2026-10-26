"""Numbers transcribed from arXiv:2609.04561 (Tables 1, 2, 3, 5, 8 and Fig. 4) and quantities derived from them."""
import math
T1 = {  # HR %: ESC-50, US8K, FSD50K
 "small":    {"original": (23.50, 12.33, 21.35), "always": (1.50, 0.81, 6.68), "gated": (1.53, 0.84, 6.84)},
 "medium":   {"original": (26.50, 14.52, 41.95), "always": (1.82, 0.62, 8.02), "gated": (2.75, 0.66, 8.78)},
 "large-v3": {"original": (44.25, 76.08, 21.35), "always": (1.50, 0.87, 0.18), "gated": (8.38, 2.74, 1.15)},
}
T2 = {  # WER %: clean, other
 "small":    {"original": (4.04, 8.38), "always": (4.51, 11.48), "gated": (4.37, 9.13)},
 "medium":   {"original": (3.66, 7.29), "always": (6.40, 15.06), "gated": (5.47, 11.68)},
 "large-v3": {"original": (4.06, 5.87), "always": (12.95, 13.13), "gated": (6.17, 6.57)},
}
T3 = {  # FRR %: clean, other
 "small":    {"original": (0.00, 0.00), "always": (0.64, 4.86), "gated": (0.41, 2.58)},
 "medium":   {"original": (0.03, 0.27), "always": (5.68, 16.87), "gated": (4.07, 9.97)},
 "large-v3": {"original": (0.04, 0.00), "always": (10.50, 11.47), "gated": (2.86, 1.40)},
}
T5 = {  # medium, tau: (HR, WER, FRR) for original and +proj
 0.4: ((13.08, 5.77, 0.80), (2.89, 11.49, 10.91)), 0.5: ((20.42, 5.54, 0.35), (3.78, 9.90, 9.01)),
 0.6: ((29.49, 5.48, 0.14), (5.09, 8.58, 7.03)), 0.7: ((42.75, 5.46, 0.09), (7.85, 7.28, 4.76)),
}
FIG4 = {"BoH": (38.75, 75.41, 18.11), "WhisperX VAD": (4.13, 3.05, 8.55)}  # large-v3
N_NS = (800, 8732, 8621); N_SP = (2620, 2939)
mean = lambda v: sum(v) / len(v)

def wilson(p, n, z=1.96):
    p /= 100; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return 100*(c-h), 100*(c+h)

if __name__ == "__main__":
    print("== Abstract check: mean over 3 models x 3 datasets")
    for m in ("original", "always", "gated"):
        print(f"  {m:9s} {mean([mean(T1[k][m]) for k in T1]):.2f}")
    o, a, g = (mean([mean(T1[k][m]) for k in T1]) for m in ("original", "always", "gated"))
    print(f"  relative reduction always {100*(1-a/o):.2f}  gated {100*(1-g/o):.2f}")
    print("  per-model averages:", {k: {m: round(mean(v), 2) for m, v in T1[k].items()} for k in T1})
    print("  gated WER increase:", [round(T2[k]['gated'][i]-T2[k]['original'][i], 2) for k in T2 for i in (0, 1)])
    print("  Fig4 avgs:", {k: round(mean(v), 2) for k, v in FIG4.items()})
    print("  Table5 orig tau=0.6 HR vs Table1 medium orig avg:", T5[0.6][0][0], round(mean(T1['medium']['original']), 2))
    print("  Table5 +proj tau=0.6 WER/FRR vs Tables2-3 medium gated avg:", T5[0.6][1][1:], round(mean(T2['medium']['gated']), 3), round(mean(T3['medium']['gated']), 3))
    print("  Table5 +proj tau=0.6 HR vs Table1 medium gated avg:", T5[0.6][1][0], round(mean(T1['medium']['gated']), 2))
    print("\n== A2: implied share of reference words removed by rejection, if accepted transcripts were unchanged")
    for k in T2:
        for m in ("gated", "always"):
            for i, s in enumerate(("clean", "other")):
                w0, w1, f = T2[k]["original"][i], T2[k][m][i], T3[k][m][i]
                w = 100*(w1-w0)/(100-w0)
                print(f"  {k:8s} {m:6s} {s:5s} dWER {w1-w0:5.2f}  FRR {f:5.2f}  implied words {w:5.2f}  ratio {w/f if f else float('nan'):.2f}")
    print("  Sec 5.1 val-clean (always, l=28 r=4): Fig.1 WER 3.94 vs text 11.59 @ FRR 9.69 -> implied ratio",
          round((100*(11.59-3.94)/(100-3.94))/9.69, 2))
    print("\n== A3: Wilson 95% CIs, large-v3 HR")
    for m in ("original", "always", "gated"):
        print(f"  {m:9s}", [f"{h:.2f} [{wilson(h,n)[0]:.2f},{wilson(h,n)[1]:.2f}]" for h, n in zip(T1['large-v3'][m], N_NS)])
    for m, v in FIG4.items():
        print(f"  {m:13s}", [f"{h:.2f} [{wilson(h,n)[0]:.2f},{wilson(h,n)[1]:.2f}]" for h, n in zip(v, N_NS)])
    print("  small ESC always vs gated:", wilson(1.50, 800), wilson(1.53, 800))
