import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import permutation_test


if __name__ == "__main__":
    hookformer = [356.2673918, 338.3709208, 363.9587444, 369.9190411, 373.9637252]
    tyrion = [335.4428994, 313.1979335, 321.6418363, 381.0925618, 360.349317]
    optsimmim = [290.7954322, 413.6106744, 236.5848697, 265.4767262, 295.5426651]
    opttranslator = [269.3482312, 259.6492801, 262.3526543, 400.7096369, 273.7162836]
    def levene_statistic(*samples):
        return stats.levene(*samples).statistic

    for i, (baseline, opponent) in enumerate([(hookformer, tyrion), (tyrion, opttranslator), (tyrion, optsimmim), (hookformer, optsimmim), (hookformer, opttranslator)]):
        if i == 0:
            print("# ==== HookFormer vs. TYRION ===================================================================")
        elif i == 1:
            print("# ==== TYRION vs. OptTranslator ======================================================================")
        elif i == 2:
            print("# ==== TYRION vs. OptSimMIM =======================================================================")
        elif i == 3:
            print("# ==== HookFormer vs. OptSimMIM ===================================================================")
        elif i == 4:
            print("# ==== HookFormer vs. OptTranslator ==================================================================")
        print(f"{baseline}: {np.mean(baseline)} +- {np.std(baseline)}")
        print(f"{opponent}: {np.mean(opponent)} +- {np.std(opponent)}\r\n")

        print("Test normal distribution")
        plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
        plt.hist(baseline, bins=20)
        plt.gca().set(title=f'Frequency Histogram {baseline}', ylabel='Frequency', xlabel="MDE [m]")
        plt.show()
        plt.hist(opponent, bins=20)
        plt.gca().set(title=f'Frequency Histogram {opponent}', ylabel='Frequency', xlabel="MDE [m]")
        plt.show()
        print("-> not normally distributed\r\n")

        print("Test equal variance")
        statistic, pvalue = stats.levene(baseline, opponent)
        print(f"Levene-Test - Statistic: {statistic}, p-value: {pvalue}")
        ref = permutation_test((baseline, opponent), levene_statistic, permutation_type='independent', alternative='greater')
        print(f"Permutation-Test - p-value: {ref.pvalue}")
        if ref.pvalue < 0.05:
            print("One test significant, the other not -> safer to use test with unequal variance")
            print("Using non-parametric test as prerequisites for parametric tests are not met\r\n")
        else:
            print("Still using non-parametric test as prerequisites for parametric tests are not fully met and just to be sure\r\n")

        print("One-sided Mann-Whitney-U-Test")
        statistic, pvalue = stats.mannwhitneyu(baseline, opponent, alternative="greater")
        print(f"Statistic: {statistic}, p-value: {pvalue}")
        if pvalue < 0.05:
            print("-> significant")
        else:
            print("-> not significant\r\n")

        print("Effect size metric: Cohen's d")
        d = (np.mean(baseline) - np.mean(opponent)) / np.std(baseline)
        print(f"Cohen's d: {d}\r\n")

        input("Press Enter to continue...")

    print("# ==== OptSimMIM vs. OptTranslator ===================================================================")
    print(f"{optsimmim}: {np.mean(optsimmim)} +- {np.std(optsimmim)}")
    print(f"{opttranslator}: {np.mean(opttranslator)} +- {np.std(opttranslator)}\r\n")

    print("Test normal distribution")
    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
    plt.hist(optsimmim, bins=20)
    plt.gca().set(title=f'Frequency Histogram {optsimmim}', ylabel='Frequency', xlabel="MDE [m]")
    plt.show()
    plt.hist(opttranslator, bins=20)
    plt.gca().set(title=f'Frequency Histogram {opttranslator}', ylabel='Frequency', xlabel="MDE [m]")
    plt.show()
    print("-> not normally distributed\r\n")

    print("Test equal variance")
    statistic, pvalue = stats.levene(optsimmim, opttranslator)
    print(f"Levene-Test - Statistic: {statistic}, p-value: {pvalue}")
    ref = permutation_test((optsimmim, opttranslator), levene_statistic, permutation_type='independent',
                           alternative='greater')
    print(f"Permutation-Test - p-value: {ref.pvalue}")
    if ref.pvalue < 0.05:
        print("One test significant, the other not -> safer to use test with unequal variance")
        print("Using non-parametric test as prerequisites for parametric tests are not met\r\n")
    else:
        print(
            "Still using non-parametric test as prerequisites for parametric tests are not fully met and just to be sure\r\n")

    print("One-sided Mann-Whitney-U-Test")
    statistic, pvalue = stats.mannwhitneyu(optsimmim, opttranslator, alternative="greater")
    print(f"Statistic: {statistic}, p-value: {pvalue}")
    if pvalue < 0.05:
        print("-> significant")
    else:
        print("-> not significant\r\n")

    print("Effect size metric: Cohen's d")
    s = np.sqrt(((len(optsimmim) - 1) * np.var(optsimmim) + (len(opttranslator) - 1) * np.var(opttranslator)) / (len(optsimmim) + len(opttranslator) - 2))
    d = (np.mean(optsimmim) - np.mean(opttranslator)) / s
    print(f"Cohen's d: {d}\r\n")
