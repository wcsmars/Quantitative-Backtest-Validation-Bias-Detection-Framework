"""Train/test split hygiene: declaration, overlap, embargo, ordering, coverage.

All date arithmetic is done in *common-index periods* (the aligned trading
calendar), not calendar days, so weekends/holidays never inflate a gap.

Scope: this family audits the declared windows against the supplied panel
only. It is static (the pipeline callables are unused), so it can never
verify where a model was actually fitted - a truthful declaration over the
wrong panel, or a wrong declaration over the right one, both surface here
as window/data mismatches, and the checks refuse to certify hygiene from a
sample containing no training (or no test) observations.
"""
from __future__ import annotations

import math
from typing import Callable

import pandas as pd

from ..config import AuditConfig
from ..inputs import BacktestArtifacts
from ..types import CheckResult, Severity, failed, passed, skipped, warned

# Module constants (AuditConfig carries no coverage thresholds for this
# family, so the fixed cutoffs live here).
MIN_TEST_PERIODS = 60      # absolute floor for a meaningful test window
MIN_TEST_FRAC = 0.05       # ...or this fraction of the sample, if larger
MIN_TRAIN_PERIODS = 60     # hygiene floor for in-data training periods
MIN_TRAIN_FRAC = 0.05      # ...or this fraction of the sample, if larger
                           # (mirrors the test floor; the required training
                           # size is model-dependent, so a small-but-nonzero
                           # train draws a LOW coverage WARN, while zero
                           # in-data periods is categorical - the whole
                           # family becomes vacuous and is gated in run():
                           # a bogus ancient train declaration, adversarial
                           # or a year typo, would otherwise leave the
                           # overlap and embargo checks PASSing on nothing,
                           # with only a LOW coverage WARN to show for it)
MIN_COVERAGE_FRAC = 0.60   # train+test should jointly cover >= this fraction

_FAMILY_SKIP_HINT = "to enable train/test contamination checks"


def run(artifacts: BacktestArtifacts, config: AuditConfig, *,
        signal_func: Callable | None = None,
        backtest_func: Callable | None = None) -> list[CheckResult]:
    """Train/test contamination checks. Static: pipeline callables unused."""
    tr, te = artifacts.train_period, artifacts.test_period

    # -- 1. split_declared: without both windows the family cannot run -----
    if tr is None and te is None:
        return [skipped(
            "contamination.split_declared",
            f"pass train_period=(start,end) and test_period=(start,end) "
            f"{_FAMILY_SKIP_HINT}")]
    if tr is None or te is None:
        missing = "train_period" if tr is None else "test_period"
        present = "test_period" if tr is None else "train_period"
        return [skipped(
            "contamination.split_declared",
            f"pass {missing}=(start,end) {_FAMILY_SKIP_HINT} "
            f"({present} is declared but {missing} is missing)")]

    tr_start, tr_end = pd.Timestamp(tr[0]), pd.Timestamp(tr[1])
    te_start, te_end = pd.Timestamp(te[0]), pd.Timestamp(te[1])

    idx = artifacts.common_index
    n = len(idx)
    in_train = (idx >= tr_start) & (idx <= tr_end)
    in_test = (idx >= te_start) & (idx <= te_end)
    n_train = int(in_train.sum())
    n_test = int(in_test.sum())

    # -- 0. vacuous-declaration gate -----------------------------------------
    # A declared window sharing zero common-index periods with the supplied
    # data voids every downstream measurement: overlap is vacuously 0, the
    # embargo gap is measured against nothing, and coverage can be carried
    # by the other window alone - the overlap/embargo checks would PASS on
    # hygiene the family never observed, with only a LOW coverage WARN to
    # show for it (any ancient/typo'd train year, or a train window landing
    # in a weekend data hole, produces this).
    # test_before_train stays live: it compares declared timestamps only
    # and is the one check that catches a future fake train window.
    if n_train == 0 or n_test == 0:
        return _vacuous_split(idx, n, tr_start, tr_end, te_start, te_end,
                              n_train, n_test)

    results: list[CheckResult] = [passed(
        "contamination.split_declared",
        f"train_period {tr_start.date()}..{tr_end.date()} ({n_train} periods) "
        f"and test_period {te_start.date()}..{te_end.date()} ({n_test} periods) "
        f"declared over a {n}-period sample (declared windows audited "
        f"against the supplied panel; actual fit provenance is outside this "
        f"static family's reach)",
        n_train=n_train, n_test=n_test, n_periods=n)]

    # -- 2. split_overlap (CRITICAL) ----------------------------------------
    both = in_train & in_test
    n_overlap = int(both.sum())
    if n_overlap > 0:
        overlap_dates = idx[both]
        o_start, o_end = overlap_dates[0], overlap_dates[-1]
        results.append(failed(
            "contamination.split_overlap",
            f"train and test windows share {n_overlap} periods "
            f"({o_start.date()}..{o_end.date()}) - every metric reported as "
            f"out-of-sample is partly in-sample",
            severity=Severity.CRITICAL,
            remediation="Re-split so the test window starts strictly after "
                        "the training window ends (plus an embargo of >= "
                        "label_horizon periods), then re-fit and re-evaluate "
                        "every out-of-sample statistic.",
            n_overlap=n_overlap,
            overlap_start=str(o_start.date()), overlap_end=str(o_end.date()),
            n_train=n_train, n_test=n_test))
    else:
        results.append(passed(
            "contamination.split_overlap",
            f"0 of {n} common-index periods fall in both windows "
            f"(train {n_train} periods, test {n_test} periods)",
            n_overlap=0, n_train=n_train, n_test=n_test))

    # -- 3. insufficient_embargo (HIGH, WARN) --------------------------------
    h = int(artifacts.label_horizon)
    needed = int(config.embargo_periods(h))
    if n_overlap > 0:
        results.append(skipped(
            "contamination.insufficient_embargo",
            "embargo not evaluated: train and test windows overlap - fix "
            "contamination.split_overlap first, then re-audit"))
    elif te_start <= tr_end:
        # Two distinct sub-cases share this branch; the pointer must only
        # cite a check that actually speaks to the case.
        if te_end <= tr_start:
            # Test wholly precedes train: test_before_train WARNs there.
            results.append(skipped(
                "contamination.insufficient_embargo",
                "embargo not evaluated: test window does not start after "
                "the training window ends (see "
                "contamination.test_before_train)"))
        else:
            # Declared windows overlap in calendar time yet share zero
            # common-index periods - only possible when a data hole covers
            # the whole declared overlap. No other check names this (
            # split_overlap counts shared periods and PASSes at 0), and the
            # realized train-end..test-start gap is 0 periods, so say it
            # directly instead of citing a passing check.
            results.append(skipped(
                "contamination.insufficient_embargo",
                f"embargo not evaluated: the declared windows overlap in "
                f"calendar time (test starts {te_start.date()}, on/before "
                f"the train end {tr_end.date()}) - only a gap in the data "
                f"prevents shared periods, and the realized gap between "
                f"the last train period and the first test period is 0. "
                f"Re-declare the split with the test window starting after "
                f"the training window ends (plus an embargo of >= {needed} "
                f"periods)."))
    else:
        gap = int(((idx > tr_end) & (idx < te_start)).sum())
        if gap < needed:
            results.append(warned(
                "contamination.insufficient_embargo",
                f"gap between train end and test start is {gap} periods but "
                f"labels span label_horizon={h} periods; the last {h} training "
                f"labels overlap the first test returns - purge/embargo >= "
                f"{needed} periods (Lopez de Prado purged CV)",
                severity=Severity.HIGH,
                remediation=f"Leave >= {needed} common-index periods between "
                            f"the last training date and the first test date "
                            f"(purged/embargoed split) so no training label "
                            f"spans into the test window.",
                gap=gap, needed=needed))
        else:
            results.append(passed(
                "contamination.insufficient_embargo",
                f"{gap} periods between train end and test start >= required "
                f"embargo of {needed} (label_horizon={h})",
                gap=gap, needed=needed))

    # -- 4. test_before_train (MEDIUM, WARN) ---------------------------------
    results.append(_test_before_train(tr_start, te_start, te_end))

    # -- 5. split_coverage (LOW, WARN) ----------------------------------------
    needed_test = max(MIN_TEST_PERIODS, math.ceil(MIN_TEST_FRAC * n))
    needed_train = max(MIN_TRAIN_PERIODS, math.ceil(MIN_TRAIN_FRAC * n))
    n_union = int((in_train | in_test).sum())
    coverage = n_union / n if n else float("nan")
    train_frac = n_train / n if n else float("nan")
    test_frac = n_test / n if n else float("nan")
    problems = []
    if n_test < needed_test:
        problems.append(
            f"test window has only {n_test} periods < required {needed_test} "
            f"(max({MIN_TEST_PERIODS}, {MIN_TEST_FRAC:.0%} of {n}-period "
            f"sample)) - too short to mean anything")
    if n_train < needed_train:
        # hygiene floor, not a correctness bar: the required training size
        # is model-dependent; zero in-data periods is the categorical case,
        # gated before this check ever runs
        problems.append(
            f"train window has only {n_train} in-data periods < required "
            f"{needed_train} (max({MIN_TRAIN_PERIODS}, {MIN_TRAIN_FRAC:.0%} "
            f"of {n}-period sample)) - too few observations to have fitted "
            f"anything trustworthy")
    if coverage < MIN_COVERAGE_FRAC:
        problems.append(
            f"train+test together cover only {coverage:.1%} of the "
            f"{n}-period sample (< {MIN_COVERAGE_FRAC:.0%}) - what happened "
            f"in the gap?")
    details = dict(n_test=n_test, needed_test=int(needed_test),
                   n_train=n_train, needed_train=int(needed_train),
                   coverage_frac=round(coverage, 4),
                   train_frac=round(train_frac, 4),
                   test_frac=round(test_frac, 4))
    if problems:
        results.append(warned(
            "contamination.split_coverage", "; ".join(problems),
            severity=Severity.LOW,
            remediation=f"Use a test window of >= {needed_test} and a train "
                        f"window of >= {needed_train} in-data periods, and "
                        f"account for every excluded period - unexplained "
                        f"holes in the sample often hide cherry-picked dates.",
            details=details))
    else:
        results.append(passed(
            "contamination.split_coverage",
            f"test window has {n_test} periods ({test_frac:.1%} of sample, "
            f">= {needed_test} required); train window has {n_train} "
            f"in-data periods (>= {needed_train} required); train+test "
            f"cover {coverage:.1%} of the {n}-period sample",
            details=details))

    return results


def _test_before_train(tr_start: pd.Timestamp, te_start: pd.Timestamp,
                       te_end: pd.Timestamp) -> CheckResult:
    """Check 4, on declared timestamps only - meaningful even when a window
    shares no periods with the data, so the vacuous-declaration path keeps
    it live (it is the one check that catches a future fake train window)."""
    if te_end <= tr_start:
        return warned(
            "contamination.test_before_train",
            f"test window ({te_start.date()}..{te_end.date()}) ends before the "
            f"training window starts ({tr_start.date()}) - parameters were "
            f"chosen on data from AFTER the test window (hindsight selection); "
            f"a walk-forward claim is invalid",
            severity=Severity.MEDIUM,
            remediation="Train only on data that precedes the test window (or "
                        "use true walk-forward re-fits) so parameter selection "
                        "cannot use information from after evaluation.",
            train_start=str(tr_start.date()), test_start=str(te_start.date()),
            test_end=str(te_end.date()))
    return passed(
        "contamination.test_before_train",
        f"test window ends {te_end.date()}, after the training window "
        f"starts {tr_start.date()} - evaluation data does not wholly "
        f"precede parameter selection",
        train_start=str(tr_start.date()), test_end=str(te_end.date()))


def _vacuous_split(idx: pd.DatetimeIndex, n: int,
                   tr_start: pd.Timestamp, tr_end: pd.Timestamp,
                   te_start: pd.Timestamp, te_end: pd.Timestamp,
                   n_train: int, n_test: int) -> list[CheckResult]:
    """A declared window with 0 in-data periods.

    n_test == 0 (with or without n_train == 0) is a FAIL: every statistic
    reported as out-of-sample was measured on no data from this panel -
    there is no honest workflow behind it. n_train == 0 alone is a WARN,
    not a FAIL: auditing only the OOS slice of a panel while the model was
    honestly fitted on earlier history not supplied here is legitimate -
    but the family then cannot certify split hygiene, so the downstream
    checks SKIP instead of emitting vacuous PASSes. test_before_train
    stays live (declared timestamps only). All five ids are emitted so the
    include-filter/placeholder plumbing keyed on the family ids holds.
    """
    span = (f"{idx[0].date()}..{idx[-1].date()}" if n else "an empty sample")
    which = ("test_period" if n_train else
             "train_period and test_period" if n_test == 0 else
             "train_period")
    decl = (f"declared train_period {tr_start.date()}..{tr_end.date()} has "
            f"{n_train} and test_period {te_start.date()}..{te_end.date()} "
            f"has {n_test} common-index periods in the supplied {n}-period "
            f"sample ({span})")
    if n_test == 0:
        head = failed(
            "contamination.split_declared",
            f"{decl} - every statistic reported as out-of-sample was "
            f"measured on NO data from this panel; either the declaration "
            f"is wrong (a typo'd year silently voids the whole family) or "
            f"the wrong panel was supplied",
            severity=Severity.HIGH,
            remediation=f"Fix {which} to a window the supplied data "
                        f"actually covers ({span}), or supply the panel the "
                        f"split refers to, then re-run the audit.",
            n_train=n_train, n_test=n_test, n_periods=n)
    else:
        head = warned(
            "contamination.split_declared",
            f"{decl} - either the declaration is wrong (a typo'd year "
            f"silently voids the whole family) or the model was fitted on "
            f"history not present in this panel; this static family audits "
            f"declared windows against the supplied panel only and cannot "
            f"certify split hygiene from a sample containing no training "
            f"periods",
            severity=Severity.HIGH,
            remediation=f"If the train window is real, include its history "
                        f"in the supplied panel so the split can be "
                        f"audited; otherwise fix train_period to dates the "
                        f"data covers ({span}).",
            n_train=n_train, n_test=n_test, n_periods=n)
    skip_msg = (f"not evaluated: 0 in-data periods in the declared {which} "
                f"(see contamination.split_declared) - {{what}} measured "
                f"against an empty window is vacuously clean, not evidence "
                f"of hygiene.")
    return [
        head,
        skipped("contamination.split_overlap",
                skip_msg.format(what="overlap")),
        skipped("contamination.insufficient_embargo",
                skip_msg.format(what="an embargo gap")),
        _test_before_train(tr_start, te_start, te_end),
        skipped("contamination.split_coverage",
                skip_msg.format(what="joint coverage")),
    ]
