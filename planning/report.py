"""Command-line entry point for the district planning review.

Examples::

    python3 -m planning.report --as-of 2026-11-19
    python3 -m planning.report --confirm C005-0 --campus 河口 --year 2028
    python3 -m planning.report --history --json

The tool reads the synthetic tables, never writes them, and assigns no
individual child to a class.  A confirmation check is advisory: it does not
create a receipt.
"""

from __future__ import annotations

import argparse
import json
import sys

from .analyzer import check_confirmation, district_report
from .data import load_snapshot
from .demand import consistency_issues
from .history import history_summary


def _money(value: float) -> str:
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def render_history(report: dict, summary: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"观察日：{report['as_of']}（数据截止 {report['data_cut']}）")
    add("")
    _render_history(add, report, summary)
    return "\n".join(lines)


def _render_history(add, report: dict, summary: dict) -> None:
    add("== 一、历史对账（交付 / 入住 / 首次入学分开） ==")
    report_history = report["history"]
    status = "，".join(f"{k} {v}" for k, v in summary["status"].items())
    add(f"分期状态：{status}；完整交付 {summary['complete_phases']} 个")
    add("进度阶段的历史滑移（第几次回报 : 阶段 : 滑移年数 -> 条数）：")
    for key, dist in summary["slip_by_ordinal_milestone"].items():
        add(f"  {key}: {dist}")
    add("交付→入住 / 交付→首次入学 的时滞：")
    for key, dist in summary["lags"].items():
        add(f"  {key}: {dist}")
    add("按计划年度的观察状态（缺结果不等于零）：")
    for plan_year, statuses in sorted(report_history["by_plan_year"].items()):
        add(f"  {plan_year}: {statuses}")
    add("只交付部分、或结果尚未回传的分期（不按完工、也不按零）：")
    for row in summary["partial_or_open_phases"]:
        add(
            f"  {row['phase']}：交付 {row['delivered']}/{row['planned']}，{row['observation']}"
        )
    add("")


def render_text(report: dict, args, summary: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"观察日：{report['as_of']}（数据截止 {report['data_cut']}）")
    add("")
    _render_history(add, report, summary)

    add("== 二、当前分期的需求（按期走向） ==")
    for group in report["groups"]:
        add(
            f"  {group['entry_year']} 年 {group['campus']} {group['grade']} 年级 "
            f"{group['tier_label']} {group['count']}（{group['scope']}）"
        )
        add(f"      适用范围：{group['applicability']}")
        for alt in group["alternatives"]:
            add(f"      备选：{alt}")
    if not report["groups"]:
        add("  当前观察日没有 2027/2028 年入学登记")
    add("")

    add("== 二·补充、按校区/年级/学年的偏多情况 ==")
    # Band capacity is the real constraint, so grade counts are shown under
    # their band with the band's slack for context.
    by_campus_year: dict[tuple[str, int], dict[int, float]] = {}
    for row in report["grade_rows"]:
        by_campus_year.setdefault((row["campus"], row["year"]), {})[row["grade"]] = row["identified"]
    commit_rows = {
        (row["campus"], row["year"], row["band"]): row for row in report["commit"]["rows"]
    }
    for (campus, year), grades in sorted(by_campus_year.items()):
        if year not in (report["as_of"][:4] and int(report["as_of"][:4]), int(report["as_of"][:4]) + 1):
            continue
        pieces = "，".join(
            f"{grade}年级 {count:g}" for grade, count in sorted(grades.items())
        )
        if not pieces:
            continue
        band_notes = "；".join(
            f"{band}含预估余 {commit_rows[(campus, year, band)]['slack']:g}"
            for band in ("低年段", "高年段")
            if (campus, year, band) in commit_rows
        )
        add(f"  {year} {campus} 已识别：{pieces}（{band_notes}）")
    add("  历史样本充分性（校区/市场；不足的不能据以承诺）：")
    for row in report["history_sufficiency"]:
        mark = "可支撑" if row["sufficient"] else "不足"
        add(
            f"    {row['campus']}/{row['market']}：{mark}；观察 {row['observed_phases']} 期，"
            f"交付 {row['delivered_homes']} 套，首次入学/交付 = "
            f"{row['ratio'] if row['ratio'] is not None else '无'}"
            + (f"，波动 {row['dispersion']}" if row["dispersion"] is not None else "")
            + (f"（{'；'.join(row['reasons'])}）" if row["reasons"] else "")
        )
    add("")

    add("== 三、按校区/年级/学年的缺口与首处不能履约学年 ==")
    add(f"承诺走向：{report['commit']['branch']}")
    if report["commit"]["first_breach"]:
        add(f"  第一处不能履约的学年：{report['commit']['first_breach']['year']}")
        add(
            "  说明：缺口分两种口径。已识别需求（续读+签收+已选+待定）的缺口是硬缺口；"
            "含未登记分期预估的缺口更大，但预估只用于成组排班，不落到个人。"
        )
        if report["commit"].get("first_breach_identified"):
            add(
                "  只看已识别需求的首处缺口行："
                + "；".join(
                    f"{u['campus']}{u['band']}缺{u['gap']}"
                    for u in report["commit"]["first_breach_identified"]["units"]
                )
            )
    for breach_label, breach in report["breaches"].items():
        add(f"  走向 {breach_label}：最早 {breach['first_year']} 年")
        for unit in breach["units"]:
            add(
                f"    {unit['campus']} {unit['band']}：需求 {unit['demand']}，"
                f"可用 {unit['seats']}，缺口 {unit['gap']}（{unit['tiers']}）"
            )
    add("  校区/学段余量（正数=可接收，负数=缺口；含已签调拨）：")
    for row in report["commit"]["rows"]:
        if abs(row["slack"]) < 0.05 and not row["demand"]:
            continue
        extra = ""
        if row.get("forecast", 0):
            extra = (
                f"；其中未登记预估 {_money(row['forecast'])}"
                f"（未交付 {_money(row['forecast_unbuilt'])}，"
                f"已交付未回传 {_money(row['forecast_unreported'])}）"
                f"，只看已识别需求余 {_money(row['slack_on_identified'])}"
            )
        add(
            f"    {row['year']} {row['campus']} {row['band']}："
            f"需求 {_money(row['demand'])} / 座位 {row['seats']}"
            f"（自有 {row['durable_seats']}），余 {_money(row['slack'])}"
            f"；{row['tiers']}{extra}"
        )
    add("")

    add("== 四、现在就承诺 vs 保留到锁定期 ==")
    table = report["scenario_table"]
    add(f"  承诺按「{table['commit_branch']}」走向；保留则覆盖所有仍可能的走向。")
    add("  等待的代价（正数=保留方案少掉的接收余量）：")
    cost = [row for row in table["rows"] if row["sacrificed"] > 0]
    for row in cost:
        add(
            f"    {row['year']} {row['campus']} {row['band']}："
            f"承诺余 {row['commit_free']} → 保留余 {row['hold_free']}，"
            f"牺牲 {row['sacrificed']}"
        )
    if not cost:
        add("    没有可量化的接收余量代价")
    gaps = [
        row
        for row in table["rows"]
        if row["hold_gap"] > 0 and row["sacrificed"] <= 0
    ]
    if gaps:
        add("  无论承诺还是保留都存在的缺口（等待解决不了）：")
        for row in gaps:
            add(
                f"    {row['year']} {row['campus']} {row['band']}："
                f"缺口 {row['hold_gap']}（承诺余 {row['commit_free']}）"
            )
    add("")

    add("== 五、锁定日与进度 ==")
    for row in report["lock"]:
        add(
            f"  {row['phase']} → {row['destination']}："
            f"最近锁定 {row['earliest_lock'] or '无'}"
            f"（{row['earliest_offer'] or '无对应意向'}），"
            f"可依赖锁定 {row['next_lock'] or '无'}"
            f"（{row['binding_offer'] or '无'}，至 {row['binding_last'] or '无'}）；"
            f"{row['verdict']}"
        )
        add(f"      通知：{row['notification']}")
        if row.get("unsigned_intent_void"):
            add(f"      签认期限：{row['unsigned_intent_void']}")
        for branch in row["branches"]:
            add(
                f"      {branch['label']} {branch['delivery_year']} 年（权重 {branch['weight']:.2f}，{branch['basis']}）"
            )
    add("")

    add("== 六、可调拨选项与已签调拨 ==")
    for loan in report["loans"]:
        add(
            f"  已签 {loan['offer']}：{loan['source']}→{loan['destination']} "
            f"{loan['band']} {loan['first']}-{loan['last']}，{loan['seats']} 座，"
            f"{loan['expires_after']} 年后失效"
        )
    for warning in report["loan_warnings"]:
        add(f"  警告 {warning}")
    for option in report["loan_options"]:
        if option.get("usable"):
            add(
                f"  可用 {option['offer']}（{option['status']}）：{option['source']}→"
                f"{option['destination']} {option['band']}，"
                f"{option['first']}-{option['last']}，加 {option['seats_added']} 座，"
                f"调出校已登记余量 {option['source_slack_after']}"
                + (
                    f"（若计入未登记预估为 {option['source_slack_including_forecast']}）"
                    if option.get("source_forecast_risk")
                    else ""
                )
            )
        elif option.get("status") in ("已撤回", "资源不可用", "不在借期"):
            add(f"  不可用 {option['offer']}：{option['reason']}")
    add("")

    add("== 七、必须重议的安排 ==")
    for kind, rows in report["renegotiations"].items():
        if not rows:
            continue
        add(f"  {kind}：")
        for row in rows:
            subject = (
                row.get("phase")
                or row.get("family")
                or row.get("offer")
                or "（未命名）"
            )
            add(f"    {subject}：{row['detail']}")
            add(f"      影响：{row['effect']}")
    add("")

    add("== 八、未决与数据冲突 ==")
    for issue in report.get("consistency", []):
        add(f"  [{issue['kind']}] {issue.get('phase') or ''}{issue.get('family') or ''}：{issue['detail']}")
        add(f"      处理：{issue['action']}")
    add("")

    if report.get("confirmation"):
        check = report["confirmation"]
        add("== 九、新签收影响复核 ==")
        add(
            f"  拟签 {check['child']} → {check['campus']}，{check['year']} 学年；"
            f"{'可签' if check['feasible'] else '不可签'}"
        )
        for row in check.get("sibling_status", []):
            add(f"  兄弟姐妹：{row['child']} — {row['action']}")
        for conflict in check["conflicts"]:
            add(f"  冲突：{conflict}")
        seen = set()
        for block in check["blocking"]:
            if block in seen:
                continue
            seen.add(block)
            add(f"  受阻：{block}")
        for warning in check.get("warnings", []):
            add(f"  风险：{warning}")
        if check.get("blocking") and check.get("blocking_detail"):
            add("  卡住原因分类：")
            for item in check["blocking_detail"]:
                add(f"    [{item['kind']}] {item['detail']}")
        if check["first_breach"]:
            add(
                f"  该儿童首处不能履约学年："
                f"{check['first_breach']['first_year']}"
            )
        else:
            add("  该儿童所在年级/校区在预测期内未见缺口")
        add("  按年级（接收校与家庭原走向校区）：")
        for row in sorted(
            check.get("grade_impacts", []),
            key=lambda x: (x["campus"], x["year"], x["grade"]),
        ):
            add(
                f"    {row['year']} {row['campus']} {row['grade']} 年级 "
                f"{row['tier']} {row['count']}"
            )
        add("  影响（接收校全学年 + 签认年各校）：")
        for impact in check["impacts"]:
            add(
                f"    {impact['year']} {impact['campus']} {impact['band']}："
                f"需求 {impact['demand']} / 座位 {impact['seats']}，余 {impact['slack']}；"
                f"{impact['tiers']}"
            )
        if check.get("no_option_reason"):
            add(f"  无可履约选项：{check['no_option_reason']}")
        if check["alternatives"]:
            add("  仍可用的调拨选项：")
            for option in check["alternatives"]:
                add(
                    f"    {option['offer']} {option['source']}→{option['destination']} "
                    f"{option['first']}-{option['last']}，加 {option['seats_added']} 座"
                )
        add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="随迁入学跨学年调拨复核")
    parser.add_argument("--as-of", help="观察日（ISO 日期）；缺省为数据截止日")
    parser.add_argument("--confirm", metavar="CHILD", help="拟签收儿童，如 C005-0")
    parser.add_argument("--campus", help="拟签收校区")
    parser.add_argument("--year", type=int, help="拟签收学年")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--history", action="store_true", help="只输出历史对账")
    args = parser.parse_args(argv)

    snapshot = load_snapshot(as_of=args.as_of)
    report = district_report(snapshot)
    report["consistency"] = consistency_issues(snapshot)
    summary = history_summary(snapshot)

    if args.confirm and args.campus and args.year:
        check = check_confirmation(snapshot, args.confirm, args.campus, args.year)
        report["confirmation"] = {
            "child": check.child,
            "campus": check.campus,
            "year": check.year,
            "feasible": check.feasible,
            "family": check.family,
            "siblings": check.siblings,
            "sibling_status": check.sibling_status,
            "conflicts": check.conflicts,
            "blocking": check.blocking,
            "first_breach": check.first_breach,
            "grade_impacts": check.grade_impacts,
            "blocking_detail": check.blocking_detail,
            "no_option_reason": check.no_option_reason,
            "warnings": check.warnings,
            "impacts": check.impacts,
            "alternatives": check.alternatives,
        }

    if args.json:
        payload = summary if args.history else report
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    elif args.history:
        print(render_history(report, summary))
    else:
        print(render_text(report, args, summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
