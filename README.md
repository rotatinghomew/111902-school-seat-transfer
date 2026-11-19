# 随迁入学规划资料

本地合成业务资料，不含真实儿童或学校信息。仓库包含住宅分期、逐次进度、历史交付与入学结果、家庭登记、在校生、校区资源、调拨意向及签认记录。

`data/supplies.json` 按业务表保存记录，`data/fields.json` 记录各表字段及资料中出现的值类型。`phases` 标识分期及可能接收的校区；`progress` 按记录接收时间保留分期消息，`history` 分开记录交付、入住和首次入学结果。`families` 与 `applicants` 关联家庭选择和儿童，`continuing` 记录已在校的升班占用。`resource_calendar` 按校区、学段、时段及学年记录教师与教室；`loan_offers` 和 `loan_events` 保存调拨意向及其签认或撤回，`confirmations` 保存接收确认。

使用 Linux、Python 3.11 及标准库运行 `python3 -m unittest discover -s tests -v`，检查数据字段、类型和关键关联。资料仓库不包含预测或调拨办理入口。
