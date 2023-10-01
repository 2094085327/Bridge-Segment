month = "JanFebMarAprMayJunJulAugSepOctNovDec"
# 请补充下面的语句，使用eval和input获取用户输入的数字月份，需要加入提示语句"请输入数字月份1~12:"
monthid = input("请输入数字月份1~12:")
# 请补充下面的语句，使之能够将用户输入的月份monthid转换到month字符串中该月份的起始位置
pos = (int(monthid) - 1) * 3
# 请补充下面的语句，将format中缺失的部分补充完整
print("{}月对应的英文缩写是:{}".format(monthid, month[pos:pos + 3]))
