tag_path_labor='data/labor/tags.txt'
f = open(tag_path_labor, "r", encoding='utf-8')
tag_dic = {}
task_cnt = 0
for line in f:
    task_cnt += 1
    if line[-1] == '\n':
        tag_dic[line[:-1]] = task_cnt
    else:
        tag_dic[line[:]] = task_cnt
    print(line[:])
    print(tag_dic)