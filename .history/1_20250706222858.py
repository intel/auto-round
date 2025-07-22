dic = {"a":3,"b":2,"c":1}
q = max(zip(dic.keys(),dic.values()))
p = max(zip(dic.values(),dic.keys()))
print(q[0])
print(p[1])