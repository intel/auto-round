dic = {"a":1,"b":2,"c":3}
q = max(zip(dic.keys(),dic.values()))
p = max(zip(dic.values(),dic.keys()))
print(q[0])
print(p[0])