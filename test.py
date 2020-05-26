p = int(input())
g = input()

g = list(map(int, g))
r = [0]*(len(g)+1)
rez = [0] * (len(g)+1)

for i in range(len(g) - 1, -1, -1):
    prod = p * g[i]
    rez[i+1] = prod % 10 + r[i+1]
    r[i] = int(prod / 10)

rez[0] += r[0]

rez = sum(rez)
while rez >= 10:
    rez = str(rez)
    rez = sum(map(int, rez))

print(rez)
