from pathlib import Path


a = Path('/mlcv1/WorkingSpace/Personal/baotg/Kalapa/src/postprocess/fix-address/temp.csv').open('r').readlines()
b = Path("/mlcv/WorkingSpace/Personals/thuyendt/research/kapala/gt_v2.csv").open('r').readlines()

a = [x.strip() for x in a][1:]
b = [x.strip() for x in b][1:]

m = 0
n = 0
for l in a:
    m += len(l.split(',')[1])

for l in b:
    n += len(l.split(',')[1])
    
print("Predict: ", m)
print("GT: ", n)
print("Percent: ", m / n)