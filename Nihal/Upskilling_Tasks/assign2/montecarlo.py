import numpy as np

#100x100 board is centred about (50,50)
inside = 0
total = 0
centre = np.array([50,50])
while(total<1001):
    randpoint = np.random.rand(1,2)*100
    total+=1
    if np.linalg.norm(randpoint-centre)<50:
        inside+=1
print(4*(inside/total))