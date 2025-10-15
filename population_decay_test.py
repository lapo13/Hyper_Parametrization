i = 300
for j in range(75):
     if j>0 and j%2==0:
          i *= 0.67
     int_i = int(i)
     if int_i <= 4:
          print(f"Stop at iteration {j+1} with population size {int_i}")
     else:
          print(f"Iteration {j+1}: population size {int_i}")