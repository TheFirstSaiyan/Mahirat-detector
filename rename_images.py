import os.path
import os 
  
def main(): 
    i = 1    
    for filename in os.listdir("dhoni"): 
       dst ="dhoni" + str(i) + os.path.splitext(filename)[1]
       src ='dhoni/'+ filename 
       dst ='dhoni/'+ dst       
       os.rename(src, dst) 
       i += 1

  
if __name__ == '__main__': 
       main() 