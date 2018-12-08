import os

# The precomputed centroids
x = str(-.1428)
y = str(-.3049)
z = str(3.6890)
root = '/hdd/test/'

with open('submission_task2.csv', 'w') as sub:
  sub.write('guid/image/axis,value\n')
  for f in os.listdir(root):
      if os.path.isdir(os.path.join(root,f)):  
          for ff in os.listdir(os.path.join(root, f)):
              if ".jpg" in ff:
                mod_name = f + '/' + ff.split('_')[0]
                sub.write(mod_name + '/x,' + str(x) + '\n')
                sub.write(mod_name + '/y,' + str(y) + '\n')
                sub.write(mod_name + '/z,' + str(z) + '\n')
