
all_lines = []
with open('submission.csv', 'r') as f:
    all_lines = f.readlines()

mod_lines = [x.split(',') for x in all_lines]
with open('new_sub.csv', 'w') as ff:
    for line in mod_lines:
        print(line)
        mod_one = line[1]
        if '0' in line[1]:
            mod_one = '1\n'
        if '1' in line[1]:
            mod_one = '1\n'
        if '2' in line[1]:
            mod_one = '0\n'

        mod_str = line[0] + ',' + mod_one
        print(mod_str)
        ff.write(mod_str)

