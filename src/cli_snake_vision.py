

def cli_snake_vision(observation):
    print_str = ''
    map_obj = {5: 'W',
               4: 'R',
               3: 'G',
               2: 'S',
               1: 'H',
               0: '0'}
    obs_up = observation[0:11]
    obs_down = observation[11:22]
    obs_left = observation[22:33]
    obs_right = observation[33:]
    row = obs_left
    dist_leftW = 0
    dist_upW = 0
    line = []
    for i in obs_left:
        if i != 1:
            key = map_obj[int(i.item())]
            line.insert(0, key)
        if i == 5:
            break
        dist_leftW += 1

    for i in obs_right:
        key = map_obj[int(i.item())]
        line.append(key)
        if i == 5:
            break

    row = []
    for i in obs_up:
        if i != 1:
            key = map_obj[int(i.item())]
            row.insert(0, key)
        if i == 5:
            break
        dist_upW += 1
    for i in obs_down:
        key = map_obj[int(i.item())]
        row.append(key)
        if i == 5:
            break

    for row_idx in range(len(row)):
        if row_idx == dist_upW:
            for j in line:
                print_str += j + ' '
        else:
            for line_idx in range(len(line)):
                if line_idx != dist_leftW:
                    print_str += '  '
                else:
                    print_str += row[row_idx]
        print_str += '\n'
    print(print_str)
