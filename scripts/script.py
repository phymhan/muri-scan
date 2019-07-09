import os
from datetime import datetime
import math
import skvideo.io

def time2sec(s):
    s_ = s.split(':')
    return float(s_[0])*3600 + float(s_[1])*60 + float(s_[2])

FMT = '%H:%M:%S'
# src_lin = '/media/ligong/Picasso/Share/muridata/muri_videos'
src_lin = '/home/lh599/muridata/muri_videos'
dst = '/home/lh599/Research/MURI/clips/r3'

## data on server
# ZAM is not included because of folder naming is inconsistent
locs = os.listdir(src_lin)
cnt_game = 0
cnt_player = 0
for loc in locs:
    games = os.listdir(os.path.join(src_lin, loc))
    for game in games:
        if not game.lower().endswith(loc.lower()):
            continue
        cnt_game += 1
        game_no = int(game.lower().replace(loc.lower(), ''))
        print(f'location {loc}, game {game_no}')
        players = os.listdir(os.path.join(src_lin, loc, game))
        for player in players:
            p_ = player.lower().split('_')
            if len(p_) >= 3 and p_[1].startswith('p') and not p_[-1].replace('.mp4', '').endswith(')') and 'part' not in player.lower():
                player_no = int(p_[1][1:])
                print(f'--> player {player_no} ({player})')
                cnt_player += 1

print(f'number of games: {cnt_game}, number of players: {cnt_player}')


## data in timestamp
# video clip format: f'{loc}_{game_no}_{player_no}_R{R}_.mp4'
if not os.path.exists(dst):
    os.makedirs(dst)
locs = os.listdir(src_lin)
cnt_game = 0
cnt_player = 0
with open('round_timestamps.csv', 'r') as f:
    ts = f.readlines()
ts_title = ts[0].split(',')
# round 3
R = 3
start_j = ts_title.index(f'R{R}_ding')
stop_j = ts_title.index(f'R{R+1}_ding')

ts = ts[1:]
game_ids = []
for ts_line in ts:
    game_ids.append(ts_line.split(',')[1])
print(game_ids)

locs = os.listdir(src_lin)
for loc in locs:
    games = os.listdir(os.path.join(src_lin, loc))
    for game in games:
        if not game.lower().endswith(loc.lower()) or game not in game_ids:
            continue
        game_idx = game_ids.index(game)
        ts_line = ts[game_idx].split(',')
        time_start = ts_line[start_j]
        time_stop = ts_line[stop_j]
        if time_start == '' or time_stop == '':
            continue
        time_duration = str(datetime.strptime(time_stop, FMT)-datetime.strptime(time_start, FMT))
        cnt_game += 1
        game_no = int(game.lower().replace(loc.lower(), ''))
        print(f'location {loc}, game {game_no}')
        players = os.listdir(os.path.join(src_lin, loc, game))
        for player in players:
            p_ = player.lower().split('_')
            if len(p_) >= 3 and p_[1].startswith('p') and not p_[-1].replace('.mp4', '').endswith(')') and 'part' not in player.lower():
                player_no = int(p_[1][1:])
                file_in = os.path.join(src_lin, loc, game, player)
                player_new = f'{loc}_{game_no}_{player_no}_R{R}_.mp4'
                file_out = os.path.join(dst, player_new)
                meta = skvideo.io.ffprobe(file_in)
                if 'video' not in meta:
                    continue
                if float(meta['video']['@duration']) >= time2sec(time_stop):
                    print(f'--> player {player_no} ({player})')
                    cnt_player += 1
                    os.system(f'ffmpeg -ss {time_start} -t {time_duration} -i "{file_in}" -c copy "{file_out}"')
                else:
                    print(f'!!!R{R} exceeds duration ({player})!!!')

print(f'number of games: {cnt_game}, number of players: {cnt_player}')
