import os
from evo.core import metrics
from evo.tools import log
from evo.tools import file_interface
import copy
from collections import defaultdict

def get_rmse(rootdir,file,file_path):
    traj_ref = file_interface.read_kitti_poses_file(os.path.join('poses', file[1:3] + '.txt'))
    traj_est = file_interface.read_kitti_poses_file(file_path)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False)
    data = (traj_ref, traj_est_aligned)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    print('{} rmse: {}'.format(file, ape_stat))
    return ape_stat

log.configure_logging(verbose=False, debug=False, silent=False)
pose_relation = metrics.PoseRelation.translation_part
ape_metric = metrics.APE(pose_relation)
rootdir = './ro3'
ad = defaultdict(list)
dd = defaultdict(list)
cd = defaultdict(list)

for file in sorted(os.listdir(rootdir)):
    file_path = os.path.join(rootdir,file)
    if os.path.isfile(file_path)  and file[-3:] == 'txt':
        try:
            ape_stat = get_rmse(rootdir,file,file_path)
            if file[0] == 'a':
                ad[file[1:3]].append(ape_stat)
            elif file[0] == 'd':
                dd[file[1:3]].append(ape_stat)
            elif file[0] == 'c':
                cd[file[1:3]].append(ape_stat)
        except:
            print('{} error'.format(file))
            os.remove(file_path)

for k, v in ad.items():
    print('sdsr S{} mean rmse: {}'.format(k,round(sum(v)/len(v),2)))

for k, v in dd.items():
    print('dsr S{} mean rmse: {}'.format(k,round(sum(v)/len(v),2)))

for k, v in cd.items():
    print('ORB S{} mean rmse: {:.2}'.format(k,round(sum(v)/len(v),2)))
