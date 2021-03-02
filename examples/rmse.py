import os
from evo.core import metrics
from evo.tools import log
from evo.tools import file_interface
import copy
from collections import defaultdict
import sys
from shutil import copyfile

def get_rmse(ape_metric,file,file_path):
    traj_ref = file_interface.read_kitti_poses_file(os.path.join('poses', file[1:3] + '.txt'))
    traj_est = file_interface.read_kitti_poses_file(file_path)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False)
    data = (traj_ref, traj_est_aligned)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    # print('{} rmse: {}'.format(file, ape_stat))
    return ape_stat

def main(res_root_path, save_root_path):
    log.configure_logging(verbose=False, debug=False, silent=False)
    pose_relation = metrics.PoseRelation.translation_part
    ape_metric = metrics.APE(pose_relation)

    # ad = defaultdict(list)
    dd = defaultdict(list)
    cd = defaultdict(list)

    for file in sorted(os.listdir(res_root_path)):
        file_path = os.path.join(res_root_path,file)
        if os.path.isfile(file_path)  and file[-3:] == 'txt':
            try:
                ape_stat = get_rmse(ape_metric,file,file_path)
                # if file[0] == 'a':
                #     ad[file[1:3]].append(ape_stat)
                # el
                if file[0] == 'd':
                    dd[file[1:3]].append((file,ape_stat))
                elif file[0] == 'c':
                    cd[file[1:3]].append((file,ape_stat))
            except:
                print('{} error'.format(file))
                os.remove(file_path)

    # for k, v in ad.items():
    #     print('sdsr S{} mean rmse: {}'.format(k,round(sum(v)/len(v),2)))

    total_dsr_rmse = 0
    for k, v in dd.items():
        v = sorted(v,key=lambda x:x[1])
        for vi in v[5:]:
            file_path = os.path.join(res_root_path, vi[0])
            os.remove(file_path)
        if save_root_path != '0':
            idx = 0
            for vi in v[:5]:
                print('{} rmse: {}'.format(vi[0], vi[1]))
                file_path = os.path.join(res_root_path, vi[0])
                save_path = os.path.join(save_root_path, vi[0][:3]+str(idx)+vi[0][-4:])
                copyfile(file_path,save_path)
                idx += 1
        v = [vi[1] for vi in v[:5]]
        rmse = round(sum(v)/len(v),2)
        print('dsr S{} mean rmse: {}'.format(k,rmse))
        total_dsr_rmse += rmse
    print('total_dsr_rmse: ', total_dsr_rmse)
    print()

    total_ORB_rmse = 0
    for k, v in cd.items():
        v = sorted(v, key=lambda x: -x[1])
        for vi in v[5:]:
            file_path = os.path.join(res_root_path, vi[0])
            os.remove(file_path)
        if save_root_path != '0':
            idx = 0
            for vi in v[:5]:
                print('{} rmse: {}'.format(vi[0], vi[1]))
                file_path = os.path.join(res_root_path, vi[0])
                save_path = os.path.join(save_root_path, vi[0][:3] + str(idx) + '.txt')
                copyfile(file_path, save_path)
                idx += 1
        v = [vi[1] for vi in v[:5]]
        rmse = round(sum(v) / len(v), 2)
        print('ORB S{} mean rmse: {:.2}'.format(k,rmse))
        total_ORB_rmse += rmse
    print('total_ORB_rmse: ', total_ORB_rmse)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: ./res_root_path save_root_path')
    main(sys.argv[1], sys.argv[2])
