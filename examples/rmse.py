import os
from evo.core import metrics
from evo.tools import log
from evo.tools import file_interface
import copy
from collections import defaultdict
import sys
from shutil import copyfile

def get_stat(ape_metric,rpe_metric, rre_metric, file,file_path):
    traj_ref = file_interface.read_kitti_poses_file(os.path.join('poses', file[1:3] + '.txt'))
    traj_est = file_interface.read_kitti_poses_file(file_path)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False)
    data = (traj_ref, traj_est_aligned)
    ape_metric.process_data(data)
    rpe_metric.process_data(data)
    rre_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    rpe_stat = rpe_metric.get_statistic(metrics.StatisticsType.mean)
    rre_stat = rre_metric.get_statistic(metrics.StatisticsType.mean)
    return ape_stat, rpe_stat, rre_stat

def sort_stat(v,save_root_path,res_root_path,i,k,d):
    if d == 'DSR' or d == 'DS':
        vres = sorted(v, key=lambda x: x[i])
    else:
        vres = sorted(v, key=lambda x: -x[i])
    # vres = v
    idx = 0
    res = 0
    for vi in vres[:5]:
        print('{} ate: {}m, rpe: {}%, rre: {}deg/100m'.format(vi[0], vi[1], vi[2], vi[3]))
        if save_root_path != '0':
            file_path = os.path.join(res_root_path, vi[0])
            save_path = os.path.join(save_root_path, vi[0][:3] + str(idx) + vi[0][-4:])
            copyfile(file_path, save_path)
            idx += 1
        res += vi[i]
    res = round(res / 5, 2)
    if i == 1:
        print('{} S{} rmse ate: {}m'.format(d, k, res))
    elif i == 2:
        print('{} S{} mean rpe: {}%'.format(d, k, res))
    elif i == 3:
        print('{} S{} mean rpe: {}deg/100m'.format(d, k, res))
    return res


def main(res_root_path, save_root_path):
    log.configure_logging(verbose=False, debug=False, silent=False)
    pose_relation = metrics.PoseRelation.translation_part
    ape_metric = metrics.APE(pose_relation)


    # normal mode
    delta = 100
    delta_unit = metrics.Unit.meters
    all_pairs = True

    p1 = metrics.PoseRelation.rotation_angle_deg
    rpe_metric = metrics.RPE(pose_relation,delta,delta_unit, all_pairs)
    rre_metric = metrics.RPE(p1, delta, delta_unit, all_pairs)


    ad = defaultdict(list)
    dd = defaultdict(list)
    cd = defaultdict(list)

    for file in sorted(os.listdir(res_root_path)):
        file_path = os.path.join(res_root_path,file)
        if os.path.isfile(file_path)  and file[-3:] == 'txt':
            try:
                ape_stat, rpe_stat, rre_stat = get_stat(ape_metric,rpe_metric, rre_metric,file,file_path)
                if file[0] == 'a':
                    ad[file[1:3]].append((file,ape_stat, rpe_stat, rre_stat))
                elif file[0] == 'd':
                    dd[file[1:3]].append((file,ape_stat, rpe_stat, rre_stat))
                elif file[0] == 'c':
                    cd[file[1:3]].append((file,ape_stat, rpe_stat, rre_stat))
            except:
                print('{} error'.format(file))
                os.remove(file_path)

    def print_stat(xd,d):
        total_ate = 0
        total_rpe = 0
        total_rre = 0
        for k, v in xd.items():
            rpe = sort_stat(v, '0', res_root_path, 2, k, d)
            rre = sort_stat(v, '0', res_root_path, 3, k, d)
            ate = sort_stat(v, save_root_path, res_root_path, 1, k, d)
            total_ate += ate
            total_rpe += rpe
            total_rre += rre
            print()

        print('total_{}_mean_rpe: {}%'.format(d,total_rpe))
        print('total_{}_mean_rre: {}deg/100m'.format(d,total_rre))
        print('total_{}_rmse_ate: {}m'.format(d,total_ate))
        print()
    print_stat(ad,'DS')
    print_stat(dd,'DSR')
    print_stat(cd,'ORB')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: ./res_root_path save_root_path')
    main(sys.argv[1], sys.argv[2])


# os.chdir('D:/dyna2/examples/cpp/results')
# for i in range(11):
#     if not os.path.exists('{0:02}'.format(i)):
#         os.mkdir('{0:02}'.format(i))

# for vi in v[5:]:
        #     file_path = os.path.join(res_root_path, vi[0])
        #     os.remove(file_path)
