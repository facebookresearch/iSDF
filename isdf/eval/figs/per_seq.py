# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pylab as plt
import os
import json
import cv2
import git

from isdf.eval import plot_utils


def restructure_res(res, n_obj):
    times = []
    rays_vox_l1 = []
    rays_vis_l1 = []
    bins_vis = []
    bins_vox = []
    surf_vis = []
    surf_vox = []
    obj_vis, obj_vox = None, None
    if n_obj != 0:
        obj_vis = [[] for i in range(n_obj)]
        obj_vox = [[] for i in range(n_obj)]

    for t in res.keys():
        times.append(res[t]['time'])
        rays_vox_l1.append(res[t]['rays']['vox']['av_l1'])
        rays_vis_l1.append(res[t]['rays']['vis']['av_l1'])
        bins_vox.append(res[t]['rays']['vox']['binned_l1'])
        bins_vis.append(res[t]['rays']['vis']['binned_l1'])
        surf_vox.append(res[t]['visible_surf']['vox']['av_l1'])
        surf_vis.append(res[t]['visible_surf']['vis']['av_l1'])

        for i in range(n_obj):
            objects = res[t]['objects']
            try:
                obj_vis[i].append(objects[i]["vis"]["av_l1"])
                obj_vox[i].append(objects[i]["vox"]["av_l1"])
            except:
                obj_vis[i].append(np.nan)
                obj_vox[i].append(np.nan)

    return (
        times,
        rays_vis_l1, rays_vox_l1,
        bins_vis, bins_vox,
        surf_vis, surf_vox,
        obj_vis, obj_vox,
    )


def do_plot(
    ax, times, rays_l1, surf, bins_loss, obj_loss,
    color, label=None, linestyle="-", col_offset=0,
):
    rays_l1 = np.array(rays_l1) * 100
    if rays_l1.ndim == 2:
        std = np.nanstd(rays_l1, axis=0)
        rays_l1 = np.nanmean(rays_l1, axis=0)
        ax[0, 0].fill_between(
            times, rays_l1 + std, rays_l1 - std, alpha=0.5, color=color)
    ax[0, 0].plot(
        times, rays_l1, label=label, color=color, linestyle=linestyle)

    surf = np.array(surf) * 100
    if surf.ndim == 2:
        std = np.nanstd(surf, axis=0)
        surf = np.nanmean(surf, axis=0)
        ax[0, 1].fill_between(
            times, surf + std, surf - std, alpha=0.5, color=color)
    ax[0, 1].plot(times, surf, color=color, linestyle=linestyle)

    if obj_loss is not None:
        obj_loss = np.array(obj_loss) * 100
        if obj_loss.ndim == 3:
            std = np.nanstd(obj_loss, axis=0)
            obj_loss = np.nanmean(obj_loss, axis=0)
            for i in range(obj_loss.shape[0]):
                ax[i, 1].fill_between(
                    times, obj_loss[i] + std[i], obj_loss[i] - std[i],
                    alpha=0.5, color=color)
        for i in range(obj_loss.shape[0]):
            ax[i, 1].plot(times, obj_loss[i], color=color, linestyle=linestyle)

    bins_loss = np.array(bins_loss) * 100
    std = None
    if bins_loss.ndim == 3:
        std = np.nanstd(bins_loss, axis=0)
        bins_loss = np.nanmean(bins_loss, axis=0)

    if bins_loss.shape[1] > 6:
        print("old bin limits, trimming!!!!")
        bins_loss = bins_loss[:, :6]

    for j in range(bins_loss.shape[1]):
        r = j // 3
        c = j % 3 + col_offset
        if j > 0:
            label = None
        ax[r, c].plot(times, bins_loss[:, j], label=label,
                      color=color, linestyle=linestyle)
        if std is not None:
            ax[r, c].fill_between(
                times,
                bins_loss[:, j] + std[:, j], bins_loss[:, j] - std[:, j],
                alpha=0.5, color=color)


def draw_keyframes(fig, ax, ours_root, seq, seqs_dir, scannet_dir):
    exps = [x for x in os.listdir(ours_root) if seq in x]
    exp_name = ours_root + "/" + exps[0]

    with open(exp_name + '/res.json', 'r') as f:
        res = json.load(f)
    with open(exp_name + "/config.json") as json_file:
        config = json.load(json_file)

    if config['dataset']['format'] == 'replicaCAD':
        data_dir = seqs_dir
        kf_dir = os.path.join(data_dir, seq, "results/")
    elif config['dataset']['format'] == 'ScanNet':
        data_dir = scannet_dir
        kf_dir = os.path.join(data_dir, seq, "frames/color/")

    kf_ixs = res["kf_indices"]
    if len(kf_ixs) > 8:
        choice = np.random.choice(
            np.arange(len(kf_ixs), dtype=int), 8, replace=False)
        kf_ixs = np.array(kf_ixs)[choice]
        kf_ixs.sort()

    kf_times = [x / 30. for x in kf_ixs]
    kf_time_labels = [f"{x:.0f}" for x in kf_times]

    kfs = []
    for ix in kf_ixs:
        if config['dataset']['format'] == 'replicaCAD':
            s = f"{ix:06}"
            rgb_file = kf_dir + "frame" + s + ".png"
        elif config['dataset']['format'] == 'ScanNet':
            rgb_file = kf_dir + str(ix) + ".jpg"

        im = cv2.imread(rgb_file)
        kfs.append(im)

    kfs = np.array(kfs)
    h, w = kfs.shape[1:3]
    kfs = np.hstack((kfs))

    gs = ax[2, 0].get_gridspec()
    for a in ax[2, :]:
        a.remove()
    axbig = fig.add_subplot(gs[2, :])
    axbig.imshow(kfs[..., ::-1])
    axbig.set_xlabel("Keyframe times (s)", fontsize=ax_label_fontsize)
    x_ticks = np.arange(len(kf_ixs)) * w + w / 2
    axbig.set_xticks(x_ticks)
    axbig.set_xticklabels(kf_time_labels, fontsize=ticks_fontsize)
    axbig.set_yticklabels([])
    axbig.set_yticks([])

    # sparsify x ticks
    kf_times = np.array(kf_times)
    carried = 0
    dts = kf_times[1:] - kf_times[:-1]
    for i, dt in enumerate(dts):
        if (dt + carried) / kf_times[-1] < 0.08:
            kf_time_labels[i + 1] = ""
            carried += dt
        else:
            carried = 0

    return kf_times, kf_time_labels


def save_plots(
    save_dir, seq, n_obj, ours_root, ours_label, ours_cols,
    seqs_dir, scannet_dir, voxblox_root=None, gpuf_root=None,
):

    nrows = 3
    ncols = 4
    col_offset = 1
    if n_obj != 0:
        ncols = 5
        col_offset = 2

    fig_vis, ax_vis = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(6 * ncols, 12))
    fig_vox, ax_vox = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(6 * ncols, 12))
    fig_vis.subplots_adjust(hspace=0.3, wspace=0.23)
    fig_vox.subplots_adjust(hspace=0.3, wspace=0.23)

    # Plot ours --------------------------------------------

    if not isinstance(ours_root, list):
        ours_root = [ours_root]
        ours_label = [ours_label]
        ours_cols = [ours_cols]

    for i in range(len(ours_label)):
        root = ours_root[i]
        label = ours_label[i]
        col = ours_cols[i]

        exps = [x for x in os.listdir(root) if seq in x]
        rays_vis_l1_all = []
        rays_vox_l1_all = []
        surf_vis_all = []
        surf_vox_all = []
        bins_vis_all = []
        bins_vox_all = []
        obj_vis_all, obj_vox_all = None, None
        if n_obj != 0:
            obj_vis_all = []
            obj_vox_all = []

        for i, exp in enumerate(exps):

            res_file = os.path.join(root, exp, "vox_res.json")
            with open(res_file, 'r') as f:
                res = json.load(f)

            # Check the experiment finished
            last_t = plot_utils.get_last_eval_t(root, exp)
            eval_times = [res[k]['time'] for k in res.keys()]
            if last_t not in eval_times:
                continue

            (
                times, rays_vis_l1, rays_vox_l1, bins_vis, bins_vox,
                surf_vis, surf_vox, obj_vis, obj_vox
            ) = restructure_res(res, n_obj)

            rays_vis_l1_all.append(rays_vis_l1)
            rays_vox_l1_all.append(rays_vox_l1)
            surf_vis_all.append(surf_vis)
            surf_vox_all.append(surf_vox)
            bins_vis_all.append(bins_vis)
            bins_vox_all.append(bins_vox)
            if n_obj != 0:
                obj_vis_all.append(obj_vis)
                obj_vox_all.append(obj_vox)

        do_plot(ax_vis, times, rays_vis_l1_all, surf_vis_all, bins_vis_all,
                obj_vis_all, col, label=label, col_offset=col_offset)
        do_plot(ax_vox, times, rays_vox_l1_all, surf_vox_all, bins_vox_all,
                obj_vox_all, col, label=label, col_offset=col_offset)

    # Plot kinectfusion+ --------------------------------------------

    if gpuf_root is not None:

        res_file = os.path.join(gpuf_root, seq, "vox_res.json")
        with open(res_file, 'r') as f:
            res = json.load(f)

        (
            times, rays_vis_l1, rays_vox_l1, bins_vis, bins_vox,
            surf_vis, surf_vox, obj_vis, obj_vox
        ) = restructure_res(res, n_obj)

        do_plot(ax_vis, times, rays_vis_l1, surf_vis, bins_vis, obj_vis,
                "C2", label="KinectFusion+", col_offset=col_offset)
        do_plot(ax_vox, times, rays_vox_l1, surf_vox, bins_vox, obj_vox,
                "C2", label="KinectFusion+", col_offset=col_offset)

    # Plot voxblox --------------------------------------------

    if voxblox_root is not None:

        res_file = os.path.join(voxblox_root, seq, "res.json")
        with open(res_file, 'r') as f:
            res = json.load(f)
        res = res['sdf_eval']
        if 'bins_lb' in res.keys():
            res.pop('bins_lb')
        if 'bins_ub' in res.keys():
            res.pop('bins_ub')

        times = []
        rays_vox_l1 = []
        rays_nn_l1 = []
        rays_fill_l1 = []
        bins_vox = []
        bins_nn = []
        bins_fill = []
        surf_vox = []
        surf_nn = []
        surf_fill = []
        obj_vox, obj_nn, obj_fill = None, None, None
        if n_obj != 0:
            obj_vox = [[] for i in range(n_obj)]
            obj_nn = [[] for i in range(n_obj)]
            obj_fill = [[] for i in range(n_obj)]
        for t in res.keys():
            times.append(res[t]['time'])
            rays_vox_l1.append(res[t]['rays']['vox']['av_l1'])
            rays_nn_l1.append(res[t]['rays']['nn']['av_l1'])
            rays_fill_l1.append(res[t]['rays']['fill']['av_l1'])
            bins_vox.append(res[t]['rays']['vox']['binned_l1'])
            bins_nn.append(res[t]['rays']['nn']['binned_l1'])
            bins_fill.append(res[t]['rays']['fill']['binned_l1'])
            surf_vox.append(res[t]['visible_surf']['vox']['av_l1'])
            surf_nn.append(res[t]['visible_surf']['nn']['av_l1'])
            surf_fill.append(res[t]['visible_surf']['fill']['av_l1'])

            objects = res[t]['objects']
            for i in range(n_obj):
                try:
                    obj_vox[i].append(objects[i]["vox"]["av_l1"])
                    obj_nn[i].append(objects[i]["nn"]["av_l1"])
                    obj_fill[i].append(objects[i]["fill"]["av_l1"])
                except:
                    obj_vox[i].append(np.nan)
                    obj_nn[i].append(np.nan)
                    obj_fill[i].append(np.nan)

        # do_plot(ax_vox, times, rays_vox_l1, surf_vox, bins_vox, obj_vox,
                # "C1", label="Voxblox", col_offset=col_offset)
        do_plot(ax_vis, times, rays_nn_l1, surf_nn, bins_nn, obj_nn,
                "C1", label="Voxblox NN", col_offset=col_offset)
        # do_plot(ax_vis, times, rays_fill_l1, surf_fill, bins_fill, obj_fill,
        #         "C1", label="Voxblox fill", linestyle="--",
        #         col_offset=col_offset)

    # Draw keyframes ------------------------------------------
    x_ticks, x_ticklabels = draw_keyframes(
        fig_vis, ax_vis, root, seq, seqs_dir, scannet_dir)
    x_ticks, x_ticklabels = draw_keyframes(
        fig_vox, ax_vox, root, seq, seqs_dir, scannet_dir)

    for ax in [ax_vis, ax_vox]:
        for r in range(2):
            for c in range(ax.shape[1]):
                ax[r, c].set_yscale('log')
                ax[r, c].set_ylim([1, 100])
                yticks = [1, 2, 5, 10, 20, 50, 100]
                ytick_labels = [f'{y:.0f}' for y in yticks]
                ax[r, c].set_yticks(yticks)
                ax[r, c].set_yticklabels(ytick_labels, fontsize=ticks_fontsize)

        ax[0, 0].title.set_text("Average")
        ax[0, 0].title.set_size(title_fontsize)
        ax[0, 0].set_ylabel("SDF error [cm]", fontsize=ax_label_fontsize)
        ax[0, 0].legend(fontsize=legend_fontsize)
        ax[0, 0].set_xlabel("Keyframe times (s)", fontsize=ax_label_fontsize)

        [x.set_linewidth(3.) for x in ax[0, 0].spines.values()]

        ax[0, 1].title.set_text("Surface (s = 0cm)")
        ax[0, 1].title.set_size(title_fontsize)
        ax[0, 1].set_ylabel("SDF error [cm]", fontsize=ax_label_fontsize)
        ax[1, 1].set_ylabel("SDF error [cm]", fontsize=ax_label_fontsize)
        # [x.set_linewidth(3.) for x in ax[1, 0].spines.values()]

        if n_obj != 0:
            [x.set_linewidth(3.) for x in ax[0, 1].spines.values()]
            [x.set_linewidth(3.) for x in ax[1, 1].spines.values()]
            ax[0, 1].title.set_text("Object 1")
            ax[1, 1].title.set_text("Object 2")
            ax[0, 1].title.set_size(title_fontsize)
            ax[1, 1].title.set_size(title_fontsize)


        for c in range(ax.shape[1]):
            ax[1, c].set_xlabel("Keyframe times (s)", fontsize=ax_label_fontsize)
            ax[1, c].set_xticks(x_ticks)
            ax[1, c].set_xticklabels(x_ticklabels, fontsize=ticks_fontsize)
            ax[0, c].set_xticks(x_ticks)
            ax[0, c].set_xticklabels(x_ticklabels, fontsize=ticks_fontsize)

        bin_limits = np.array([-1e99, 0., 0.1, 0.2, 0.5, 1., 1e99]) * 100
        for j in range(len(bin_limits) - 1):
            r = j // 3
            c = j % 3 + col_offset
            if j == 0:
                # ax[r, c].title.set_text("s < {}cm".format(int(bin_limits[1])))
                pass
            elif j == len(bin_limits) - 2:
                ax[r, c].title.set_text(
                    "{}cm <= s".format(int(bin_limits[-2])))
                ax[r, c].title.set_size(title_fontsize)
            else:
                ax[r, c].title.set_text(
                    "{}cm <= s < {}cm".format(
                        int(bin_limits[j]), int(bin_limits[j + 1])))
                ax[r, c].title.set_size(title_fontsize)
            if j == 0:
                ax[r, c].legend(fontsize=legend_fontsize)

        if n_obj == 1:
            ax[1, 1].set_yscale('log')
            ax[1, 1].title.set_text("")
            ax[1, 1].plot([0, 1], [0, 1], color="black")
            ax[1, 1].plot([0, 1], [1, 0], color="black")
            ax[1, 1].set_xlabel("")
            ax[1, 1].set_ylabel("")
            ax[1, 1].set_xlim([0, 1])
            ax[1, 1].set_ylim([0, 1])
            ax[1, 1].set_xticks([0, 1])
            ax[1, 1].set_yticks([0, 1])
            ax[1, 1].set_xticklabels(["", ""])
            ax[1, 1].set_yticklabels(["", ""])

        ax[1, 0].set_yscale('log')
        ax[1, 0].title.set_text("")
        ax[1, 0].plot([0, 1], [0, 1], color="white")
        ax[1, 0].set_xlabel("")
        ax[1, 0].set_ylabel("")
        ax[1, 0].set_xlim([0, 1])
        ax[1, 0].set_ylim([0, 1])
        ax[1, 0].set_xticks([0, 1])
        ax[1, 0].set_yticks([0, 1])
        ax[1, 0].set_xticklabels(["", ""])
        ax[1, 0].set_yticklabels(["", ""])
        ax[1, 0].spines['top'].set_visible(False)
        ax[1, 0].spines['right'].set_visible(False)
        ax[1, 0].spines['bottom'].set_visible(False)
        ax[1, 0].spines['left'].set_visible(False)

    fig_vis.savefig(save_dir + f"/{seq}_vis.png")
    fig_vox.savefig(save_dir + f"/{seq}_vox.png")


# Directories --------------------------------------------

root = git.Repo(
    '.', search_parent_directories=True).working_tree_dir + "/"


save_dir = root + "res/figs/per_seq_plots/"
os.makedirs(save_dir, exist_ok=True)

ours_root = root + "res/iSDF/best/"
voxblox_root = root + "res/voxblox/gt_traj/0.055/"
gpuf_root = root + "res/gpu_fusion/7cm_unocc/"

scannet_dir = "/mnt/sda/ScanNet/scans/"
seqs_dir = root + "/data/seqs/"

title_fontsize = 16
ax_label_fontsize = 15
legend_fontsize = 15
ticks_fontsize = 13

# Sequences ----------------------------------------------

replicaCAD_seqs = ['apt_2_mnp', 'apt_2_nav', 'apt_2_obj',
                   'apt_3_mnp', 'apt_3_nav', 'apt_3_obj']

scanNet_seqs = ['scene0010_00', 'scene0030_00', 'scene0031_00',
                'scene0004_00', 'scene0005_00', 'scene0009_00']

seqs = replicaCAD_seqs + scanNet_seqs
n_objects = [1, 0, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0]

seqs = scanNet_seqs[1:2]
n_objects = [0]

for i in range(len(seqs)):

    seq = seqs[i]
    n_obj = n_objects[i]

    print("\n", seq, "\n")

    save_plots(
        save_dir, seq, n_obj, ours_root, "iSDF", "C0",
        seqs_dir, scannet_dir, voxblox_root, gpuf_root,
    )
