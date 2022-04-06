# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pylab as plt
import os
import json
import git

from isdf.eval import plot_utils


# seqs and settings ---------------------------------------

chomp_ix = 2  # epsilon = 2m
cossim_ix = 1  # delta is two voxels
plot_grads = True

title_fontsize = 18
ax_label_fontsize = 20
legend_fontsize = 20
legend_fontsize_smaller = 15
ticks_fontsize = 15

plt.rc('legend', **{'fontsize': legend_fontsize})

replicaCAD_seqs = np.array(
    [['apt_2_nav', 'apt_2_obj', 'apt_2_mnp',
      'apt_3_nav', 'apt_3_obj', 'apt_3_mnp']])

scanNet_seqs = np.array(
    [['scene0010_00', 'scene0030_00', 'scene0031_00',
      'scene0004_00', 'scene0005_00', 'scene0009_00']])

seqs = np.vstack((replicaCAD_seqs, scanNet_seqs))

main_seqs = np.array(
    [['apt_3_nav', 'apt_3_obj', 'apt_3_mnp',
      'scene0030_00', 'scene0004_00', 'scene0005_00']])

supp_seqs = np.array(
    [['apt_2_nav', 'apt_2_obj', 'apt_2_mnp',
      'scene0010_00', 'scene0031_00', 'scene0009_00']])

ablt_seqs = main_seqs.reshape(2, 3)


# Plot functions ---------------------------------------------

def ema_smooth(scalars, weight):
    """ Exponential moving average smoothing """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return np.array(smoothed)


def sdf_ax_ticks(ax):
    for r in range(ax.shape[0]):
        for c in range(ax.shape[1]):
            ax[r, c].set_yscale('log')
            ax[r, c].set_ylim([2, 50])
            yticks = [2, 5, 10, 20, 50]
            ytick_labels = [f'{y:.0f}' for y in yticks]
            ax[r, c].set_yticks(yticks)
            ax[r, c].set_yticklabels(ytick_labels, fontsize=ticks_fontsize)
            ax[r, c].minorticks_off()


def chomp_ax_ticks(ax):
    for r in range(ax.shape[0]):
        for c in range(ax.shape[1]):
            ax[r, c].set_yscale('log')
            ax[r, c].set_ylim([0.02, 0.2])
            yticks = [0.02, 0.05, 0.1, 0.2]
            ytick_labels = [f'{y:.2f}' for y in yticks]
            ax[r, c].set_yticks(yticks)
            ax[r, c].set_yticklabels(ytick_labels, fontsize=ticks_fontsize)
            ax[r, c].minorticks_off()


def grad_ax_ticks(ax):
    for r in range(ax.shape[0]):
        for c in range(ax.shape[1]):
            ax[r, c].set_yscale('log')
            ax[r, c].set_ylim([0.05, 0.7])
            yticks = [0.05, 0.1, 0.2, 0.5]
            ytick_labels = [f'{y:.2f}' for y in yticks]
            ax[r, c].set_yticks(yticks)
            ax[r, c].set_yticklabels(ytick_labels, fontsize=ticks_fontsize)
            ax[r, c].minorticks_off()


# Plot ours --------------------------------------------


def do_plots(plot_fn, direc, axes, label, color):
    ax_vis, ax_vox, ax_main, ax_supp, ax_ablt, ax_percp, ax_vox_prop = axes

    ax_vis_grad_all = None
    ax_vox_grad_all = None
    ax_vis_grad_main = None
    ax_vis_grad_supp = None
    if plot_grads:
        ax_vis_grad_all = ax_vis[4:6]
        ax_vox_grad_all = ax_vox[4:6]
        ax_vis_grad_main = ax_main[2:3]
        ax_vis_grad_supp = ax_supp[2:3]

    if plot_fn == plot_voxblox:
        plot_fn(
            seqs, direc,
            label=label, color=color,
            ax_vis_sdf=ax_vis[0:2],
            ax_vis_chomp=ax_vis[2:4],
            ax_vis_grad=ax_vis_grad_all,
            ax_vox_sdf=ax_vox[0:2],
            ax_vox_chomp=ax_vox[2:4],
            ax_vox_grad=ax_vox_grad_all,
            ax_vox_prop=ax_vox_prop,
        )
    else:
        plot_fn(
            seqs, direc,
            label=label, color=color,
            ax_vis_sdf=ax_vis[0:2],
            ax_vis_chomp=ax_vis[2:4],
            ax_vis_grad=ax_vis_grad_all,
            ax_vox_sdf=ax_vox[0:2],
            ax_vox_chomp=ax_vox[2:4],
            ax_vox_grad=ax_vox_grad_all,
        )

    plot_fn(
        main_seqs, direc,
        label=label, color=color,
        ax_vis_sdf=ax_main[0:1],
        ax_vis_chomp=ax_main[1:2],
        ax_vis_grad=ax_vis_grad_main,
    )

    plot_fn(
        supp_seqs, direc,
        label=label, color=color,
        ax_vis_sdf=ax_supp[0:1],
        ax_vis_chomp=ax_supp[1:2],
        ax_vis_grad=ax_vis_grad_supp,
    )

    plot_fn(
        ablt_seqs, direc,
        label=label, color=color,
        ax_vis_sdf=ax_ablt,
    )

    plot_fn(
        ablt_seqs, direc,
        label=label, color=color,
        ax_vis_sdf=ax_percp,
    )


def plot_mean_std(ax, loss, times, r, c, color, label, title):
    if ax is not None:
        loss = np.reshape(loss, [-1, len(times)])
        mean, std = loss.mean(axis=0), loss.std(axis=0)
        ax[r, c].plot(times, mean, label=label, color=color)
        ax[r, c].fill_between(
            times, mean + std, mean - std, alpha=0.5, color=color)
        ax[r, c].title.set_text(title)
        ax[r, c].title.set_size(title_fontsize)
        ax[r, c].title.set_style('italic')
        fsize = legend_fontsize if ax.shape == 6 else legend_fontsize_smaller
        if r == 0 and c == 0 and label is not None:
            ax[r, c].legend(fontsize=fsize)


def plot_isdf(
    seqs, ours_root, label="iSDF", color="C0",
    ax_vis_sdf=None, ax_vox_sdf=None,
    ax_vis_chomp=None, ax_vox_chomp=None,
    ax_vis_grad=None, ax_vox_grad=None,
):
    print(ours_root)
    for j, seq in enumerate(seqs.flatten()):

        exps = [x for x in os.listdir(ours_root) if seq in x]
        times = []

        sdf_vis = []
        sdf_vox = []
        chomp_vis = []
        chomp_vox = []
        grad_vis = []
        grad_vox = []

        do_times = True

        last_t = plot_utils.get_last_eval_t(ours_root, exps[0])

        for i, exp in enumerate(exps):

            res_file = os.path.join(ours_root, exp, "vox_res.json")
            with open(res_file, 'r') as f:
                res = json.load(f)

            # Check the experiment finished
            eval_times = [res[k]['time'] for k in res.keys()]
            if last_t not in eval_times:
                continue

            for t in res.keys():
                sdf_vis.append(res[t]['rays']['vis']['av_l1'])
                sdf_vox.append(res[t]['rays']['vox']['av_l1'])
                chomp_vis.append(
                    res[t]['rays']['vis']['l1_chomp_costs'][chomp_ix])
                chomp_vox.append(
                    res[t]['rays']['vox']['l1_chomp_costs'][chomp_ix])
                grad_vis.append(res[t]['rays']['vis']['av_cossim'][cossim_ix])
                grad_vox.append(res[t]['rays']['vox']['av_cossim'][cossim_ix])

                if do_times:
                    times.append(res[t]['time'])

            do_times = False

        n_runs = len(sdf_vis) / len(times)
        print(seq, "- n runs:", n_runs)

        # convert to cms
        sdf_vis, sdf_vox = np.array(sdf_vis) * 100, np.array(sdf_vox) * 100

        r = j // seqs.shape[1]
        c = j % seqs.shape[1]

        axes = [
            ax_vis_sdf, ax_vox_sdf,
            ax_vis_chomp, ax_vox_chomp,
            ax_vis_grad, ax_vox_grad,
        ]
        losses = [
            sdf_vis, sdf_vox,
            chomp_vis, chomp_vox,
            grad_vis, grad_vox,
        ]

        label = label if j == 0 else None
        labels = [label] * 2 + [None] * 4
        for i in range(len(losses)):
            plot_mean_std(
                axes[i], losses[i], times, r, c, color, labels[i], seq)

        if ax_vis_sdf is not None:
            sdf_ax_ticks(ax_vis_sdf)
        if ax_vox_sdf is not None:
            sdf_ax_ticks(ax_vox_sdf)
        if ax_vis_chomp is not None:
            chomp_ax_ticks(ax_vis_chomp)
        if ax_vox_chomp is not None:
            chomp_ax_ticks(ax_vox_chomp)
        if ax_vis_grad is not None:
            grad_ax_ticks(ax_vis_grad)
        if ax_vox_grad is not None:
            grad_ax_ticks(ax_vox_grad)


# Plot gpu fusion --------------------------------------------

def plot(ax, loss, times, r, c, color, label):
    if ax is not None:
        ax[r, c].plot(times, loss, label=label, color=color)
        ax[r, c].tick_params(
            axis='x', which='major', labelsize=ticks_fontsize)
        if r == 0 and c == 0 and label is not None:
            y = 2.2 if ax.shape[1] == 6 else 0.8  # -0.27
            fsize = \
                legend_fontsize if ax.shape == 6 else legend_fontsize_smaller
            ax[r, c].legend(
                loc='upper left', bbox_to_anchor=(y, 1.375),
                fancybox=True, ncol=5, fontsize=fsize)


def plot_gpu_fusion(
    seqs, gpuf_root, label="GPU fusion", color="C2",
    ax_vis_sdf=None, ax_vox_sdf=None,
    ax_vis_chomp=None, ax_vox_chomp=None,
    ax_vis_grad=None, ax_vox_grad=None,
):

    for j, seq in enumerate(seqs.flatten()):

        res_file = os.path.join(gpuf_root, seq, "vox_res.json")
        with open(res_file, 'r') as f:
            res = json.load(f)

        times = []

        sdf_vis = []
        sdf_vox = []
        chomp_vis = []
        chomp_vox = []
        grad_vis = []
        grad_vox = []

        for t in res.keys():
            times.append(res[t]['time'])
            sdf_vis.append(res[t]['rays']['vis']['av_l1'])
            sdf_vox.append(res[t]['rays']['vox']['av_l1'])
            chomp_vis.append(
                res[t]['rays']['vis']['l1_chomp_costs'][chomp_ix])
            chomp_vox.append(
                res[t]['rays']['vox']['l1_chomp_costs'][chomp_ix])
            grad_vis.append(res[t]['rays']['vis']['av_cossim'][cossim_ix])
            grad_vox.append(res[t]['rays']['vox']['av_cossim'][cossim_ix])

        r = j // seqs.shape[1]
        c = j % seqs.shape[1]

        sdf_vis, sdf_vox = np.array(sdf_vis) * 100, np.array(sdf_vox) * 100

        axes = [
            ax_vis_sdf, ax_vox_sdf,
            ax_vis_chomp, ax_vox_chomp,
            ax_vis_grad, ax_vox_grad,
        ]
        losses = [
            sdf_vis, sdf_vox,
            chomp_vis, chomp_vox,
            grad_vis, grad_vox,
        ]

        label = label if j == 0 else None
        labels = [label] * 2 + [None] * 4
        for i in range(len(losses)):
            plot(axes[i], losses[i], times, r, c, color, labels[i])


# Plot voxblox --------------------------------------------

def plot_voxblox(
    seqs, voxblox_root, label="Voxblox", color="C1",
    ax_vis_sdf=None, ax_vox_sdf=None,
    ax_vis_chomp=None, ax_vox_chomp=None,
    ax_vis_grad=None, ax_vox_grad=None,
    ax_vox_prop=None,
):

    for j, seq in enumerate(seqs.flatten()):

        res_file = os.path.join(voxblox_root, seq, "res.json")
        with open(res_file, 'r') as f:
            res = json.load(f)
        res = res['sdf_eval']

        if "bins_lb" in res.keys():
            res.pop('bins_lb')
        if "bins_ub" in res.keys():
            res.pop('bins_ub')

        times = []

        prop_vox = []  # proportion of visible region mapped by voxblox

        sdf_vis = []
        sdf_vox = []
        chomp_vis = []
        chomp_vox = []
        grad_vis = []
        grad_vox = []

        for t in res.keys():
            times.append(res[t]['time'])
            prop_vox.append(res[t]['rays']['vox']['prop_vox'])

            # nn for sdf and grad, fill for chomp
            sdf_vis.append(res[t]['rays']['nn']['av_l1'])
            sdf_vox.append(res[t]['rays']['vox']['av_l1'])
            chomp_vis.append(
                res[t]['rays']['fill']['l1_chomp_costs'][chomp_ix])
            chomp_vox.append(
                res[t]['rays']['vox']['l1_chomp_costs'][chomp_ix])
            grad_vis.append(res[t]['rays']['nn']['av_cossim'][cossim_ix])
            grad_vox.append(res[t]['rays']['vox']['av_cossim'][cossim_ix])

        sdf_vis, sdf_vox = np.array(sdf_vis) * 100, np.array(sdf_vox) * 100

        w = 0.7
        grad_vis = ema_smooth(grad_vis, w)
        grad_vox = ema_smooth(grad_vox, w)

        r = j // seqs.shape[1]
        c = j % seqs.shape[1]

        if ax_vox_prop is not None:
            # voxblox visible proportion
            ax_vox_prop[r, c].title.set_text(seq)
            ax_vox_prop[r, c].title.set_size(title_fontsize)
            ax_vox_prop[r, c].plot(times, prop_vox)
            ax_vox_prop[r, c].set_ylim([0, 1.05])

        axes = [
            ax_vis_sdf, ax_vox_sdf,
            ax_vis_chomp, ax_vox_chomp,
            ax_vis_grad, ax_vox_grad,
        ]
        losses = [
            sdf_vis, sdf_vox,
            chomp_vis, chomp_vox,
            grad_vis, grad_vox,
        ]
        labels = [
            'Voxblox NN', 'Voxblox', None, None, None, None,
            # 'Voxblox fill', 'Voxblox',
            # 'Voxblox NN', 'Voxblox NN',
        ]
        if j != 0:
            labels = [None] * 6
        for i in range(len(axes)):
            plot(axes[i], losses[i], times, r, c, color, labels[i])
            if label is not None:
                label = None


def main(
    save_dir,
    isdf_dir,
    gpuf_dir,
    voxblox_dir,
):
    # create figs ---------------------------------------

    # all results
    n_losses = 3 if plot_grads else 2
    nrows, ncols = seqs.shape[0] * n_losses, seqs.shape[1]
    fig_vis, ax_vis = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(26, 4 * nrows))
    fig_vox, ax_vox = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(26, 4 * nrows))

    # split up seqs for paper

    nrows, ncols = main_seqs.shape[0] * n_losses, main_seqs.shape[1]
    fig_main, ax_main = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(26, 4 * nrows))
    fig_supp, ax_supp = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(26, 4 * nrows))

    fig_vis.subplots_adjust(hspace=0.26, wspace=0.23)
    fig_vox.subplots_adjust(hspace=0.26, wspace=0.23)
    fig_main.subplots_adjust(hspace=0.26, wspace=0.23)
    fig_supp.subplots_adjust(hspace=0.26, wspace=0.23)

    # vox prop visible plot

    fig_vox_prop, ax_vox_prop = plt.subplots(
        nrows=2, ncols=6, figsize=(26, 8))

    # supervision ablation study

    fig_ablt, ax_ablt = plt.subplots(
        nrows=2, ncols=3, figsize=(15, 6.5))
    fig_ablt.subplots_adjust(hspace=0.26)

    # decreased perception time plot

    fig_percp, ax_percp = plt.subplots(
        nrows=2, ncols=3, figsize=(15, 6.5))
    fig_percp.subplots_adjust(hspace=0.26)

    axes = [
        ax_vis, ax_vox, ax_main, ax_supp, ax_ablt, ax_percp, ax_vox_prop
    ]

    # plot curves ---------------------------------------

    """
    # SDF supervision ablation plot

    incSDF_root = git.Repo(
        '.', search_parent_directories=True).working_tree_dir

    do_plots(
        plot_isdf,
        incSDF_root + "/res/iSDF/sdf_supervision/ray/",
        "iSDF + Ray", color="C3")
    do_plots(
        plot_isdf,
        incSDF_root + "/res/iSDF/sdf_supervision/normal/",
        "iSDF + Normal", color="C4")
    do_plots(
        plot_isdf,
        incSDF_root + "/res/iSDF/sdf_supervision/pc/",
        "iSDF + Batch distance", color="C0")
    """

    # """
    # Perception time plots

    # do_plots(
    #     plot_isdf,
    #     incSDF_root + "/res/iSDF/frac_time_perception/0.5/",
    #     "iSDF", color="C0")
    # do_plots(
    #     plot_voxblox,
    #     incSDF_root + "/res/voxblox/gt_traj/0.078/",
    #     "Voxblox NN", color="C1")
    # do_plots(
    #     plot_gpu_fusion,
    #     incSDF_root + "/res/gpu_fusion/frac_perception_time_unocc/0.5/",
    #     "KinectFusion+", color="C2")
    # """

    do_plots(
        plot_isdf, isdf_dir, axes, "iSDF + Batch distance", color="C0")

    if voxblox_dir:
        do_plots(plot_voxblox, voxblox_dir, axes, "Voxblox", "C1")

    if gpuf_dir:
        do_plots(plot_gpu_fusion, gpuf_dir, axes, "KinectFusion+", "C2")

    # format with axis labels ---------------------------------

    for fig in [fig_vis, fig_vox]:
        y0 = 0.08 if plot_grads else 0.06
        y1 = 0.76 if plot_grads else 0.7
        y2 = 0.5 if plot_grads else 0.3
        fig.text(
            0.5, y0, 'Sequence time [s]', ha='center',
            fontsize=ax_label_fontsize)
        fig.text(
            0.085, y1, 'SDF error [cm]', va='center',
            rotation='vertical', fontsize=ax_label_fontsize)
        fig.text(
            0.085, y2, 'Collision cost error', va='center',
            rotation='vertical', fontsize=ax_label_fontsize)
        if plot_grads:
            fig.text(
                0.085, 0.24, 'Gradient cosine distance', va='center',
                rotation='vertical', fontsize=ax_label_fontsize)

    for fig in [fig_main, fig_supp]:
        y0 = 0.06 if plot_grads else 0.04
        fig.text(
            0.5, y0, 'Sequence time [s]', ha='center',
            fontsize=ax_label_fontsize)
        y1 = 0.77 if plot_grads else 0.7
        y2 = 0.5 if plot_grads else 0.3
        fig.text(
            0.09, y1, 'SDF error [cm]', va='center', ha='center',
            rotation='vertical', fontsize=ax_label_fontsize)
        fig.text(
            0.09, y2, 'Collision cost error', va='center', ha='center',
            rotation='vertical', fontsize=ax_label_fontsize)
        if plot_grads:
            fig.text(
                0.09, 0.22, 'Gradient cosine distance',
                va='center', ha='center',
                rotation='vertical', fontsize=ax_label_fontsize)

    for fig in [fig_ablt, fig_percp]:
        fig_ablt.text(
            0.5, 0.03, 'Sequence time [s]', ha='center',
            fontsize=ax_label_fontsize)
        fig_ablt.text(
            0.085, 0.5, 'SDF error [cm]', va='center', ha='center',
            rotation='vertical', fontsize=ax_label_fontsize)

    # save ---------------------------------------

    fig_vox.savefig(save_dir + "/vox.png")
    fig_vis.savefig(save_dir + "/vis.png")

    fig_main.savefig(save_dir + "/main_seqs.png")
    fig_supp.savefig(save_dir + "/supp_seqs.png")

    fig_vox_prop.savefig(save_dir + "/vox_prop.png")

    fig_ablt.savefig(save_dir + "/supervision_ablt.png")
    fig_ablt.savefig(save_dir + "/perception_time.png")


if __name__ == "__main__":

    # Directories --------------------------------------------

    incSDF_root = git.Repo(
        '.', search_parent_directories=True).working_tree_dir

    save_dir = incSDF_root + "results/figs/all_seq_plots/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    isdf_dir = incSDF_root + "/results/iSDF/exp0/"
    # voxblox_dir = incSDF_root + "/results/voxblox/5.5cm/"
    # gpuf_dir = incSDF_root + "/results/kinectfusion+/7cm_unocc/"
    voxblox_dir = None
    gpuf_dir = None

    main(
        save_dir,
        isdf_dir,
        gpuf_dir,
        voxblox_dir,
    )
