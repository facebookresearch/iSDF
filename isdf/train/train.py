#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import json
import os
from datetime import datetime
import argparse
import cv2
from torch.utils.tensorboard import SummaryWriter

from isdf import visualisation
from isdf.modules import trainer


def train(
    device,
    config_file,
    chkpt_load_file=None,
    incremental=True,
    # vis
    show_obj=False,
    update_im_freq=50,
    update_mesh_freq=200,
    # save
    save_path=None,
    use_tensorboard=False,
):
    # init trainer-------------------------------------------------------------
    isdf_trainer = trainer.Trainer(
        device,
        config_file,
        chkpt_load_file=chkpt_load_file,
        incremental=incremental,
    )

    # saving init--------------------------------------------------------------
    save = save_path is not None
    if save:
        with open(save_path + "/config.json", "w") as outfile:
            json.dump(isdf_trainer.config, outfile, indent=4)

        if isdf_trainer.save_checkpoints:
            checkpoint_path = os.path.join(save_path, "checkpoints")
            os.makedirs(checkpoint_path)
        if isdf_trainer.save_slices:
            slice_path = os.path.join(save_path, 'slices')
            os.makedirs(slice_path)
            isdf_trainer.write_slices(
                slice_path, prefix="0.000_", include_gt=True)
        if isdf_trainer.save_meshes:
            mesh_path = os.path.join(save_path, 'meshes')
            os.makedirs(mesh_path)

        writer = None
        if use_tensorboard:
            writer = SummaryWriter(save_path)

    # eval init--------------------------------------------------------------
    if isdf_trainer.do_eval:
        res = {}
        if isdf_trainer.sdf_eval:
            res['sdf_eval'] = {}
        if isdf_trainer.mesh_eval:
            res['mesh_eval'] = {}
    if isdf_trainer.do_vox_comparison:
        vox_res = {}

    last_eval = 0

    # live vis init--------------------------------------------------------------
    # if isdf_trainer.live:
    kf_vis = None
    cv2.namedWindow('iSDF keyframes', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("iSDF keyframes", 100, 700)

    # main  loop---------------------------------------------------------------
    print("Starting training for max", isdf_trainer.n_steps, "steps...")
    size_dataset = len(isdf_trainer.scene_dataset)

    break_at = -1

    t = 0
    for _ in range(isdf_trainer.n_steps):
        # break at end -------------------------------------------------------
        if t == break_at and len(isdf_trainer.eval_times) == 0:
            if save:
                if isdf_trainer.save_slices:
                    isdf_trainer.write_slices(slice_path)

                if isdf_trainer.do_eval:
                    kf_list = isdf_trainer.frames.frame_id[:-1].tolist()
                    res['kf_indices'] = kf_list
                    with open(os.path.join(save_path, 'res.json'), 'w') as f:
                        json.dump(res, f, indent=4)

            break

        # get/add data---------------------------------------------------------
        finish_optim = \
            isdf_trainer.steps_since_frame == isdf_trainer.optim_frames
        if incremental and (finish_optim or t == 0):
            # After n steps with new frame, check whether to add it to kf set.
            if t == 0:
                add_new_frame = True
            else:
                add_new_frame = isdf_trainer.check_keyframe_latest()

            if add_new_frame:
                new_frame_id = isdf_trainer.get_latest_frame_id()
                if new_frame_id >= size_dataset:
                    break_at = t + 400
                    print("**************************************",
                          "End of sequence",
                          "**************************************")
                else:
                    print("Total step time", isdf_trainer.tot_step_time)
                    print("frame______________________", new_frame_id)

                    frame_data = isdf_trainer.get_data([new_frame_id])
                    isdf_trainer.add_frame(frame_data)

                    if t == 0:
                        isdf_trainer.last_is_keyframe = True
                        isdf_trainer.optim_frames = 200

            if t == 0 or (isdf_trainer.last_is_keyframe and not add_new_frame):
                kf_vis = visualisation.draw.add_im_to_vis(
                    kf_vis, isdf_trainer.frames.im_batch_np[-1], reduce_factor=6)
                cv2.imshow('iSDF keyframes', kf_vis)
                cv2.waitKey(1)

        # optimisation step---------------------------------------------
        losses, step_time = isdf_trainer.step()
        if not isdf_trainer.live:
            status = [k + ': {:.6f}  '.format(losses[k]) for k in losses.keys()]
            status = "".join(status) + '-- Step time: {:.2f}  '.format(step_time)
            print(t, status)

        if save and writer is not None:
            for key in losses.keys():
                writer.add_scalar("losses/{key}", losses[key], t)

        # visualisation----------------------------------------------------------
        if (
            not isdf_trainer.live and update_im_freq is not None and
            (t % update_im_freq == 0)
        ):
            display = {}
            isdf_trainer.update_vis_vars()
            display["keyframes"] = isdf_trainer.frames_vis()
            # display["slices"] = isdf_trainer.slices_vis()
            if show_obj:
                obj_slices_viz = isdf_trainer.obj_slices_vis()

            if update_mesh_freq is not None and (t % update_mesh_freq == 0):
                scene = isdf_trainer.draw_3D(
                    show_pc=False, show_mesh=t > 200, draw_cameras=True,
                    camera_view=False, show_gt_mesh=False)
                if show_obj:
                    try:
                        obj_scene = isdf_trainer.draw_obj_3D()
                    except:
                        print('Failed to draw mesh')

            display["scene"] = scene
            if show_obj and obj_scene is not None:
                display["obj_scene"] = obj_scene
            if show_obj and obj_slices_viz is not None:
                display["obj_slices"] = obj_slices_viz
            yield display

        t += 1

        # render live view ----------------------------------------------------
        view_freq = 10
        if t % view_freq == 0 and isdf_trainer.live:
            latest_vis = isdf_trainer.latest_frame_vis()
            cv2.imshow('iSDF (frame rgb, depth), (rendered normals, depth)', latest_vis)
            key = cv2.waitKey(5)

            # active keyframes vis
            # kf_active_vis = isdf_trainer.keyframe_vis(reduce_factor=4)
            # cv2.imshow('iSDF keyframes v2', kf_active_vis)
            # cv2.waitKey(1)

            if key == 115:
                # s key to show SDF slices
                isdf_trainer.view_sdf()

            if key == 99:
                # c key clears keyframes
                print('Clearing keyframes...')
                isdf_trainer.clear_keyframes()
                kf_vis = None
                t = 0

        # save ----------------------------------------------------------------
        if save and len(isdf_trainer.save_times) > 0:
            if isdf_trainer.tot_step_time > isdf_trainer.save_times[0]:
                save_t = f"{isdf_trainer.save_times.pop(0):.3f}"
                print(
                    f"Saving at {save_t}s",
                    f" --  model {isdf_trainer.save_checkpoints} ",
                    f"slices {isdf_trainer.save_slices} ",
                    f"mesh {isdf_trainer.save_meshes} "
                )

                if isdf_trainer.save_checkpoints:
                    torch.save(
                        {
                            "step": t,
                            "model_state_dict":
                                isdf_trainer.sdf_map.state_dict(),
                            "optimizer_state_dict":
                                isdf_trainer.optimiser.state_dict(),
                            "loss": losses['total_loss'].item(),
                        },
                        os.path.join(
                            checkpoint_path, "step_" + save_t + ".pth")
                    )

                if isdf_trainer.save_slices:
                    isdf_trainer.write_slices(
                        slice_path, prefix=save_t + "_",
                        include_gt=False, include_diff=False,
                        include_chomp=False, draw_cams=True)

                if isdf_trainer.save_meshes and isdf_trainer.tot_step_time > 0.4:
                    isdf_trainer.write_mesh(mesh_path + f"/{save_t}.ply")

        # evaluation -----------------------------------------------------

        if len(isdf_trainer.eval_times) > 0:
            if isdf_trainer.tot_step_time > isdf_trainer.eval_times[0]:
                eval_t = isdf_trainer.eval_times[0]
                print("voxblox eval at ----------------------------->", eval_t)
                vox_res[isdf_trainer.tot_step_time] = isdf_trainer.eval_fixed()
                if save:
                    with open(os.path.join(save_path, 'vox_res.json'), 'w') as f:
                        json.dump(vox_res, f, indent=4)

        elapsed_eval = isdf_trainer.tot_step_time - last_eval
        if isdf_trainer.do_eval and elapsed_eval > isdf_trainer.eval_freq_s:
            last_eval = isdf_trainer.tot_step_time - \
                isdf_trainer.tot_step_time % isdf_trainer.eval_freq_s

            if isdf_trainer.sdf_eval and isdf_trainer.gt_sdf_file is not None:
                visible_res = isdf_trainer.eval_sdf(visible_region=True)
                obj_errors = isdf_trainer.eval_object_sdf()

                print("Time ---------->", isdf_trainer.tot_step_time)
                print("Visible region SDF error: {:.4f}".format(
                    visible_res["av_l1"]))
                print("Objects SDF error: ", obj_errors)

                if not incremental:
                    full_vol_res = isdf_trainer.eval_sdf(visible_region=False)
                    print("Full region SDF error: {:.4f}".format(
                        full_vol_res["av_l1"]))
                if save:
                    res['sdf_eval'][t] = {
                        'time': isdf_trainer.tot_step_time,
                        'rays': visible_res,
                    }
                    if obj_errors is not None:
                        res['sdf_eval'][t]['objects_l1'] = obj_errors

            if isdf_trainer.mesh_eval:
                acc, comp = isdf_trainer.eval_mesh()
                print("Mesh accuracy and completion:", acc, comp)
                if save:
                    res['mesh_eval'][t] = {
                        'time': isdf_trainer.tot_step_time,
                        'acc': acc,
                        'comp': comp,
                    }

            if save:
                with open(os.path.join(save_path, 'res.json'), 'w') as f:
                    json.dump(res, f, indent=4)
                if writer is not None:
                    writer.add_scalar(
                        "sdf_error_visible/average", visible_res["av_l1"], t)



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description="iSDF.")
    parser.add_argument("--config", type=str, help="input json config")
    parser.add_argument(
        "-ni",
        "--no_incremental",
        action="store_false",
        help="disable incremental SLAM option",
    )
    parser.add_argument(
        "-hd", "--headless",
        action="store_true",
        help="run headless (i.e. no visualisations)"
    )
    args, _ = parser.parse_known_args()  # ROS adds extra unrecongised args

    config_file = args.config
    headless = args.headless
    incremental = args.no_incremental
    chkpt_load_file = None

    # vis
    show_obj = False
    update_im_freq = 40
    update_mesh_freq = 200
    if headless:
        update_im_freq = None
        update_mesh_freq = None

    # save
    save = False
    use_tensorboard = False
    if save:
        now = datetime.now()
        time_str = now.strftime("%m-%d-%y_%H-%M-%S")
        save_path = "../../results/iSDF/" + time_str
        os.mkdir(save_path)
    else:
        save_path = None

    scenes = train(
        device,
        config_file,
        chkpt_load_file=chkpt_load_file,
        incremental=incremental,
        # vis
        show_obj=show_obj,
        update_im_freq=update_im_freq,
        update_mesh_freq=update_mesh_freq,
        # save
        save_path=save_path,
        use_tensorboard=use_tensorboard,
    )

    if headless:
        on = True
        while on:
            try:
                out = next(scenes)
            except StopIteration:
                on = False

    else:
        n_cols = 2
        if show_obj:
            n_cols = 3
        tiling = (1, n_cols)
        visualisation.display.display_scenes(
            scenes, height=int(680 * 0.7), width=int(1200 * 0.7), tile=tiling
        )
