import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
from shutil import copyfile
from plyfile import PlyData, PlyElement

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from .data.scannet.model_util_scannet import ScannetDatasetConfig
from .lib.visual_question_answering.dataset import ScanVQADataset
from .lib.visual_question_answering.solver_3dgqa import Solver
from .lib.configs.config_vqa import CONF
from .models.vqanet.vqanet import VqaNet
from .scripts.utils.AdamW import AdamW
from .scripts.utils.script_utils import set_params_lr_dict
# import crash_on_ipy
from .lib.ap_helper.ap_helper_fcos import APCalculator, parse_predictions, parse_groundtruths
from .lib.loss_helper.loss_vqa import get_loss
from .lib.visual_question_answering.eval_helper import get_eval
# from .data.scannet.model_util_scannet import ScannetDatasetConfig
# from lib.visual_question_answering.dataset import ScanVQADataset
# from lib.visual_question_answering.solver_v0 import Solver
# from lib.config_vqa import CONF
# from models.vqanet.vqanet_v6 import VqaNet
# from scripts.utils.AdamW import AdamW
# from scripts.utils.script_utils import set_params_lr_dict
# import crash_on_ipy


# constants
DC = ScannetDatasetConfig()

SCANVQA_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_filter_v3.0.json")))

SCANVQA_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_filter_v3.0.json")))  # UNSEEN
SCANVQA_ANSWER_LIST = []
SCANVQA_ANSWER_LIST += [data["answer"] for data in SCANVQA_TRAIN]
SCANVQA_ANSWER_LIST += [data["answer"] for data in SCANVQA_VAL]
SCANVQA_ANSWER_LIST = sorted(list(set(SCANVQA_ANSWER_LIST)))

def get_dataloader(args, scanvqa, scene_list, split, config):
    dataset = ScanVQADataset(
        scanvqa_data=scanvqa,
        scanvqa_all_scene=scene_list,
        answer_type=SCANVQA_ANSWER_LIST,
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        lang_num_max=args.lang_num_max
    )
    print("evaluate on {} samples".format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, collate_fn=dataset.collate_fn)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

    return dataset, dataloader

def get_model(args):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = VqaNet(
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        no_reference=args.no_reference,
        dataset_config=DC
    ).cuda()

    model_name = "model_last.pth" if args.last_ckpt else "model.pth"
    # path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    path = os.path.join(args.folder, model_name)
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(args):
    if args.detection:
        scene_list = get_scannet_scene_list("val")
        scanrefer = []
        for scene_id in scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            scanrefer.append(data)
    else:
        if args.dataset == 'ScanRefer_filtered':
            SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
            SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
        elif args.dataset == 'nr3d':
            SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
            SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d.json")))
        else:
            raise NotImplementedError()

        scanrefer = SCANREFER_TRAIN if args.use_train else SCANREFER_VAL
        scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
        scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]
        new_data = []
        prefix = args.dataset
        for value in scanrefer:
            current_label = {
                'source': f'{prefix} dataset based',
                'scene_id': value['scene_id'],
                'question_type': 'grounding',
                'question': value['description'],
                'answer': ' '.join(value['object_name'].split('_')),
                'Grounding in Query': [int(value['object_id'])],  # todo
                'Grounding in Answer': [],  # todo
                'rank(filter)': 'A',
                'issue(filter)': 'template based',
                'ann_id': value['ann_id'],
                'object_id': value['object_id'],
                'object_name': value['object_name']
            }
            new_data.append(current_label)
            # current_label = {
            #     'source': f'{prefix} dataset based',
            #     'scene_id': value['scene_id'],
            #     'question_type': 'grounding',
            #     'question': value['description'].replace(value['object_name'], '[mask]'),
            #     'answer': value['object_name'],
            #     'Grounding in Query': [],  # todo
            #     'Grounding in Answer': [value['object_id']],  # todo
            #     'rank(filter)': 'A',
            #     'issue(filter)': 'template based'
            # }
            # new_data.append(current_label)
        scanrefer = new_data
    return scanrefer, scene_list

def get_scanvqa(args):
    # get initial scene list
    train_scene_list = get_scannet_scene_list("train")
    val_scene_list = get_scannet_scene_list("val")
    # train_scene_list = sorted(list(set([data["scene_id"] for data in scanvqa_train])))
    # val_scene_list = sorted(list(set([data["scene_id"] for data in scanvqa_val])))
    # slice train_scene_list
    all_scene_list = train_scene_list + val_scene_list
    scanvqa_train, scanvqa_val = SCANVQA_TRAIN, SCANVQA_VAL
    scanvqa_train = [value for value in scanvqa_train if value["scene_id"] in train_scene_list]
    scanvqa_val = [value for value in scanvqa_val if value["scene_id"] in val_scene_list]
    print("train on {} samples and val on {} samples".format(len(scanvqa_train), len(scanvqa_val)))
    return scanvqa_train, scanvqa_val, train_scene_list, val_scene_list, all_scene_list


def eval_ref(args):
    print("evaluate localization...")
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "val", DC)

    # model
    model = get_model(args)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    } if not args.no_nms else None

    # random seeds
    seeds = [args.seed] + [2 * i for i in range(args.repeat - 1)]

    # evaluate
    print("evaluating...")
    # score_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "scores.p")
    # pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "predictions.p")
    score_path = os.path.join(args.folder, "scores.p")
    pred_path = os.path.join(args.folder, "predictions.p")
    gen_flag = (not os.path.exists(score_path)) or args.force or args.repeat > 1
    if gen_flag:
        ref_acc_all = []
        ious_all = []
        masks_all = []
        others_all = []
        lang_acc_all = []
        for seed in seeds:
            # reproducibility
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)

            print("generating the scores for seed {}...".format(seed))
            ref_acc = []
            ious = []
            masks = []
            others = []
            lang_acc = []
            predictions = {}
            for idx, data in enumerate(tqdm(dataloader)):
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].cuda()

                # feed
                with torch.no_grad():
                    data["epoch"] = 0
                    data = model(data)
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    data = get_loss(
                        data_dict=data,
                        config=DC,
                        detection=True,
                        qa=True,
                        use_lang_classifier=True
                    )

                    data = get_eval(
                        data_dict=data,
                        config=DC,
                        reference=True,
                        use_lang_classifier=not args.no_lang_cls,
                        scanrefer_eval=True,
                        # use_oracle=args.use_oracle,
                        # use_cat_rand=args.use_cat_rand,
                        # use_best=args.use_best,
                        # post_processing=POST_DICT
                    )

                    ref_acc += data["ref_acc"]
                    ious += data["ref_iou"]
                    masks += data["ref_multiple_mask"]
                    others += data["ref_others_mask"]
                    lang_acc.append(data["lang_acc"].item())

                    # store predictions
                    ids = data["scan_idx"].detach().cpu().numpy()
                    for i in range(ids.shape[0]):
                        idx = ids[i]
                        scene_id = scanrefer[idx]["scene_id"]
                        object_id = scanrefer[idx]["object_id"]
                        ann_id = scanrefer[idx]["ann_id"]

                        if scene_id not in predictions:
                            predictions[scene_id] = {}

                        if object_id not in predictions[scene_id]:
                            predictions[scene_id][object_id] = {}

                        if ann_id not in predictions[scene_id][object_id]:
                            predictions[scene_id][object_id][ann_id] = {}

                        predictions[scene_id][object_id][ann_id]["pred_bbox"] = data["pred_bboxes"][i]
                        predictions[scene_id][object_id][ann_id]["gt_bbox"] = data["gt_bboxes"][i]
                        predictions[scene_id][object_id][ann_id]["iou"] = data["ref_iou"][i]

            # save the last predictions
            with open(pred_path, "wb") as f:
                pickle.dump(predictions, f)

            # save to global
            ref_acc_all.append(ref_acc)
            ious_all.append(ious)
            masks_all.append(masks)
            others_all.append(others)
            lang_acc_all.append(lang_acc)

        # convert to numpy array
        ref_acc = np.array(ref_acc_all)
        ious = np.array(ious_all)
        masks = np.array(masks_all)
        others = np.array(others_all)
        lang_acc = np.array(lang_acc_all)

        # save the global scores
        with open(score_path, "wb") as f:
            scores = {
                "ref_acc": ref_acc_all,
                "ious": ious_all,
                "masks": masks_all,
                "others": others_all,
                "lang_acc": lang_acc_all
            }
            pickle.dump(scores, f)

    else:
        print("loading the scores...")
        with open(score_path, "rb") as f:
            scores = pickle.load(f)

            # unpack
            ref_acc = np.array(scores["ref_acc"])
            ious = np.array(scores["ious"])
            masks = np.array(scores["masks"])
            others = np.array(scores["others"])
            lang_acc = np.array(scores["lang_acc"])

    multiple_dict = {
        "unique": 0,
        "multiple": 1
    }
    others_dict = {
        "not_in_others": 0,
        "in_others": 1
    }

    # evaluation stats
    stats = {k: np.sum(masks[0] == v) for k, v in multiple_dict.items()}
    stats["overall"] = masks[0].shape[0]
    stats = {}
    for k, v in multiple_dict.items():
        stats[k] = {}
        for k_o, v_o in others_dict.items():
            stats[k][k_o] = np.sum(np.logical_and(masks[0] == v, others[0] == v_o))
        stats[k]["overall"] = np.sum(masks[0] == v)

    stats["overall"] = {}
    for k_o, v_o in others_dict.items():
        stats["overall"][k_o] = np.sum(others[0] == v_o)
    stats["overall"]["overall"] = masks[0].shape[0]

    # aggregate scores
    scores = {}
    for k, v in multiple_dict.items():
        for k_o in others_dict.keys():
            ref_accs, acc_025ious, acc_05ious = [], [], []
            for i in range(masks.shape[0]):
                running_ref_acc = np.mean(ref_acc[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])]) \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_025iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.25)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_05iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.5)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0

                # store
                ref_accs.append(running_ref_acc)
                acc_025ious.append(running_acc_025iou)
                acc_05ious.append(running_acc_05iou)

            if k not in scores:
                scores[k] = {k_o: {} for k_o in others_dict.keys()}

            scores[k][k_o]["ref_acc"] = np.mean(ref_accs)
            scores[k][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
            scores[k][k_o]["acc@0.5iou"] = np.mean(acc_05ious)

        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][masks[i] == multiple_dict[k]]) if np.sum(masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.25)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.5)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        scores[k]["overall"] = {}
        scores[k]["overall"]["ref_acc"] = np.mean(ref_accs)
        scores[k]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
        scores[k]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    scores["overall"] = {}
    for k_o in others_dict.keys():
        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][others[i] == others_dict[k_o]]) if np.sum(others[i] == others_dict[k_o]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.25)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.5)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        # aggregate
        scores["overall"][k_o] = {}
        scores["overall"][k_o]["ref_acc"] = np.mean(ref_accs)
        scores["overall"][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
        scores["overall"][k_o]["acc@0.5iou"] = np.mean(acc_05ious)
   
    ref_accs, acc_025ious, acc_05ious = [], [], []
    for i in range(masks.shape[0]):
        running_ref_acc = np.mean(ref_acc[i])
        running_acc_025iou = ious[i][ious[i] >= 0.25].shape[0] / ious[i].shape[0]
        running_acc_05iou = ious[i][ious[i] >= 0.5].shape[0] / ious[i].shape[0]

        # store
        ref_accs.append(running_ref_acc)
        acc_025ious.append(running_acc_025iou)
        acc_05ious.append(running_acc_05iou)

    # aggregate
    scores["overall"]["overall"] = {}
    scores["overall"]["overall"]["ref_acc"] = np.mean(ref_accs)
    scores["overall"]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
    scores["overall"]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    # report
    print("\nstats:")
    for k_s in stats.keys():
        for k_o in stats[k_s].keys():
            print("{} | {}: {}".format(k_s, k_o, stats[k_s][k_o]))

    for k_s in scores.keys():
        print("\n{}:".format(k_s))
        for k_m in scores[k_s].keys():
            for metric in scores[k_s][k_m].keys():
                print("{} | {} | {}: {}".format(k_s, k_m, metric, scores[k_s][k_m][metric]))

    print("\nlanguage classification accuracy: {}".format(np.mean(lang_acc)))

def eval_vqa(args):
    print("evaluate localization...")
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    scanvqa_train, scanvqa_val, train_scene_list, val_scene_list, all_scene_list = get_scanvqa(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanvqa_val, val_scene_list, "val", DC)

    # model
    model = get_model(args)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    } if not args.no_nms else None

    # random seeds
    seeds = [args.seed] + [2 * i for i in range(args.repeat - 1)]

    # evaluate
    print("evaluating...")
    # score_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "scores_vqa.p")
    # pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "predictions_vqa.p")
    score_path = os.path.join(args.folder, "scores_vqa.p")
    pred_path = os.path.join(args.folder, "predictions_vqa.p")
    gen_flag = (not os.path.exists(score_path)) or args.force or args.repeat > 1
    if gen_flag:
        answer_acc_all = []
        type_acc_all = []
        mAP50_all = []
        mAP25_all = []
        for seed in seeds:
            # reproducibility
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)

            print("generating the scores for seed {}...".format(seed))
            answer_acc = []
            test_type_acc = [[] for i in range(4)]
            for idx, data in enumerate(tqdm(dataloader)):
            # for data in tqdm(dataloader):
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].cuda()
                # feed
                with torch.no_grad():
                    data["epoch"] = 0
                    data = model(data)
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    data = get_loss(
                        data_dict=data,
                        config=DC,
                        detection=True,
                        qa=True,
                        use_lang_classifier=True
                    )

                    data = get_eval(
                        data_dict=data,
                        config=DC,
                        reference=True,
                        use_lang_classifier=not args.no_lang_cls,
                        scanrefer_eval=False,
                        refresh_map=1 if idx == len(dataloader)-1 else -1,
                        eval25=True,
                        # use_oracle=args.use_oracle,
                        # use_cat_rand=args.use_cat_rand,
                        # use_best=args.use_best,
                        # post_processing=POST_DICT
                    )
                    # import ipdb
                    # ipdb.set_trace()

                    answer_acc += data["answer_acc"]
                    for i in range(4):
                        test_type_acc[i] += data['test_type_acc'][i]
                    # masks += data["ref_multiple_mask"]
                    mAP50_metrics = data["mAP50_metrics"]
                    mAP25_metrics = data["mAP25_metrics"]

                    # store predictions
                    ids = data["scan_idx"].detach().cpu().numpy()

            # # save the last predictions
            # with open(pred_path, "wb") as f:
            #     pickle.dump(predictions, f)
            answer_acc = sum(answer_acc) / len(answer_acc)
            test_type_acc = [sum(val) / len(val) for val in test_type_acc]

            print('answer_acc:', answer_acc)
            print('test_type_acc(number, other, yes/no, other):', test_type_acc)
            print('mAP(0.25)', mAP25_metrics['mAP'], mAP25_metrics)
            print('mAP(0.5)', mAP50_metrics['mAP'], mAP50_metrics)

            # save to global
            answer_acc_all.append(answer_acc)
            type_acc_all.append(test_type_acc)
            mAP25_all.append(mAP25_metrics)
            mAP50_all.append(mAP50_metrics)

        # # convert to numpy array
        # ref_acc = np.array(ref_acc_all)
        # ious = np.array(ious_all)
        # masks = np.array(masks_all)
        # # save the global scores
        # with open(score_path, "wb") as f:
        #     scores = {
        #         "ref_acc": ref_acc_all,
        #         "ious": ious_all,
        #         "masks": masks_all,
        #         "others": others_all,
        #         "lang_acc": lang_acc_all
        #     }
        #     pickle.dump(scores, f)

    else:
        print("loading the scores...")
        raise NotImplementedError()
        with open(score_path, "rb") as f:
            scores = pickle.load(f)

            # unpack
            ref_acc = np.array(scores["ref_acc"])
            ious = np.array(scores["ious"])
            masks = np.array(scores["masks"])
            others = np.array(scores["others"])
            lang_acc = np.array(scores["lang_acc"])



def eval_det(args):
    print("evaluate detection...")
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "val", DC)

    # model
    model = get_model(args, DC)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    }
    AP_IOU_THRESHOLDS = [0.25, 0.5]
    AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    sem_acc = []
    for data in tqdm(dataloader):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()
        # feed
        with torch.no_grad():
            data = model(data)
            data = get_loss(
                data_dict=data,
                config=DC,
                detection=True,
                qa=False,
                use_lang_classifier=False
            )

            data = get_eval(
                data_dict=data,
                config=DC,
                reference=False,
                use_lang_classifier=not args.no_lang_cls,
                scanrefer_eval=False,
                # use_oracle=args.use_oracle,
                # use_cat_rand=args.use_cat_rand,
                # use_best=args.use_best,
                # post_processing=POST_DICT
            )

            sem_acc.append(data["sem_acc"].item())

            batch_pred_map_cls = parse_predictions(data, POST_DICT) 
            batch_gt_map_cls = parse_groundtruths(data, POST_DICT) 
            for ap_calculator in AP_CALCULATOR_LIST:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # aggregate object detection results and report
    print("\nobject detection sem_acc: {}".format(np.mean(sem_acc)))
    for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
        print()
        print("-"*10, "iou_thresh: %f"%(AP_IOU_THRESHOLDS[i]), "-"*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            print("eval %s: %f"%(key, metrics_dict[key]))

class GroudingQuestionAnswering(nn.Module):
    def __init__(self):
        super().__init__()

    def get_dataloader(self, args, scanvqa, scene_list, split, config, augment, shuffle=True):
        dataset = ScanVQADataset(
            scanvqa_data=scanvqa[split],
            scanvqa_all_scene=scene_list,
            answer_type=self.SCANVQA_ANSWER_LIST,
            split=split,
            num_points=args.num_points,
            use_height=(not args.no_height),
            use_color=args.use_color,
            use_normal=args.use_normal,
            use_multiview=args.use_multiview,
            lang_num_max=args.lang_num_max,
            augment=augment,
            shuffle=shuffle
        )
        # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, collate_fn=dataset.collate_fn)

        return dataset, dataloader

    def get_model(self, args):
        # initiate model
        input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
        model = VqaNet(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            input_feature_dim=input_channels,
            num_proposal=args.num_proposals,
            use_lang_classifier=(not args.no_lang_cls),
            no_reference=args.no_reference,
            dataset_config=DC
        )

        # trainable model
        if args.use_pretrained:
            # load model
            print("loading pretrained VoteNet...")

            pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
            load_result = model.load_state_dict(torch.load(pretrained_path), strict=False)
            print(load_result, flush=True)

            # mount
            # model.backbone_net = pretrained_model.backbone_net
            # model.vgen = pretrained_model.vgen
            # model.proposal = pretrained_model.proposal

            if args.no_detection:
                # freeze pointnet++ backbone
                for param in model.backbone_net.parameters():
                    param.requires_grad = False

                # freeze voting
                for param in model.vgen.parameters():
                    param.requires_grad = False

                # freeze detector
                for param in model.proposal.parameters():
                    param.requires_grad = False

        # to CUDA
        model = model.cuda()

        return model


    def get_num_params(self, model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

        return num_params

    def get_solver(self, args, dataloader):
        model = self.get_model(args)
        weight_dict = {
            # 'encoder': {'lr': 0.000001},
            # 'decoder': {'lr': 0.000001},
            'lang': {'lr': 0.0001},
            'relation': {'lr': 0.0001},
            'crossmodal': {'lr': 0.0001},
        }
        params = set_params_lr_dict(model, base_lr=args.lr, weight_decay=args.wd, weight_dict=weight_dict)
        # params = model.parameters()
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)

        # CONF.PATH.OUTPUT = os.path.join(CONF.PATH.OUTPUT, 'visual_question_answering')
        if args.use_checkpoint:
            print("loading checkpoint {}...".format(args.use_checkpoint))
            stamp = args.use_checkpoint
            root = os.path.join(CONF.PATH.OUTPUT, stamp)
            checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if args.tag: stamp += "_"+args.tag.upper()
            root = os.path.join(CONF.PATH.OUTPUT, stamp)
            os.makedirs(root, exist_ok=True)

        # scheduler parameters for training solely the detection pipeline
        if args.coslr:
            lr_args = {
                'type': 'cosine',
                'T_max': args.epoch,
                'eta_min': 1e-5,
            }
        else:
            lr_args = None
        if args.no_reference:
            bn_args = {
                'step': 20,
                'rate': 0.5
            }
        else:
            bn_args = None

        print('learning rate and batch norm args', lr_args, bn_args, flush=True)
        solver = Solver(
            model=model,
            config=DC,
            dataloader=dataloader,
            optimizer=optimizer,
            stamp=stamp,
            val_step=args.val_step,
            detection=not args.no_detection,
            reference=not args.no_reference,
            use_lang_classifier=not args.no_lang_cls,
            lr_args=lr_args,
            bn_args=bn_args,
        )
        num_params = self.get_num_params(model)

        return solver, num_params, root


    def save_info(self, args, root, num_params, train_dataset, val_dataset):
        info = {}
        for key, value in vars(args).items():
            info[key] = value

        info["num_train"] = len(train_dataset)
        info["num_val"] = len(val_dataset)
        info["num_train_scenes"] = len(train_dataset.scene_list)
        info["num_val_scenes"] = len(val_dataset.scene_list)
        info["num_params"] = num_params

        with open(os.path.join(root, "info.json"), "w") as f:
            json.dump(info, f, indent=4)


    def get_scannet_scene_list(self, split):
        scene_list = sorted(
            [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

        return scene_list

    # def get_splited_data(scanvqa_data, scene_list, lang_num_max):
    #     # filter data in chosen scenes
    #     new_scanvqa = []
    #     scanvqa = []
    #     scene_id = ""
    #     for data in scanvqa_data:
    #         if data["scene_id"] in scene_list:
    #             if scene_id != data["scene_id"]:
    #                 scene_id = data["scene_id"]
    #                 if len(new_scanvqa) > 0:
    #                     scanvqa.append(new_scanvqa)
    #                     new_scanvqa = []
    #             if len(new_scanvqa) == lang_num_max:
    #                 scanvqa.append(new_scanvqa)
    #                 new_scanvqa = []
    #             new_scanvqa.append(data)
    #     if len(new_scanvqa) > 0:
    #         scanvqa.append(new_scanvqa)
    #     return scanvqa

    def get_scanvqa(self, scanvqa_train, scanvqa_val, num_scenes, lang_num_max):
        # get initial scene list
        train_scene_list = self.get_scannet_scene_list("train")
        val_scene_list = self.get_scannet_scene_list("val")
        # train_scene_list = sorted(list(set([data["scene_id"] for data in scanvqa_train])))
        # val_scene_list = sorted(list(set([data["scene_id"] for data in scanvqa_val])))
        if num_scenes == -1:
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes

        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]
        all_scene_list = train_scene_list + val_scene_list

        scanvqa_train = [value for value in scanvqa_train if value["scene_id"] in train_scene_list]
        scanvqa_val = [value for value in scanvqa_val if value["scene_id"] in val_scene_list]
        # scanvqa_train = get_splited_data(scanvqa_train, train_scene_list, lang_num_max)
        # scanvqa_val = get_splited_data(scanvqa_val, val_scene_list, lang_num_max)

        # print("scanvqa_iter_number", len(scanvqa_train), len(scanvqa_val), 'lang_per_iter', lang_num_max)
        # # sum = 0
        # # for i in range(len(scanvqa_train)):
        # #     sum += len(scanvqa_train[i])
        # #     # print(len(scanvqa_train_new[i]))
        # # # for i in range(len(scanvqa_val_new)):
        # # #    print(len(scanvqa_val_new[i]))
        # # print("train data number", sum)  # 1418 363
        # # all scanvqa scene
        # print(sum([len(data) for data in scanvqa_train]))
        # print("train on {} samples and val on {} samples".format( \
        #     sum([len(data) for data in scanvqa_train]),
        #     sum([len(data) for data in scanvqa_val])
        # ))

        print("train on {} samples and val on {} samples".format(len(scanvqa_train), len(scanvqa_val)))
        return scanvqa_train, scanvqa_val, train_scene_list, val_scene_list, all_scene_list


    def train(self, args):
        # init training dataset
        print("preparing data...")
        scanvqa_train, scanvqa_val, all_scene_list, train_scene_list, val_scene_list = \
            self.get_scanvqa(self.SCANVQA_TRAIN, self.SCANVQA_VAL, args.num_scenes, args.lang_num_max)

        scanvqa = {
            "train": scanvqa_train,
            "val": scanvqa_val
        }

        # dataloade
        train_dataset, train_dataloader = self.get_dataloader(args, scanvqa, train_scene_list, "train", DC, augment=True, shuffle=True)
        val_dataset, val_dataloader = self.get_dataloader(args, scanvqa, val_scene_list, "val", DC, augment=False, shuffle=False)
        dataloader = {
            "train": train_dataloader,
            "val": val_dataloader
        }

        print("initializing...")
        solver, num_params, root = self.get_solver(args, dataloader)

        print("Start training...\n")
        self.save_info(args, root, num_params, train_dataset, val_dataset)
        solver(args.epoch, args.verbose)

    def run_train(self, data_path):
        parser = argparse.ArgumentParser()
        parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="3dgqa")
        parser.add_argument("--gpu", type=str, help="gpu", default="0")
        parser.add_argument("--batch_size", type=int, help="batch size", default=8)
        parser.add_argument("--epoch", type=int, help="number of epochs", default=200)
        parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=50)
        parser.add_argument("--val_step", type=int, help="iterations of validating", default=1000)
        parser.add_argument("--lr", type=float, help="learning rate", default=2e-3)
        parser.add_argument("--wd", type=float, help="weight decay", default=1e-3)
        parser.add_argument("--lang_num_max", type=int, help="lang num max", default=8)
        parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
        parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
        parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
        parser.add_argument("--seed", type=int, default=42, help="random seed")
        parser.add_argument("--coslr", action='store_true', help="cosine learning rate", default=True)
        parser.add_argument("--amsgrad", action='store_true', help="optimizer with amsgrad")

        parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.", default=True)
        parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
        parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
        parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
        parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.", default=True)
        parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.", default=False)
        parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.", default=False)
        parser.add_argument("--use_pretrained", type=str,
                            help="Specify the folder name containing the pretrained detection module.")
        parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
        parser.add_argument("--use_generated_dataset", action="store_true", help="Use Generated Dataset.")
        parser.add_argument("--use_generated_masked_dataset", action="store_true", help="Use Generated Masked Dataset.")
        args = parser.parse_args()

        # setting
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # reproducibility
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        
        # training dataset
        CONF.PATH.DATA = data_path
        base_path = os.path.dirname(os.path.abspath(__file__))
        output_path = data_path[:-4]+'outputs'
        with open(base_path+'/lib/configs/config.py', 'r+') as cf:
            content = cf.readlines()
            content[12] = f'CONF.PATH.DATA = "{data_path}"\n'
            content[53] = f'CONF.PATH.OUTPUT = "{output_path}"\n'
        with open(base_path+'/lib/configs/config.py', 'w+') as cf:
            cf.writelines(content)
        # print('----------------------------------------------------------------')
        CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
        CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
        CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
        CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")
        CONF.PATH.OUTPUT = output_path

        # append to syspath
        # for _, path in CONF.PATH.items():
        #     sys.path.append(path)
        print(sys.path, 'sys path', flush=True)

        # scannet data
        CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
        CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
        CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")

        # Scan2CAD
        CONF.PATH.SCAN2CAD = os.path.join(CONF.PATH.DATA, "Scan2CAD_dataset") # TODO change this

        # data
        CONF.SCANNET_DIR =  CONF.PATH.DATA + "/scannet/scans" # TODO change this
        CONF.SCANNET_FRAMES_ROOT = CONF.PATH.DATA + "/frame_square/" # TODO change this
        CONF.PROJECTION = CONF.PATH.DATA + "/multiview_projection_scanrefer" # TODO change this
        CONF.ENET_FEATURES_ROOT = CONF.PATH.DATA + "/enet_features" # TODO change this

        CONF.ENET_FEATURES_SUBROOT = os.path.join(CONF.ENET_FEATURES_ROOT, "{}") # scene_id
        CONF.ENET_FEATURES_PATH = os.path.join(CONF.ENET_FEATURES_SUBROOT, "{}.npy") # frame_id
        CONF.SCANNET_FRAMES = os.path.join(CONF.SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode
        # CONF.SCENE_NAMES = sorted(os.listdir(CONF.SCANNET_DIR))
        CONF.ENET_WEIGHTS = os.path.join(CONF.PATH.BASE, "data/scannetv2_enet.pth")
        # CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
        CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats_maxpool.hdf5")
        CONF.NYU40_LABELS = os.path.join(CONF.PATH.SCANNET_META, "nyu40_labels.csv")

        # scannet
        CONF.SCANNETV2_TRAIN = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_train.txt")
        CONF.SCANNETV2_VAL = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_val.txt")
        CONF.SCANNETV2_TEST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_test.txt")
        CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")

        # output
        CONF.PATH.AXIS_ALIGNED_MESH = os.path.join(CONF.PATH.OUTPUT, "ScanNet_axis_aligned_mesh")

        # pretrained
        CONF.PATH.PRETRAINED = os.path.join(CONF.PATH.BASE, "pretrained")

        # Pretrained features
        CONF.PATH.GT_FEATURES = os.path.join(CONF.PATH.CLUSTER, "gt_{}_features") # dataset
        # CONF.PATH.VOTENET_FEATURES = os.path.join(CONF.PATH.CLUSTER, "votenet_features")
        CONF.PATH.VOTENET_FEATURES = os.path.join(CONF.PATH.CLUSTER, "votenet_{}_predictions") # dataset
        
        self.SCANVQA_TRAIN = []
        self.SCANVQA_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_filter_v3.0.json")))
        # more dataset
        self.SCANVQA_MORE = []
        # self.SCANVQA_MORE += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_generated.json")))
        # self.SCANVQA_MORE += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanRefer_filtered_generated.json")))
        # self.SCANVQA_MORE += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/nr3d_generated.json")))
        # template-based; dataset-size x 2
        self.SCANVQA_MASK = []
        self.SCANVQA_MASK += self.SCANVQA_TRAIN
        # self.SCANVQA_MASK += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_generated.json")))
        # self.SCANVQA_MASK += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanRefer_filtered_generated_masked.json")))
        # self.SCANVQA_MASK += json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/nr3d_generated_masked.json")))

        # split via scene id
        # validation dataset
        self.SCANVQA_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanVQA/ScanVQA_filter_v3.0.json")))

        self.SCANVQA_ANSWER_LIST = []  # calculate during training

        # add generated dataset
        if args.use_generated_dataset:
            self.SCANVQA_TRAIN += self.SCANVQA_MORE
        if args.use_generated_masked_dataset:  # dataset size * 2; so training epoch /= 2
            self.SCANVQA_TRAIN += self.SCANVQA_MASK
            args.epoch //= 2
        self.SCANVQA_ANSWER_LIST += [data["answer"] for data in self.SCANVQA_TRAIN]
        self.SCANVQA_ANSWER_LIST += [data["answer"] for data in self.SCANVQA_VAL]
        self.SCANVQA_ANSWER_LIST = sorted(list(set(self.SCANVQA_ANSWER_LIST)))

        self.train(args)
    
    def inference(self, model_path, data_path):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer_filtered of nr3d", default="ScanRefer_filtered")
        parser.add_argument("--folder", type=str, help="Folder containing the model")
        parser.add_argument("--gpu", type=str, help="gpu", default="0")
        parser.add_argument("--batch_size", type=int, help="batch size", default=8)
        parser.add_argument("--lang_num_max", type=int, help="lang num max", default=1)
        parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
        parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
        parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
        parser.add_argument("--force", action="store_true", help="enforce the generation of results", default=True)  # Not Useful
        parser.add_argument("--seed", type=int, default=42, help="random seed")
        parser.add_argument("--repeat", type=int, default=5, help="Number of times for evaluation")
        parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.", default=True)
        parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
        parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.", default=True)
        parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.", default=True)
        parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.", default=True)
        parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.", default=False)
        parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.", default=False)
        parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
        parser.add_argument("--use_oracle", action="store_true", help="Use ground truth bounding boxes.")
        parser.add_argument("--use_cat_rand", action="store_true", help="Use randomly selected bounding boxes from correct categories as outputs.")
        parser.add_argument("--use_best", action="store_true", help="Use best bounding boxes as outputs.")
        parser.add_argument("--reference", action="store_true", help="evaluate the reference localization results", default=True)
        parser.add_argument("--detection", action="store_true", help="evaluate the object detection results")
        parser.add_argument("--vqa", action="store_true", help="evaluate the vqa results", default=True)
        parser.add_argument("--last_ckpt", action="store_true", help="evaluate the last_ckpt results")

        args = parser.parse_args()
        args.folder = model_path
        
        CONF.PATH.DATA = data_path
        base_path = os.path.dirname(os.path.abspath(__file__))
        output_path = data_path[:-4]+'outputs'
        with open(base_path+'/lib/configs/config.py', 'r+') as cf:
            content = cf.readlines()
            content[12] = f'CONF.PATH.DATA = "{data_path}"\n'
            content[53] = f'CONF.PATH.OUTPUT = "{output_path}"\n'
        with open(base_path+'/lib/configs/config.py', 'w+') as cf:
            cf.writelines(content)
        # print('----------------------------------------------------------------')
        CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
        CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
        CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
        CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")
        CONF.PATH.OUTPUT = output_path

        # append to syspath
        # for _, path in CONF.PATH.items():
        #     sys.path.append(path)
        print(sys.path, 'sys path', flush=True)

        # scannet data
        CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
        CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
        CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")

        # Scan2CAD
        CONF.PATH.SCAN2CAD = os.path.join(CONF.PATH.DATA, "Scan2CAD_dataset") # TODO change this

        # data
        CONF.SCANNET_DIR =  CONF.PATH.DATA + "/scannet/scans" # TODO change this
        CONF.SCANNET_FRAMES_ROOT = CONF.PATH.DATA + "/frame_square/" # TODO change this
        CONF.PROJECTION = CONF.PATH.DATA + "/multiview_projection_scanrefer" # TODO change this
        CONF.ENET_FEATURES_ROOT = CONF.PATH.DATA + "/enet_features" # TODO change this

        CONF.ENET_FEATURES_SUBROOT = os.path.join(CONF.ENET_FEATURES_ROOT, "{}") # scene_id
        CONF.ENET_FEATURES_PATH = os.path.join(CONF.ENET_FEATURES_SUBROOT, "{}.npy") # frame_id
        CONF.SCANNET_FRAMES = os.path.join(CONF.SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode
        # CONF.SCENE_NAMES = sorted(os.listdir(CONF.SCANNET_DIR))
        CONF.ENET_WEIGHTS = os.path.join(CONF.PATH.BASE, "data/scannetv2_enet.pth")
        # CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
        CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats_maxpool.hdf5")
        CONF.NYU40_LABELS = os.path.join(CONF.PATH.SCANNET_META, "nyu40_labels.csv")

        # scannet
        CONF.SCANNETV2_TRAIN = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_train.txt")
        CONF.SCANNETV2_VAL = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_val.txt")
        CONF.SCANNETV2_TEST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_test.txt")
        CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")

        # output
        CONF.PATH.AXIS_ALIGNED_MESH = os.path.join(CONF.PATH.OUTPUT, "ScanNet_axis_aligned_mesh")

        # pretrained
        CONF.PATH.PRETRAINED = os.path.join(CONF.PATH.BASE, "pretrained")

        # Pretrained features
        CONF.PATH.GT_FEATURES = os.path.join(CONF.PATH.CLUSTER, "gt_{}_features") # dataset
        # CONF.PATH.VOTENET_FEATURES = os.path.join(CONF.PATH.CLUSTER, "votenet_features")
        CONF.PATH.VOTENET_FEATURES = os.path.join(CONF.PATH.CLUSTER, "votenet_{}_predictions") # dataset

        if args.reference:
            assert args.lang_num_max == 1, 'lang max num == 1; avoid bugs'
        # # setting
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # evaluate
        if args.reference: eval_ref(args)
        if args.detection: eval_det(args)
        if args.vqa: eval_vqa(args)


if __name__ == '__main__':
    m = GroudingQuestionAnswering()
    # m.run_train('/data2/wangzhen/3DVL_Codebase/data')
    m.inference('/data2/wangzhen/3DVL_Codebase/outputs/exp_vqa/2023-02-13_10-23-17_3DGQA')