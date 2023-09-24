#from GLFC import GLFC_model
from GLFC3 import GLFC3_model
from ResNet import resnet18_cbam
import torch
import copy
import random
import os.path as osp
import os
from myNetwork import network, LeNet
from Fed_utils import * 
from ProxyServer import * 
from mini_imagenet import *
from tiny_imagenet import *
from option import args_parser


import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from dualpromptlib.datasets import build_continual_dataloader
# from dualpromptlib.engine import *
from dualpromptlib.fed_pip_engine import *
import dualpromptlib.models2 as models
import dualpromptlib.utils as utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

from fed_pip_utils import * 
import copy

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    
    # train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
    #                                         transforms.RandomHorizontalFlip(p=0.5),
    #                                         transforms.ColorJitter(brightness=0.24705882352941178),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    # # test_transform = transforms.Compose([transforms.RandomCrop((args.img_size, args.img_size), padding=4),
    # test_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
    #                                         transforms.RandomHorizontalFlip(p=0.5),
    #                                         transforms.ColorJitter(brightness=0.24705882352941178),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    # train_dataset = iCIFAR100('dataset', transform=train_transform, download=True)
    # test_dataset = iCIFAR100('dataset', test_transform=test_transform, train=False, download=True)

    # current_classes = random.sample([x for x in range(0, 10)], 6)
    # print("current class")
    # print(current_classes)
    # tr_dataset = train_dataset.getTrainData(current_classes, [], [])
    # train_loader = DataLoader(dataset=tr_dataset,
    #                               shuffle=True,
    #                               batch_size=24,
    #                               num_workers=16,
    #                               pin_memory=True)

    # niter=0
    # for X_batch, y_batch in train_loader:[task_id]['train']
    #     print("iteration: " + str(niter))
    #     print(X_batch.shape)
    
    data_loaders=[]
    class_masks=[]

    models=[]
    original_models=[]
    models_without_ddp=[]
    optimizers=[]
    lr_schedulers=[]

    data_loader, class_mask = build_continual_dataloader(args)
    data_loaders.append(data_loader)
    class_masks.append(class_mask)
    #data_loader2, class_mask2 = build_continual_dataloader(args)
    for i in range(1, args.num_clients):
        dl =  copy.deepcopy(data_loader)
        cm =  copy.deepcopy(class_mask)
        random.shuffle(dl)
        data_loaders.append(dl)
        class_masks.append(cm)

    # data_loader0=data_loader[0]['train']
    # # print(vars(data_loader0.dataset))
    # set_labels=set()
    # print("Check train labels:")
    # for input, target in data_loader0:
    #     set_labels.update(set(target.numpy()))
    #     # break
    # print(list(set_labels))

    # data_loader0=data_loader[0]['val']
    # set_labels=set()
    # print("Check val labels:")
    # for input, target in data_loader0:
    #     # print(input.shape)
    #     # print(target.shape)
    #     set_labels.update(set(target.numpy()))
    #     # break
    # print(list(set_labels))



    # # data_loader2 = random.shuffle(data_loader)

    # data_loader0=data_loader2[0]['train']
    # # print(vars(data_loader0.dataset))
    # set_labels=set()
    # print("Check train labels:")
    # for input, target in data_loader0:
    #     set_labels.update(set(target.numpy()))
    #     # break
    # print(list(set_labels))

    # data_loader0=data_loader2[0]['val']
    # set_labels=set()
    # print("Check val labels:")
    # for input, target in data_loader0:
    #     # print(input.shape)
    #     # print(target.shape)
    #     set_labels.update(set(target.numpy()))
    #     # break
    # print(list(set_labels))
 

    # print(f"Creating server original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    original_model.to(device)

    
    print(f"Creating server model: {args.model}")
    server_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
    )
    server_model.to(device)
    

    # print(vars(server_model.head))

    for i in range(args.num_clients):
        print(f"Creating model : {args.model} for client {i}")
        # original_model = original_model
        model =   copy.deepcopy(server_model).to(device)

        models.append(model)
        original_models.append(original_model)



    # original_model.to(device)
    # model.to(device)  

    # original_model2.to(device)
    # model2.to(device)  
    
    
    # FedAvg(server,[model,model2])
    # print(model.g_prompt)
    # print(model.e_prompt.prompt_key)
    # print(model.e_prompt.prompt)
    

    if args.freeze:
        # all parameters are frozen for original vit model
        for i in range(0,args.num_clients):
            for p in original_models[i].parameters():
                p.requires_grad = False
            
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in models[i].named_parameters():
                if n.startswith(tuple(args.freeze)):
                    p.requires_grad = False
        
    print(args)

    # if args.eval:
    #     acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    #     for task_id in range(args.num_tasks):
    #         checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
    #         if os.path.exists(checkpoint_path):
    #             print('Loading checkpoint from:', checkpoint_path)
    #             checkpoint = torch.load(checkpoint_path)
    #             model.load_state_dict(checkpoint['model'])
    #         else:
    #             print('No checkpoint found at:', checkpoint_path)
    #             return
    #         _ = evaluate_till_now(model, original_model, data_loader, device, 
    #                                         task_id, class_mask, acc_matrix, args,)
        
    #     return
    server_model_without_ddp = server_model
    for i in range(0,args.num_clients):
        models_without_ddp.append(models[i])
        

    if args.distributed:
        server_model_without_ddp = server_model.module
        for i in range(0,args.num_clients):
            models[i] = torch.nn.parallel.DistributedDataParallel(models[i], device_ids=[args.gpu])
            models_without_ddp[i] = models[i].module
            

        # model2 = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # model_without_ddp2 = model2.module
    
    n_parameters = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0


    server_optimizer = create_optimizer(args, server_model_without_ddp)
    for i in range(0,args.num_clients):
        optimizer = create_optimizer(args, models_without_ddp[i])
        optimizers.append(optimizer)

    # optimizer2 = create_optimizer(args, model_without_ddp2)

    if args.sched != 'constant':
        for i in range(0,args.num_clients):
            lr_scheduler, _ = create_scheduler(args, optimizers[i])
            lr_schedulers.append(lr_scheduler)
    elif args.sched == 'constant':
        for i in range(0,args.num_clients):
            lr_scheduler = None
            lr_schedulers.append(lr_scheduler)
        

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # train_and_evaluate(model, model_without_ddp, original_model,
    #                 criterion, data_loader, optimizer, lr_scheduler,
    #                 device, class_mask, args)

    
    # train_transform = transforms.Compose([transforms.RandomCrop((args.img_size, args.img_size), padding=4),



    # train_and_evaluate2(model, model_without_ddp, original_model,
    #                 model2, model_without_ddp2, original_model2,
    #                 criterion, data_loader, optimizer, lr_scheduler,
    #                 device, class_mask, args)
    # FedDistribute(server_model,models,args.distributed)
    FedAvg(server_model,models,args.distributed)

    

    all_time_round=0
    global_prototype =  None
    global_prototype_var =  None
    

    for task_id in range(0,args.num_tasks):
        clients_participations = [0] * args.num_clients

        for n_round in range(0,args.rounds_per_task):
            print("Task ["+str(task_id+1)+"] Global Round : "+str(all_time_round+1))

            clients_index = random.sample(range(args.num_clients), args.local_clients)
            print(clients_index)

            # if (n_round==0):
                # FedDistribute(server_model,[models[i] for i in clients_index],args.distributed)
            # else:
            # FedDistributeWithHead(server_model,[models[i] for i in clients_index],args.distributed)
            # train_and_evaluate_pertask(models, models_without_ddp, original_models,
            #                 criterion, data_loaders, optimizers, lr_schedulers,
            #                 device, class_masks, task_id, args)
                        
            idx_notrain = [x for x in clients_index if clients_participations[x]==0]
            idx_trained = [x for x in clients_index if clients_participations[x]>0]
            FedDistribute(server_model,[models[i] for i in idx_trained],args.distributed)
            FedDistributeWithHead(server_model,[models[i] for i in idx_notrain],args.distributed)

            if (n_round) == 0:
                clients_prototype, clients_prototype_var = train_pertask([models[i] for i in clients_index], 
                                [models_without_ddp[i] for i in clients_index], 
                                [original_models[i] for i in clients_index],
                                criterion, 
                                # [data_loaders[i] for i in clients_index],
                                [data_loaders[0]],
                                [optimizers[i] for i in clients_index], 
                                [lr_schedulers[i] for i in clients_index],
                                device, 
                                [class_masks[i] for i in clients_index], 
                                task_id, args)

            else:
                clients_prototype, clients_prototype_var  = train_pertask_v2([models[i] for i in clients_index], 
                                [models_without_ddp[i] for i in clients_index], 
                                [original_models[i] for i in clients_index],
                                criterion, 
                                # [data_loaders[i] for i in clients_index],
                                [data_loaders[0]],
                                [optimizers[i] for i in clients_index], 
                                [lr_schedulers[i] for i in clients_index],
                                device, 
                                [class_masks[i] for i in clients_index], 
                                task_id, global_prototype, global_prototype_var, args)
                print("Train 1 round with prototype done")



            for i in clients_index:
                clients_participations[i] =  clients_participations[i] + 1

            # global_prototype =  FedAvgPrototype(clients_prototype,task_id)

            clients_weight =  [clients_participations[i] for i in clients_index]
            # global_prototype, global_prototype_var =  FedAvgPrototype2(clients_prototype,clients_prototype_var,clients_weight,task_id)
            global_prototype, global_prototype_var =  FedWeightedAvgPrototype(clients_prototype,clients_prototype_var,clients_weight,task_id,args.classes_per_task)
        # train_and_evaluate_pertask(models, models_without_ddp, original_models,
        #                 criterion, data_loaders, optimizers, lr_schedulers,
        #                 device, class_masks, 0, args)

            if (n_round < (args.rounds_per_task-1)):
                FedWeightedAvg(server_model, [models[i] for i in clients_index], clients_weight, args.distributed)
            else:
                FedWeightedAvgWithHead(server_model, [models[i] for i in clients_index], clients_weight, args.distributed)
                # idx_notrain = [ x for x in range(0,args.num_clients) if clients_participations[x]==0 ]
                # FedDistributeWithHead(server_model,[models[i] for i in idx_notrain],args.distributed)

            all_time_round =  all_time_round + 1

        FedDistribute(server_model,models,args.distributed)

        evaluate_server_global_model(server_model, server_model_without_ddp, original_model,
                            criterion, data_loaders[0], server_optimizer, None,
                            device, None, task_id, args)

        print("Clients Participation on Task: [" +str(task_id+1)+"]")
        print(clients_participations)
        
        # evaluate_all_clients(models, models_without_ddp, original_models,
        #                     criterion, data_loaders, optimizers, lr_schedulers,
        #                     device, class_masks, task_id, args)


        # FedAvg(server_model, models,args.distributed)
        # FedDistribute(server_model,models,args.distributed)


    # train_and_evaluate_pertask(models, models_without_ddp, original_models,
    #                 criterion, data_loaders, optimizers, lr_schedulers,
    #                 device, class_masks, 1, args)

    # FedAvg(server_model, models,args.distributed)
    # FedDistribute(server_model,models,args.distributed)

    # train_and_evaluate(models[0], models_without_ddp[0], original_models[0],
    #                 criterion, data_loaders[0], optimizers[0], lr_schedulers[0],
    #                 device, class_masks[0], args)

    # train_and_evaluate_pertask(models, models_without_ddp, original_models,
    #                 criterion, data_loaders, optimizers, lr_schedulers,
    #                 device, class_masks, 2, args)

    # train_and_evaluate2(model, model_without_ddp, original_model,
    #                 model2, model_without_ddp2, original_model2,
    #                 criterion, train_loader, optimizer, lr_scheduler,
    #                 device, class_mask, args)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    # parser.add_argument('--output_dir', default='./output/', type=str, help='output dir')

    config = parser.parse_known_args()[-1][0]

    # config = 'cifar100_dualprompt'
    # config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_dualprompt':
        print("chek config cifar100")
        from dualpromptlib.configs.cifar100_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 DualPrompt configs')
        # print()


    elif config == 'fedcl_cifar100_dualprompt':
        print("chek config FedCL cifar100")
        from dualpromptlib.configs.fedcl_cifar100_dualprompt import get_args_parser
        config_parser = subparser.add_parser('fedcl_cifar100_dualprompt', help='Split-CIFAR100 DualPrompt configs')

    elif config == 'fedcl_cifar100_5t_dualprompt':
        print("chek config FedCL cifar100 5T")
        from dualpromptlib.configs.fedcl_cifar100_5t_dualprompt import get_args_parser
        config_parser = subparser.add_parser('fedcl_cifar100_5t_dualprompt', help='Split-CIFAR100 5T DualPrompt configs')

    elif config == 'fedcl_cifar100_20t_dualprompt':
        print("chek config FedCL cifar100 20T")
        from dualpromptlib.configs.fedcl_cifar100_20t_dualprompt import get_args_parser
        config_parser = subparser.add_parser('fedcl_cifar100_20t_dualprompt', help='Split-CIFAR100 20T DualPrompt configs')


    elif config == 'fedcl_tinyimagenet10t_dualprompt':
        print("chek config FedCL TinyImagenet")
        from dualpromptlib.configs.fedcl_tinyimagenet10t_dualprompt import get_args_parser
        config_parser = subparser.add_parser('fedcl_tinyimagenet10t_dualprompt', help='Split-TinyImagenet DualPrompt configs')

    elif config == 'fedcl_tinyimagenet20t_dualprompt':
        print("chek config FedCL TinyImagenet 20T")
        from dualpromptlib.configs.fedcl_tinyimagenet20t_dualprompt import get_args_parser
        config_parser = subparser.add_parser('fedcl_tinyimagenet20t_dualprompt', help='Split-TinyImagenet 20T DualPrompt configs')

    elif config == 'fedcl_tinyimagenet5t_dualprompt':
        print("chek config FedCL TinyImagenet 5T")
        from dualpromptlib.configs.fedcl_tinyimagenet5t_dualprompt import get_args_parser
        config_parser = subparser.add_parser('fedcl_tinyimagenet5t_dualprompt', help='Split-TinyImagenet 5T DualPrompt configs')


    elif config == 'fedcl_imagenetsubset_dualprompt':
        print("chek config FedCL ImagenetSubset")
        from dualpromptlib.configs.fedcl_imagenetsubset_dualprompt import get_args_parser
        config_parser = subparser.add_parser('fedcl_imagenetsubset_dualprompt', help='Split-ImagenetSubset DualPrompt configs')

    elif config == 'fedcl_imagenetsubset5t_dualprompt':
        print("chek config FedCL ImagenetSubset 5T")
        from dualpromptlib.configs.fedcl_imagenetsubset5t_dualprompt import get_args_parser
        config_parser = subparser.add_parser('fedcl_imagenetsubset5t_dualprompt', help='Split-ImagenetSubset 5T DualPrompt configs')

    elif config == 'fedcl_imagenetsubset20t_dualprompt':
        print("chek config FedCL ImagenetSubset 20T")
        from dualpromptlib.configs.fedcl_imagenetsubset20t_dualprompt import get_args_parser
        config_parser = subparser.add_parser('fedcl_imagenetsubset20t_dualprompt', help='Split-ImagenetSubset 20T DualPrompt configs')


    elif config == 'imr_dualprompt':
        from dualpromptlib.configs.imr_dualprompt import get_args_parser
        config_parser = subparser.add_parser('imr_dualprompt', help='Split-ImageNet-R DualPrompt configs')
    else:
        raise NotImplementedError
        
    get_args_parser(config_parser)

    parser.add_argument('--output_dir', default='./output/', type=str, help='output dir')
    args = parser.parse_args()
    print(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)