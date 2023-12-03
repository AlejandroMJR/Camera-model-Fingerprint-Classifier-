import os


def directory_find(atom, root='.'):

    list_dirs = []
    for path, dirs, files in os.walk(root):

        if atom in dirs and 'trainset' not in path:
            list_dirs.append(os.path.join(path, atom))

    return list_dirs


def path_find(atom, root='.'):

    list_dirs = []
    for path, dirs, files in os.walk(root):

        if atom in path:
            list_dirs.append(os.path.join(path))

    return list_dirs

# real root folders
web_images_real_dict = {'covid': '/nas/public/exchange/semafor/eval1/web_images/original/COVIDimages_splitted/pictures',
                     'milano': '/nas/public/exchange/semafor/eval1/web_images/original/milano/pictures',
                     'military': '/nas/public/exchange/semafor/eval1/web_images/original/military_webimages/pictures',
                     'siena': '/nas/public/exchange/semafor/eval1/web_images/original/siena/pictures',
                     'climate': '/nas/public/exchange/semafor/eval1/web_images/original/climate_change/pictures',
                     'napoli': '/nas/public/exchange/semafor/eval1/web_images/original/napoli/pictures'}

wblots_real_dict = {'resize256': '/nas/home/smandelli/Pycharm_projects/stylegan2-ada-pytorch/dataset_wblots_orig'}

# PIX2PIX
pix2pix_real_root = '/nas/home/smandelli/Pycharm_projects/pix2pix-tf/pix2pix_real_images/'
pix2pix_real_keys = next(os.walk(pix2pix_real_root))[1]
pix2pix_real_dirs = [os.path.join(pix2pix_real_root, path) for path in pix2pix_real_keys]
pix2pix_real_dict = {pix2pix_real_keys[i]: pix2pix_real_dirs[i] for i in range(len(pix2pix_real_keys))}

efros_test_root = '/nas/home/smandelli/Pycharm_projects/wifs2020/dataset/'
atom = '0_real'
efros_real_dirs = directory_find(atom, efros_test_root)
# build dictionary
efros_real_keys = ['-'.join(x.split(efros_test_root)[-1].split(atom)[0].split('/')[:-1]) for x in efros_real_dirs]
efros_real_dict = {efros_real_keys[i]: efros_real_dirs[i] for i in range(len(efros_real_keys))}

# EFROS reduced to image-to-image translation models
i2i_models = ['crn', 'cyclegan', 'gaugan', 'imle', 'stargan']
i2i_cyclegan_models = ['cyclegan']

efros_real_i2i_keys = []
efros_real_i2i_dirs = []
for model in i2i_models:
    for k_idx, k in enumerate(efros_real_keys):
        if model in k:
            efros_real_i2i_keys.append(k)
            efros_real_i2i_dirs.append(efros_real_dirs[k_idx])
efros_real_i2i_dict = {efros_real_i2i_keys[i]: efros_real_i2i_dirs[i] for i in range(len(efros_real_i2i_keys))}
efros_real_i2i_cyclegan_dict = {efros_real_i2i_keys[i]: efros_real_i2i_dirs[i] for i in range(len(efros_real_i2i_keys)) if 'cyclegan' in efros_real_i2i_keys[i]}

# stargan2 real:
stargan2_real_root = '/nas/home/smandelli/Pycharm_projects/stargan-v2/data/afhq/'
stargan2_real_init_keys = next(os.walk(stargan2_real_root))[1]
stargan2_real_dirs = []
stargan2_real_keys = []
for key in stargan2_real_init_keys:
    stargan2_real_dirs.extend(path_find(key, stargan2_real_root)[1:])
    stargan2_real_keys.extend(['-'.join(x.split('data/')[-1].split('/')) for x in path_find(key, stargan2_real_root)[1:]])
stargan2_real_root = '/nas/home/smandelli/Pycharm_projects/stargan-v2/data/celeba_hq/'
for key in stargan2_real_init_keys:
    stargan2_real_dirs.extend(path_find(key, stargan2_real_root)[1:])
    stargan2_real_keys.extend(['-'.join(x.split('data/')[-1].split('/')) for x in path_find(key, stargan2_real_root)[1:]])

stargan2_real_dict = {stargan2_real_keys[i]: stargan2_real_dirs[i] for i in range(len(stargan2_real_keys))}

ffhq_real_dict = {'ffhq': '/nas/home/nbonettini/projects/jstsp-benford-gan/additional_gan_images/ffhq'}

eval1_pristine_dict = {'eval1_pristine': '/nas/public/dataset/semafor/eval_1/selected_pristine_images'}

# primo training: P = 128, N = 1 dà TPR@0.1 = 54.39%
# real_images = [web_images_real_dict, wblots_real_dict, efros_real_dict]
# secondo training: P = 128, N = 5, only image-to-image translation models
# real_images = [pix2pix_real_dict, efros_real_i2i_dict]
# terzo training: P = 128, N = 5, only Efros image-to-image translation models
# real_images = [efros_real_i2i_dict]
# quarto training : P = 128, N = 1, all images in RGB, aug, and then convert to gray
# real_images = [pix2pix_real_dict, web_images_real_dict, efros_real_dict, wblots_real_dict, stargan2_real_dict]
# quinto training : P = 128, N = 4, ONLY STARGAN2 images in RGB, aug, and then convert to gray
# real_images = [stargan2_real_dict]
# sesto training : P = 128, N = 4, ONLY CYCLEGAN images in RGB, aug, and then convert to gray
# real_images = [efros_real_i2i_cyclegan_dict]
# settimo training : P = 128, N = 1, all images in RGB, aug, convert to gray, normalize every img by its mean/std
# real_images = [pix2pix_real_dict, web_images_real_dict, efros_real_dict, wblots_real_dict, stargan2_real_dict]
# ottavo training : P = 128, N = 1, all images in RGB, aug
#real_images = [pix2pix_real_dict, web_images_real_dict, efros_real_dict, wblots_real_dict, stargan2_real_dict]
# test sulle immagini nvidia con ffhq:
# real_images = [ffhq_real_dict]
# test sulle immagini nvidia con le pristine dell'evaluation 1:
real_images = [eval1_pristine_dict]

#################################################### GAN root folders

# PIX2PIX
pix2pix_root = '/nas/public/exchange/semafor/eval1/pix2pix/synth_data/'
pix2pix_keys = next(os.walk(pix2pix_root))[1]
pix2pix_dirs = [os.path.join(pix2pix_root, path) for path in pix2pix_keys]
pix2pix_dict = {pix2pix_keys[i]: pix2pix_dirs[i] for i in range(len(pix2pix_keys))}

# STYLEGAN2
sg2_root = '/nas/public/exchange/semafor/eval1/stylegan2/100k-generated-images/'
atom = 'stylegan2-config-f-psi-0.5'
sg2_dirs = directory_find(atom, sg2_root)
sg2_dirs.remove(os.path.join('/nas/public/exchange/semafor/eval1/stylegan2/100k-generated-images/car-512x384', atom))
sg2_keys = ['-'.join(x.split(sg2_root)[-1].split('/')) for x in sg2_dirs]
atom = 'stylegan2-config-f-psi-1.0'
other_dirs = directory_find(atom, sg2_root)
other_dirs.remove(os.path.join('/nas/public/exchange/semafor/eval1/stylegan2/100k-generated-images/car-512x384', atom))
sg2_dirs.extend(other_dirs)
sg2_keys.extend(['-'.join(x.split(sg2_root)[-1].split('/')) for x in other_dirs])
sg2_dict = {sg2_keys[i]: sg2_dirs[i] for i in range(len(sg2_keys))}

# EFROS
atom = '1_fake'
efros_fake_dirs = directory_find(atom, efros_test_root)
# build dictionary
efros_fake_keys = ['-'.join(x.split(efros_test_root)[-1].split(atom)[0].split('/')[:-1]) for x in efros_fake_dirs]
efros_fake_dict = {efros_fake_keys[i]: efros_fake_dirs[i] for i in range(len(efros_fake_keys))}

# EFROS reduced to image-to-image translation models
i2i_models = ['crn', 'cyclegan', 'gaugan', 'imle', 'stargan']

efros_fake_i2i_keys = []
efros_fake_i2i_dirs = []
for model in i2i_models:
    for k_idx, k in enumerate(efros_fake_keys):
        if model in k:
            efros_fake_i2i_keys.append(k)
            efros_fake_i2i_dirs.append(efros_fake_dirs[k_idx])
efros_fake_i2i_dict = {efros_fake_i2i_keys[i]: efros_fake_i2i_dirs[i] for i in range(len(efros_fake_i2i_keys))}
efros_fake_i2i_cyclegan_dict = {efros_fake_i2i_keys[i]: efros_fake_i2i_dirs[i] for i in range(len(efros_fake_i2i_keys))
                                if 'cyclegan' in efros_fake_i2i_keys[i]}

# WBLOTS GENERATED WITH STG2-pytorch
wblots_fake_dict = {'sg2-pytorch': '/nas/home/smandelli/Pycharm_projects/stylegan2-ada-pytorch/out_wblots_trunc-0.7-'
                                   'snapshot-003400'}

# WEB IMAGES GENERATED WITH STG2-tensorflow
web_images_fake_root = '/nas/home/smandelli/Pycharm_projects/stylegan2-tf/results_sg2_web_images/'
atom = '00000-generate-images'
web_images_fake_dirs = directory_find(atom, web_images_fake_root)
web_images_fake_keys = ['-'.join(x.split(web_images_fake_root)[-1].split(atom)[0].split('/')[:-1]) for x in
                        web_images_fake_dirs]
web_images_fake_dict = {web_images_fake_keys[i]: web_images_fake_dirs[i] for i in range(len(web_images_fake_keys))}

# SG2 IMAGES OF MILITARY VEHICLES
sg2_military_vehicles_root = '/nas/home/smandelli/Pycharm_projects/semafor_web_images/synth_vs_real_detection/' \
                             'stylegan-ada-military-vehicles-0309-100'
sg2_military_vehicles_dict = {'military_vehicles': sg2_military_vehicles_root}

# stargan2 fake:
stargan2_fake_afhq_root = '/nas/home/smandelli/Pycharm_projects/stargan-v2/expr/results/afhq_N-70'
stargan2_fake_celeba_hq_root = '/nas/home/smandelli/Pycharm_projects/stargan-v2/expr/results/celeba_hq_N-71'
stargan2_fake_dict = {'afhq_N-70': stargan2_fake_afhq_root, 'celeba_hq_N-71':stargan2_fake_celeba_hq_root}

# primo training: P = 128, N = 1 dà TPR@0.1 = 54.39%
# fake_images = [pix2pix_dict, sg2_dict, efros_fake_dict, web_images_fake_dict]
# secondo training: P = 128, N = 5, only image-to-image translation model
# fake_images = [pix2pix_dict, efros_fake_i2i_dict]
# terzo training: P = 128, N = 5, only Efros image-to-image translation models
# fake_images = [efros_fake_i2i_dict]
# quarto training : P = 128, N = 1, all images in RGB, aug, and then convert to gray
# fake_images = [pix2pix_dict, sg2_dict, efros_fake_dict, web_images_fake_dict, sg2_military_vehicles_dict,
#                wblots_fake_dict, stargan2_fake_dict]
# quinto training : P = 128, N = 4, ONLY STARGAN2 images in RGB, aug, and then convert to gray
# fake_images = [stargan2_fake_dict]
# sesto training: P = 128, N = 4, ONLY cyclegan images in RGB, aug, and then convert to gray
# fake_images = [efros_fake_i2i_cyclegan_dict]
# settimo training : P = 128, N = 1, all images in RGB, aug, normalize every img by its mean/std
# fake_images = [pix2pix_dict, sg2_dict, efros_fake_dict, web_images_fake_dict, sg2_military_vehicles_dict,
#                wblots_fake_dict, stargan2_fake_dict]
# ottavo training : P = 128, N = 1, all images in RGB, aug
fake_images = [pix2pix_dict, sg2_dict, efros_fake_dict, web_images_fake_dict, sg2_military_vehicles_dict,
               wblots_fake_dict, stargan2_fake_dict]

################### FOLDERS

# pkl_dir = 'pkl'
# log_dir = 'runs'
# models_dir = 'weights'
# results_dir = 'results'
# pkl_dir = 'pkl_gray_i2i'
# log_dir = 'runs_gray_i2i'
# models_dir = 'weights_gray_i2i'
# results_dir = 'results_gray_i2i'
# pkl_dir = 'pkl_gray_i2i_efros'
# log_dir = 'runs_gray_i2i_efros'
# models_dir = 'weights_gray_efros'
# results_dir = 'results_gray_efros'
# log_dir = 'runs_rgb2gray_i2i_efros'
# models_dir = 'weights_rgb2gray_i2i_efros'
# results_dir = 'results_rgb2gray_i2i_efros'
# pkl_dir = 'pkl_rgb2gray'
# log_dir = 'runs_rgb2gray'
# models_dir = 'weights_rgb2gray'
# results_dir = 'results_rgb2gray'
# pkl_dir = 'pkl_rgb2gray_stargan2'
# log_dir = 'runs_rgb2gray_stargan2'
# models_dir = 'weights_rgb2gray_stargan2'
# results_dir = 'results_rgb2gray_stargan2'
# pkl_dir = 'pkl_rgb2gray_cyclegan'
# log_dir = 'runs_rgb2gray_cyclegan'
# models_dir = 'weights_rgb2gray_cyclegan'
# results_dir = 'results_rgb2gray_cyclegan'
# pkl_dir = 'pkl_rgb2gray_img_dyn'
# log_dir = 'runs_rgb2gray_img_dyn'
# models_dir = 'weights_rgb2gray_img_dyn'
# results_dir = 'results_rgb2gray_img_dyn'
#pkl_dir = 'pkl_rgb'
log_dir = 'runs_rgb'
models_dir = 'weights_rgb'
results_dir = 'results_rgb'

# per l'hackathon2:
pkl_dir = 'pkl_rgb_hack2'