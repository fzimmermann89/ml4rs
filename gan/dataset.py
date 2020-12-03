from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as TF


def extract_patches_by_size(img_paths, patch_width=32, patch_height=32):
    img_data = []
    save_dirs = []

    for img_path in img_paths:
        p = Path(img_path)
        img = Image.open(p)
        data = TF.to_tensor(img)
        data.unsqueeze(0)
        img_data.append(data)

        save_dir = Path(f'patches/{p.stem}')
        save_dirs.append(save_dir)

        # create folder if needed
        if not Path.exists(save_dir):
            Path.mkdir(save_dir, parents=True)

    nw = img_data[0].shape[2] // patch_width
    nh = img_data[0].shape[1] // patch_height

    for i in range(nh):
        for j in range(nw):
            start_i = i * patch_height
            start_j = j * patch_width

            for k in range(len(img_data)):
                patch = img_data[k][:, start_i:start_i + patch_height, start_j:start_j + patch_width]

                # save patch
                patch_img = TF.to_pil_image(patch)
                patch_img.save(Path(save_dirs[k], f'{i}-{j}.png'))


def extract_patches_by_num(img_paths, nw=3, nh=3):
    img_data = []
    save_dirs = []

    for img_path in img_paths:
        p = Path(img_path)
        img = Image.open(p)
        data = TF.to_tensor(img)
        data.unsqueeze(0)
        img_data.append(data)

        save_dir = Path(f'tiles/{p.stem}')
        save_dirs.append(save_dir)

        # create folder if needed
        if not Path.exists(save_dir):
            Path.mkdir(save_dir, parents=True)

    patch_width = img_data[0].shape[2] // nw
    patch_height = img_data[0].shape[1] // nh

    for i in range(nh):
        for j in range(nw):
            start_i = i * patch_height
            start_j = j * patch_width

            for k in range(len(img_data)):
                patch = img_data[k][:, start_i:start_i + patch_height, start_j:start_j + patch_width]

                # save patch
                patch_img = TF.to_pil_image(patch)
                patch_img.save(Path(save_dirs[k], f'{i}-{j}.png'))


if __name__ == '__main__':
    extract_patches_by_size(['WV2_Site1/gt.bmp', 'WV2_Site1/t1.bmp', 'WV2_Site1/t2.bmp'])
    extract_patches_by_num(['WV2_Site1/gt.bmp', 'WV2_Site1/t1.bmp', 'WV2_Site1/t2.bmp'])
