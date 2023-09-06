def get_num_patches(image_height, image_width, patch_size, stride):
    num_patches_height = (image_height - patch_size) // stride + 1
    num_patches_width = (image_width - patch_size) // stride + 1
    num_patches = num_patches_height * num_patches_width
    return num_patches

# 이미지 크기와 등분할 patch 크기, stride 설정
image_height = 1024
image_width = 1024
patch_size = 224
stride = 224

num_patches = get_num_patches(image_height, image_width, patch_size, stride)
print("Number of patches:", num_patches)

