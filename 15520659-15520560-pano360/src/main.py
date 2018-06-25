# coding: utf-8

#import sys
import cv2
import multiprocessing as mp
import feature
import utils
import stitch


DESCRIPTOR_SIZE   = 5
FEATURE_THRESHOLD = 0.01
MATCHING_Y_RANGE  = 30


if __name__ == '__main__':
    
    input_dirname = "../input_image/Thangmay/"
    
    pool = mp.Pool(mp.cpu_count())

    img_list, focal_length = utils.parse(input_dirname)
    
    
    print("Chuyển tọa độ những ảnh về tọa độ trụ (cylinder)")
    cylinder_img_list = pool.starmap(utils.cylindrical_projection, [(img_list[i], focal_length[i]) for i in range(len(img_list))])


    _, img_width, _ = img_list[0].shape
    stitched_image = cylinder_img_list[0].copy()

    shifts = [[0, 0]]
    cache_feature = [[], []]

    for i in range(1, len(cylinder_img_list)):
        print('Xử lý '+str(i+1)+'/'+str(len(cylinder_img_list)))
        img1 = cylinder_img_list[i-1]
        img2 = cylinder_img_list[i]
        
        print(' + Hình '+str(i)+' có ', end='',flush=True)
        descriptors1, position1 = cache_feature
        if len(descriptors1) == 0:
            corner_response1 = feature.harris_corner(img1, pool)
            descriptors1, position1 = feature.extract_description(img1, corner_response1, kernel=DESCRIPTOR_SIZE, threshold=FEATURE_THRESHOLD)
        print(str(len(descriptors1))+' điểm đặc trưng được trích xuất')

        print(' + Hình '+str(i+1)+' có ', end='', flush=True)
        corner_response2 = feature.harris_corner(img2, pool)
        descriptors2, position2 = feature.extract_description(img2, corner_response2, kernel=DESCRIPTOR_SIZE, threshold=FEATURE_THRESHOLD)
        print(str(len(descriptors2))+' điểm đặc trưng được trích xuất')

        cache_feature = [descriptors2, position2]

        
        print(' + Số điểm đặc trưng của hai hình: ', end='', flush=True)
        matched_pairs = feature.matching(descriptors1, descriptors2, position1, position2, pool, y_range=MATCHING_Y_RANGE)
        
        print(str(len(matched_pairs)) )


        print(' + Khử nhiễu và tìm best shif: ', end='', flush=True)
        
        shift = stitch.RANSAC(matched_pairs, shifts[-1])
        shifts += [shift]
        print(shift)

        print(' + Ghép hình ', end='', flush=True)
        stitched_image = stitch.stitching(stitched_image, img2, shift, pool, blending=True)
        cv2.imwrite(str(i) +'.jpg', stitched_image)
        print('\nĐã lưu hình.')


    print('Chỉnh sửa hình')
    aligned = stitch.end2end_align(stitched_image, shifts)
    cv2.imwrite('aligned.jpg', aligned)

    cropped = stitch.crop(stitched_image)
    cv2.imwrite('cropped.jpg', cropped)
