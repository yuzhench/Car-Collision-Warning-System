import cv2
import numpy as np
from loguru import logger

from superpoint_superglue_deployment import Matcher


# def main():
#     query_image = cv2.imread("car/second/dodge1.jpg")
#     ref_image = cv2.imread("car/second/dodge3.jpg")

#     query_gray = cv2.imread("car/second/dodge1.jpg", 0)
#     ref_gray = cv2.imread("car/second/dodge3.jpg", 0)

#     superglue_matcher = Matcher(
#         {
#             "superpoint": {
#                 "input_shape": (-1, -1),
#                 "keypoint_threshold": 0.003,
#             },
#             "superglue": {
#                 "match_threshold": 0.5,
#             },
#             "use_gpu": True,
#         }
#     )
#     query_kpts, ref_kpts, _, _, matches = superglue_matcher.match(query_gray, ref_gray)
#     M, mask = cv2.findHomography(
#         np.float64([query_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
#         np.float64([ref_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
#         method=cv2.USAC_MAGSAC,
#         ransacReprojThreshold=5.0,
#         maxIters=10000,
#         confidence=0.95,
#     )
#     logger.info(f"number of inliers: {mask.sum()}")
#     matches = np.array(matches)[np.all(mask > 0, axis=1)]
#     matches = sorted(matches, key=lambda match: match.distance)
#     matched_image = cv2.drawMatches(
#         query_image,
#         query_kpts,
#         ref_image,
#         ref_kpts,
#         matches[:50],
#         None,
#         flags=2,
#     )
#     cv2.imwrite("car/second/matched_image_13.jpg", matched_image)


# if __name__ == "__main__":
#     main()


def main():
    query_image = cv2.imread("traffic_cross_frames_cutted/frame_200.jpg")
    ref_image = cv2.imread("traffic_cross_frames_cutted/frame_208.jpg")

    query_gray = cv2.imread("traffic_cross_frames_cutted/frame_200.jpg", 0)
    ref_gray = cv2.imread("traffic_cross_frames_cutted/frame_208.jpg", 0)

    superglue_matcher = Matcher(
        {
            "superpoint": {
                "input_shape": (-1, -1),
                "keypoint_threshold": 0.003,
            },
            "superglue": {
                "match_threshold": 0.5,
            },
            "use_gpu": True,
        }
    )
    query_kpts, ref_kpts, _, _, matches = superglue_matcher.match(query_gray, ref_gray)

     # Check if there are enough keypoints and matches for homography estimation
    if len(query_kpts) < 4 or len(ref_kpts) < 4 or len(matches) < 4:
        logger.info("Not enough keypoints or matches for homography estimation.")
        return
    

    M, mask = cv2.findHomography(
        np.float64([query_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
        np.float64([ref_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=5.0,
        maxIters=10000,
        confidence=0.95,
    )
    num_inliers = mask.sum()
    logger.info(f"Number of inliers: {num_inliers}")
    
    if num_inliers < threshold:
        logger.info("Objects in the images are likely different.")
    else:
        logger.info("Objects in the images are likely the same.")
        matches = np.array(matches)[np.all(mask > 0, axis=1)]
        matches = sorted(matches, key=lambda match: match.distance)
        matched_image = cv2.drawMatches(
            query_image,
            query_kpts,
            ref_image,
            ref_kpts,
            matches[:50],
            None,
            flags=2,
        )
        cv2.imwrite("matched_image_13.jpg", matched_image)

if __name__ == "__main__":
    threshold = 10  # Set your threshold for the number of inliers
    main()
