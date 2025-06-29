import argparse

import torch
import open3d as o3d

from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

from src.pipeline.sfm.fast3r import Fast3RSfM
from src.utils.io import write_cameras_txt, write_images_txt, write_points3D_txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference and save results from Fast3R model.")
    parser.add_argument("--input", "-i", type=str, help="Path to the input directory containing images.", default="data")
    parser.add_argument("--output", "-o", type=str, help="Path to the output directory to save results.", default="output")
    args = parser.parse_args()

    try:
        model = Fast3R.from_pretrained("models/fast3r")
    except:
        model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    model.eval()
    lit_module.eval()

    images = load_images(args.input, size=512)


    output_dict, profiling_info = inference(
        images,
        model,
        device,
        dtype=torch.float32,
        verbose=True,
        profiling=True,
    )

    confidence = 0.1
    lit_module.align_local_pts3d_to_global(
        preds=output_dict["preds"],
        views=output_dict["views"],
        min_conf_thr_percentile=confidence,
    )

    sfm = Fast3RSfM(output_dict)

    sfm(conf_thr=confidence, downsample=True, voxel_size=0.01)
    write_cameras_txt(sfm.cameras, args.output)
    write_images_txt(sfm.views, args.output, conf_threshold=confidence)
    write_points3D_txt(sfm.pcd, args.output)
    # o3d.visualization.draw_geometries([sfm.pcd], window_name="Fast3R Point Cloud", width=800, height=600)