import argparse
import pathlib
import imageio

import numpy as np
from plyfile import PlyData
import torch
from gsplat.rendering import rasterization


def main(args):
    # Arguments
    ply_path: pathlib.Path = pathlib.Path(args.ply_path)
    output_img_path: pathlib.Path = pathlib.Path(args.output_path)
    
    # Torch setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import PLY
    plydata = PlyData.read(ply_path)
    num_gaussians = len(plydata['vertex'])
    print(f"Extracted PLY file: {args.ply_path}")
    print(f"Num. Gaussians: {num_gaussians}")
    
    # Extract Gaussian parameters from PLY
    means = torch.nn.Parameter(
        torch.from_numpy(
            np.transpose(
                np.stack(
                    (plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]),
                    dtype=np.float32,
                    )
                )
            ).to(device=device)
        )
    quats = torch.nn.Parameter(
        torch.from_numpy(
            np.transpose(
                np.stack(
                    (plydata["vertex"]["rot_0"], plydata["vertex"]["rot_1"], plydata["vertex"]["rot_2"], plydata["vertex"]["rot_3"]),
                    dtype=np.float32,
                    )
                )
            ).to(device=device)
        )
    scales = torch.nn.Parameter(
        torch.from_numpy(
            np.transpose(
                np.stack(
                    (plydata["vertex"]["scale_0"], plydata["vertex"]["scale_1"], plydata["vertex"]["scale_2"]),
                    dtype=np.float32,
                    )
                )
            ).to(device=device)
        )
    opacities = torch.nn.Parameter(
        torch.from_numpy(plydata["vertex"]["opacity"].astype(np.float32).reshape(num_gaussians, 1)).to(device=device)
        ).squeeze()
    sh0 = torch.nn.Parameter(
        torch.from_numpy(
            np.transpose(
                np.stack(
                    (plydata["vertex"]["f_dc_0"], plydata["vertex"]["f_dc_1"], plydata["vertex"]["f_dc_2"]),
                    dtype=np.float32
                    )
                )
            ).to(device=device)
        )
    shN = torch.nn.Parameter(
        torch.from_numpy(
            np.transpose(
                np.stack(
                    (
                        plydata["vertex"]["f_rest_0"],
                        plydata["vertex"]["f_rest_1"],
                        plydata["vertex"]["f_rest_2"],
                        plydata["vertex"]["f_rest_3"],
                        plydata["vertex"]["f_rest_4"],
                        plydata["vertex"]["f_rest_5"],
                        plydata["vertex"]["f_rest_6"],
                        plydata["vertex"]["f_rest_7"],
                        plydata["vertex"]["f_rest_8"],
                        plydata["vertex"]["f_rest_9"],
                        plydata["vertex"]["f_rest_10"],
                        plydata["vertex"]["f_rest_11"],
                        plydata["vertex"]["f_rest_12"],
                        plydata["vertex"]["f_rest_13"],
                        plydata["vertex"]["f_rest_14"],
                        plydata["vertex"]["f_rest_15"],
                        plydata["vertex"]["f_rest_16"],
                        plydata["vertex"]["f_rest_17"],
                        plydata["vertex"]["f_rest_18"],
                        plydata["vertex"]["f_rest_19"],
                        plydata["vertex"]["f_rest_20"],
                        plydata["vertex"]["f_rest_21"],
                        plydata["vertex"]["f_rest_22"],
                        plydata["vertex"]["f_rest_23"],
                        plydata["vertex"]["f_rest_24"],
                        plydata["vertex"]["f_rest_25"],
                        plydata["vertex"]["f_rest_26"],
                        plydata["vertex"]["f_rest_27"],
                        plydata["vertex"]["f_rest_28"],
                        plydata["vertex"]["f_rest_29"],
                        plydata["vertex"]["f_rest_30"],
                        plydata["vertex"]["f_rest_31"],
                        plydata["vertex"]["f_rest_32"],
                        plydata["vertex"]["f_rest_33"],
                        plydata["vertex"]["f_rest_34"],
                        plydata["vertex"]["f_rest_35"],
                        plydata["vertex"]["f_rest_36"],
                        plydata["vertex"]["f_rest_37"],
                        plydata["vertex"]["f_rest_38"],
                        plydata["vertex"]["f_rest_39"],
                        plydata["vertex"]["f_rest_40"],
                        plydata["vertex"]["f_rest_41"],
                        plydata["vertex"]["f_rest_42"],
                        plydata["vertex"]["f_rest_43"],
                        plydata["vertex"]["f_rest_44"],
                    ),
                    dtype=np.float32)
                )
            ).to(device=device)
        )
    colors = torch.cat(
        [
            sh0[:, None, :],              # [N, 1, 3]
            shN.reshape(-1, 3, 15).permute(0, 2, 1),     # [N, 15, 3]
        ],
        dim=1,
    )

    # Camera parameters
    camtoworld = torch.from_numpy(
        np.array(
            [                
                [0.9925463199615479,-0.07437276840209961,0.09654271602630615, -0.9115],
                [0.06815238296985626,0.9954756498336792,0.06620778888463974, -0.1460],
                [-0.10102997720241547,-0.05913468450307846,0.9931243658065796, -0.0980],
                [0.0, 0.0, 0.0, 1.0],
                # [1, 0 ,0, 0],
                # [0, 1, 0, -5],
                # [0, 0, 1, -10],
                # [0.0, 0.0, 0.0, 1.0],
                # [
                #     0.014746619388461113,
                #     0.13194887340068817,
                #     -0.9911468625068665,
                #     -0.4417670667171478
                # ],
                # [
                #     -0.9988508820533752,
                #     0.04714938998222351,
                #     -0.008584342896938324,
                #     -0.4398075044155121
                # ],
                # [
                #     0.045599278062582016,
                #     0.9901345372200012,
                #     0.1324925273656845,
                #     1.1461249589920044
                # ],
                # [
                #     0.0,
                #     0.0,
                #     0.0,
                #     1.0
                # ]
            ],
            dtype=np.float32,
        )
    ).to(device=device).unsqueeze(0)  # [1, 4, 4]

    ## For gert
    # R = camtoworld[:, :3, :3]  # [1, 3, 3]
    # T = -camtoworld[:, :3, 3]  # [1, 3]
    # precision = torch.float32
    # S = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=device, dtype=precision))
    # S = S.unsqueeze(0).expand(1, -1, -1)
    # R = torch.bmm(torch.bmm(S, R), S)
    # T = torch.bmm(S, T.unsqueeze(-1)).squeeze(-1)

    # transform_matrix = torch.tensor(
    #     [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    #     device=device,
    #     dtype=precision,
    # ).repeat(1, 1, 1)
    # R = torch.bmm(transform_matrix, R)
    # T = torch.bmm(transform_matrix, T.unsqueeze(-1)).squeeze(-1)
    # camtoworld = torch.cat([R, T.unsqueeze(-1)], dim=-1)
    # camtoworld = torch.cat([camtoworld, torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device, dtype=precision)], dim=1)

    Ks = torch.from_numpy(
        np.array(
            [
                [  1097.5108642578125, 0.0, 640],
                [  0.0, 1223.5762939453125, 360],
                [  0.0, 0.0, 1.0],
                # [
                #     639.6144409179688,
                #     0.0,
                #     486.6187744140625
                # ],
                # [
                #     0.0,
                #     639.6373291015625,
                #     505.19244384765625
                # ],
                # [
                #     0.0,
                #     0.0,
                #     1.0
                # ]

            ],
            dtype=np.float32,
        )
    ).to(device=device).unsqueeze(0)  # [1, 3, 3]

    # Rasterization
    render_colors, _, _ = rasterization(
        means=means,  # [N, 3]
        quats=quats,  # [N, 4]
        scales=torch.exp(scales),  # [N, 3]
        opacities=torch.sigmoid(opacities),  # [N]
        colors=colors,  # [N, S, 3]
        viewmats=torch.linalg.inv(camtoworld),  # [1, 4, 4]
        Ks=Ks,  # [1, 3, 3]
        width=1280,
        height=720,
        render_mode="RGB",
        sh_degree=3,
    )
    print(f"Rasterized image shape: {render_colors.shape}")
    
    # Save rasterized image
    for i in range(render_colors.shape[0]):
        rgb = render_colors[i].detach().cpu().numpy()
        rgb = (rgb.clip(0, 1) * 255).astype(np.uint8) # float [0, 1] -> uint8 [0, 255]
        imageio.imwrite(output_img_path, rgb)
    print("Finished extracting Gaussian parameters from PLY file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Rasterise 3D point clouds from PLY files.
            || uv run examples/rasterize_from_ply.py -p /datasets/OpenSplat/tumvie_mocap-1d-trans-events_left/splat.ply -o /datasets/OpenSplat/tumvie_mocap-1d-trans-events_left/splat.png
            """)
    parser.add_argument(
        "-p", "--ply_path", type=str, required=True, help="Path to the PLY file."
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="tmp.png", help="Path to save the rasterized image."
    )
    args = parser.parse_args()

    main(args)