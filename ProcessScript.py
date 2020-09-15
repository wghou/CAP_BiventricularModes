# This is the data process script for the reconstruction of Ventricular model
# The Ventricular data is download from the Cardiac Atlas Project @ https://www.cardiacatlas.org/

# The two Principal Component Analysis (PCA) atlases are described as follows:
# 1\ UKBRVLV.h5 contains PCA atlas derived from 630 healthy reference subjects from the UK Biobank Study (see [Petersen et al., 2017] @ https://jcmr-online.biomedcentral.com/articles/10.1186/s12968-017-0327-9).
# 2\ UKBRVLV_ALL.h5 contains PCA atlas derived from all 4,329 subjects from the UK Biobank Study.
# Only the first 200 PCA components are shared.

# The PCA structure is as follows:
# /COEFF: N x 200 matrix of the first 200 principal components, where N is the number of  sample points
# /LATENT: 200 elements vector of the eigenvalues,
# /EXPLAINED: 200 elements vector that show the percentage of the total variance explained by each principal component,
# /MU: N elements vector that defines the mean shape of the biventricular model.

import h5py as h5
import sys
import numpy as np
from scipy.spatial import Delaunay
import open3d as o3d
import pyvista as pv
import tetgen

# extract point cloud from the h5 file
def extract_component_from_h5_to_pcd(h5_file="", index=0, visual=False, save_file=""):

    assert(h5_file is not "")

    # read H5 file
    pc = h5.File(h5_file, 'r')

    coeff = pc['COEFF']
    latent = pc['LATENT']
    explained = pc['EXPLAINED']
    mu = np.transpose(pc['MU'])

    # calculate the coordinate of the points
    S = mu + (1.5 * np.sqrt(latent[0, 0]) * coeff[index, :])
    N = S.shape[1] // 2

    points = np.reshape(S[0, 0:N], (-1, 3))

    # convert to point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()

    # save point cloud file, such as .pcd, .xyz
    if(save_file is not ""):
        o3d.io.write_point_cloud(save_file, pcd)

    # visualize the point cloud
    if(visual is True):
        o3d.visualization.draw_geometries([pcd], width=1024, height=768, left=100, top=100)

    # return the point cloud object
    return pcd

# doesn't work for now
def convert_to_surface_mesh(pcd, visual=False, save_file=""):
    # some calculation
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1 * avg_dist

    # convert the point cloud to triangle surface mesh
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector([radius, radius * 2]))

    # save surface file, such as .ply
    if(save_file is not ""):
        o3d.io.write_triangle_mesh(save_file, mesh)

    # visualize the surface mesh
    if(visual is True):
        o3d.visualization.draw_geometries([mesh], width=1024, height=768, left=100, top=100, mesh_show_wireframe=True, mesh_show_back_face=True)

    # return the surface mesh object
    return mesh


def surface_mesh_delaunay(surface_mesh_file="", visual=False, visual_sub_grid=False, save_file=""):

    assert(surface_mesh_file is not "")

    # read surface mesh file from .ply
    surface_mesh = pv.PolyData(surface_mesh_file)

    # execute tetgen
    # don't work for now
    tet = tetgen.TetGen(surface_mesh)
    tet.tetrahedralize(nobisect=True)
    tet_grid = tet.grid

    # save the tet grid mesh
    if(save_file is not ""):
        tet.write(save_file)

    # visualize the surface mesh & tet grid
    if(visual is True):
        plotter = pv.Plotter()
        plotter.add_mesh(surface_mesh, 'r', 'wireframe')

        # not work now
        if(visual_sub_grid is True):
            # get cell centroids
            #cells = tet_grid.cells.reshape(-1, 1)[:, 1:]
            #cell_center = tet_grid.points[cells].mean(1)
            # extract cells below the 0 xy plane
            #mask = cell_center[:, 2] < 0
            #cell_ind = mask .nonzero()[0]
            #sub_tet_grid = tet_grid.extract_cells(cell_ind)
            # advanced plotting
            #plotter.add_mesh(sub_tet_grid, 'lightgrey', lighting=True, show_edges=True)
            plotter.add_mesh(tet_grid, show_edges=False)
        else:
            plotter.add_mesh(tet_grid, show_edges=False)

        plotter.show()

    # return the tet grid
    return tet_grid


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # step 1: extract cardiac component from the h5 file
    pcd = extract_component_from_h5_to_pcd(h5_file="UKBRVLV.h5", visual=False)


    # step 2: divide the biventricular model into three parts: the LV-Endocardium, the RV-Endocaridum, and the Epicardium
    pcd_lve = pcd.select_by_index(range(0, 1500))
    pcd_rve = pcd.select_by_index(range(1501, 3224))
    pcd_ep = pcd.select_by_index(range(3225, np.shape(pcd.points)[0]))
    # save & visualize these three parts
    # o3d.io.write_point_cloud("pcd_lve.xyz", pcd_lve)
    # o3d.io.write_point_cloud("pcd_lve.xyz", pcd_lve)
    # o3d.io.write_point_cloud("pcd_lve.xyz", pcd_lve)
    # o3d.visualization.draw_geometries([pcd], width=1024, height=768, left=100, top=100)

    # step 3: convert the point cloud file (.xyz) into surface mesh with MeshLab
    # the surface mesh is stored in .ply file

    # step 4: edit/merge and optimize the surface file with Blender
    #

    # step 5: convert the surface into tetrahedronal mesh with pyTetGen
    surface_mesh_delaunay("Ventricular.ply", visual=True, save_file="Ventricular_tet.vtk")

