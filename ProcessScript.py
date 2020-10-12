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
import numpy as np
import pyvista as pv
import tetgen
import open3d as o3d


def extract_component_from_h5_to_pcd(h5_file="",
                                     index=0,
                                     visual=False,
                                     save_file="") -> [pv.PolyData, pv.PolyData]:
    """ extract point cloud from the h5 file
        :param h5_file: The name of the h5 file which contains the ventricular model data
        :param index: The index of which component extract
        :param visual: Whether plot visualization or not
        :param save_file: The file name which store the data extracted form the h5 file
        :return:
    """
    assert (h5_file is not "")

    # read H5 file
    pc = h5.File(h5_file, 'r')

    coeff = pc['COEFF']
    latent = pc['LATENT']
    explained = pc['EXPLAINED']
    mu = np.transpose(pc['MU'])

    # calculate the coordinate of the points
    S = mu + (1.5 * np.sqrt(latent[0, 0]) * coeff[index, :])
    N = S.shape[1] // 2

    points_ed = np.reshape(S[0, 0:N], (-1, 3))
    points_es = np.reshape(S[0, N:], (-1, 3))

    pcd_ed = o3d.geometry.PointCloud()
    pcd_ed.points = o3d.utility.Vector3dVector(points_ed)
    pcd_ed.estimate_normals()
    if visual is True:
        o3d.visualization.draw_geometries([pcd_ed], width=1024, height=768, left=100, top=100)

    # convert to point cloud
    point_cloud_ed = pv.PolyData([points_ed])
    point_cloud_ed.compute_normals()
    point_cloud_es = pv.PolyData([points_es])
    point_cloud_es.compute_normals()

    # save point cloud file, such as .ply
    if save_file is not "":
        point_cloud_ed.save(save_file, binary=False)

    # visualize the point cloud
    if visual is True:
        plotter = pv.Plotter()
        plotter.add_mesh(point_cloud_ed)
        plotter.show()

    # return the point cloud object
    return point_cloud_ed, point_cloud_es


def extract_pcd_diff(pcd_ed: pv.PolyData, pcd_es: pv.PolyData) -> pv.PolyData:
    """
    Extract the motion field between the ed and es
    :param pcd_ed:
    :param pcd_es:
    :return:
    """
    pcd_diff = pv.PolyData([pcd_ed.points - pcd_es.points])
    return pcd_diff


def animate_cardiac_motion(pcd_ed: pv.PolyData, pcd_es: pv.PolyData, cycle: int = 10, nframe: int = 60) -> None:
    """

    :param pcd_ed:
    :param pcd_es:
    :param cycle:
    :param nframe:
    :return:
    """
    import time
    motion_field = extract_pcd_diff(pcd_ed, pcd_es)
    plotter = pv.Plotter()
    plotter.add_mesh(pcd_ed)
    plotter.show(auto_close=False)
    plotter._key_press_event_callbacks.clear()

    while True:
        for fm in range(nframe):
            plotter.update_coordinates(pcd_ed.points + motion_field.points / nframe)
            print("current frame: %d", fm)
            time.sleep(0.01)
        for fm in range(nframe):
            plotter.update_coordinates(pcd_ed.points - motion_field.points / nframe)
            print("current frame: %d", fm)
            time.sleep(0.01)

    plotter.close()
    return None


def surface_mesh_delaunay(surface_mesh_file="",
                          visual=False,
                          visual_sub_grid=True,
                          save_file=""):
    """Perform surface mesh deaunay.
        :param surface_mesh_file: The file which contains the surface mesh
        :param visual:
        :param visual_sub_grid:
        :param save_file:
        :return:
    """
    assert (surface_mesh_file is not "")

    # read surface mesh file from .ply
    surface_mesh = pv.PolyData(surface_mesh_file)

    # execute tetgen
    # don't work for now
    tet = tetgen.TetGen(surface_mesh)
    # bug: cannot run correctly
    tet.tetrahedralize(nobisect=True,
                       quality=False,
                       order=1,
                       verbose=1)

    # save the tet grid mesh
    if (save_file is not ""):
        import pyvtk
        vtkelements = pyvtk.VtkData(
            pyvtk.UnstructuredGrid(
                points=tet.node,
                tetra=tet.elem.reshape(-1, 4)
            ),
            "Mesh")
        vtkelements.tofile(save_file)

    # visualize the surface mesh & tet grid
    if (visual is True):
        pv.set_plot_theme("document")
        plotter = pv.Plotter()
        plotter.add_mesh(surface_mesh, 'r', 'wireframe')

        # plotter.show()

        # not work now
        if (visual_sub_grid is True):
            mask = np.logical_and(tet.grid.points[:, 0] > 0, tet.grid.points[:, 0] < 80)
            half_cow = tet.grid.extract_points(mask)
            plotter.add_mesh(half_cow, 'lightgrey', lighting=True, show_edges=True)
        else:
            plotter.add_mesh(tet.grid, 'lightgrey', lighting=True, show_edges=True)

        plotter.show()

    # return the tet grid
    return tet.node, tet.elem


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # step 1: extract cardiac component from the h5 file
    pcd_ed, pcd_es = extract_component_from_h5_to_pcd(h5_file="data//UKBRVLV.h5",
                                                      visual=False)

    # step 2: divide the biventricular model into three parts: the LV-Endocardium, the RV-Endocaridum, and the Epicardium
    # pv.PolyData(pcd.points[0:1500]).save("data//pcd_lve.ply", binary=False)
    # pv.PolyData(pcd.points[1501:3224]).save("data//pcd_rve.ply", binary=False)
    # pv.PolyData(pcd.points[3225:-1]).save("data//pcd_eve.ply", binary=False)

    # step 3: convert the point cloud file (.xyz) into surface mesh with MeshLab
    # the surface mesh is stored in .ply file

    # step 4: edit/merge and optimize the surface file with Blender
    #

    # step 5: convert the surface into tetrahedronal mesh with pyTetGen
    # surface_mesh_delaunay("data//Ventricular.ply",
    #                       visual=True,
    #                       visual_sub_grid=False,
    #                       save_file="data//Ventricular_tet.vtk")

    # step 6: calculate the motion field
    # motion_field = extract_pcd_diff(pcd_ed, pcd_es)

    # step 7: animation the motion
    # animate_cardiac_motion(pcd_ed, pcd_es, 60)
