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

def extract_component_from_h5_to_pcd(h5_file="",
                                     index=0,
                                     visual=False,
                                     save_file=""):
    """ extract point cloud from the h5 file

    Parameters
    ----------
    h5_file : str
        The name of the h5 file which contains the ventricular model data
    index : int
        The index of which component extrict
    visul : bool
        Wheather plot visualization or not
    save_file: str
        The file name which store the data extracted form the h5 file
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

    points = np.reshape(S[0, 0:N], (-1, 3))

    # convert to point cloud
    point_cloud = pv.PolyData(points)
    point_cloud.compute_normals()

    # save point cloud file, such as .ply
    if (save_file is not ""):
        point_cloud.save(save_file, binary=False)

    # visualize the point cloud
    if (visual is True):
        plotter = pv.Plotter()
        plotter.add_mesh(point_cloud)
        plotter.show()

    # return the point cloud object
    return point_cloud


def surface_mesh_delaunay(surface_mesh_file="",
                          visual=False,
                          visual_sub_grid=True,
                          save_file=""):
    """Perform surface mesh deaunay.

    Parameters
    ----------
    surface_mesh_file : str
        The file which contains the surface mesh
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
        plotter = pv.Plotter()
        plotter.add_mesh(surface_mesh, 'r', 'wireframe')

        # not work now
        if (visual_sub_grid is True):
            mask = np.logical_or(tet.grid.points[:, 0] < 0, tet.grid.points[:, 0] > 80)
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
    pcd = extract_component_from_h5_to_pcd(h5_file="data//UKBRVLV.h5",
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
    surface_mesh_delaunay("data//Ventricular.ply",
                          visual=True,
                          save_file="data//Ventricular_tet.vtk")
