import spectral.projection_utils as pju
import mesh.trimesh as tm



class Faust():

    def __init__(self, path):

        #TODO: create check if its a pathlib.Path instance, otherwise create one
        self.path = path # path to FAUST dataset
    

    def get_samples(self, sample_nr, test=False, scan=True):
        mode = "test" if test else "training"
        mode_prefix = "test_" if test else "tr_"
        type = "scans" if scan else "registrations"
        type_prefix = "scan_" if scan else "reg_"

        paths = [self.path / type / mode / mode_prefix + type_prefix + nr + ".ply" for nr in sample_nr]

        meshes = [tm.TriMesh(path) for path in paths]

        return meshes

    def calc_point_to_surface_map(self, shape1, shape2):
        """
        Calculate the point to surface map from shape1 to shape2

        Parameters
        -------------------------------
        shape1, shape2  : str three digit number corresponding to the sample number in the FAUST dataset

        Outputs
        -------------------------------
        """

        # Load the two representation meshes
        rep_mesh1, rep_mesh2 = self.get_samples([shape1, shape2], test=False, scan=False)

        # Load the scan mesh
        scan_mesh = self.get_samples([shape1], test=False, scan=True)
        
        points = pju.project_point_to_rep_to_rep(rep_mesh1, rep_mesh2, scan_mesh.vertlist)

        return points

    def get_geodesic_error():

        pass
