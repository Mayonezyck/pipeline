from reconstruction import icp
from reconstruction import icp_neighbor
def recon(config):
    pointcloud = None
    if config['RECON_METHOD'] == 'ICP':
        print('Reconstructing using the test method.')
        pointcloud = icp.recon(config)
    if config['RECON_METHOD'] == 'ICP_Neighbor':
        print('Reconstructing using the neighbor ICP method.')
        pointcloud = icp_neighbor.recon(config)
    return pointcloud