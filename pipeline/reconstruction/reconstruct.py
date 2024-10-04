from reconstruction import icp_recon
def recon(config):
    pointcloud = None
    if config['RECON_METHOD'] == 'ICP':
        print('Reconstructing using ICP alignment.')
        pointcloud = icp_recon.recon(config)
    return pointcloud