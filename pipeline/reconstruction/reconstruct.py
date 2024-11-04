from pipeline.reconstruction import icp
def recon(config):
    pointcloud = None
    if config['RECON_METHOD'] == 'ICP':
        print('Reconstructing using the test method.')
        pointcloud = icp.recon(config)
    return pointcloud