from reconstruction import icp_recon
#from reconstruction import test_recon
def recon(config):
    pointcloud = None
    if config['RECON_METHOD'] == 'ICP':
        print('Reconstructing using ICP alignment.')
        pointcloud = icp_recon.recon(config)
    elif config['RECON_METHOD'] == 'TEST':
        print('Reconstructing using the test method.')
        #pointcloud = test_recon.recon(config)
    return pointcloud