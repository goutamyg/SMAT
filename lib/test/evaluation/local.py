from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = '/home/goutam/Datasets/test_data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = ''
    settings.lasot_extension_subset_path = '/home/goutam/Datasets/test_data/lasot_extension_subset'
    settings.lasot_lmdb_path = ''
    settings.lasot_path = '/home/goutam/Datasets/test_data/lasot'
    settings.network_path = '/home/goutam/VisualTracking/research_code_for_github/SMAT/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/goutam/Datasets/test_data/nfs'
    settings.otb_path = '/home/goutam/Datasets/test_data/otb'
    settings.prj_dir = '/home/goutam/VisualTracking/research_code_for_github/SMAT'
    settings.result_plot_path = '/home/goutam/VisualTracking/research_code_for_github/SMAT/output/test/result_plots'
    settings.results_path = '/home/goutam/VisualTracking/research_code_for_github/SMAT/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/goutam/VisualTracking/research_code_for_github/SMAT/output'
    settings.segmentation_path = '/home/goutam/VisualTracking/research_code_for_github/SMAT/output/test/segmentation_results'
    settings.tc128_path = '/home/goutam/Datasets/test_data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/goutam/Datasets/test_data/trackingnet'
    settings.uav_path = '/home/goutam/Datasets/test_data/uav'
    settings.vot18_path = '/home/goutam/Datasets/test_data/vot2018'
    settings.vot22_path = '/home/goutam/VisualTracking/research_code_for_github/SMAT/data/vot2022'
    settings.vot_path = '/home/goutam/VisualTracking/research_code_for_github/SMAT/data/VOT2019'
    settings.youtubevos_dir = ''
    settings.avist_path = '/home/goutam/Datasets/test_data/avist'

    return settings

