import numpy as np
import pandas as pd
import tqdm
import re
import matplotlib.pyplot as plt

data_dir = '/home/users/g/gomezvar/data/SUN/HARPN/'

data_catalog = pd.read_csv(data_dir+'WORKSPACE/Analyse_summary.csv', delimiter=',',index_col=0)
print("data_catalog:")
print(data_catalog.head())

linfo = pd.read_pickle(data_dir+'WORKSPACE/Analyse_material.p')
print("linfo:")
print(linfo.head())

print(data_catalog.columns)
print("Size data_catalog:", data_catalog['filename'].size)

print(linfo.columns)
print("Size linfo:", linfo['wave'].size)

for col in data_catalog.columns:
    print(col, end=',')

# np.savetxt("../data/rv_shift.txt", data_catalog['rv_shift'].values)

#####this is the continuum cube at stage matching_mad:
continuum_cube = np.load(data_dir+'WORKSPACE/CONTINUUM/'+'Continuum_matching_mad.npy')
print("continuum_cube", np.shape(continuum_cube))

#####this is the correction map at stage matching_activity:
corr_map_matching_activity_daily = np.load(data_dir+'CORRECTION_MAP/map_matching_activity.npy')
print("corr_map_matching_activity_daily", np.shape(corr_map_matching_activity_daily))

max_num = data_catalog['filename'].size
# root_name = 'RASSINE_Stacked_spectrum_B1.00_D2015-07-19T23:46:22.542.p'
numspec = max_num

time_df = pd.DataFrame(columns=['year', 'month', 'day', 'hour', 'minutes', 'seconds', 'msec'])

# the wavelength_orig is the same for all the entries,
# therefore does not matter the index, for simplicity we use index 0
# 57: is only to remove the directory name
data_tmp = pd.read_pickle(data_dir+'WORKSPACE/'+data_catalog['filename'][0][57:])
wavelength_orig = np.float64(data_tmp['wave'])
# np.savetxt("../data/wavelegths.txt", wavelength_orig)

spectra_active_np = np.zeros((numspec, len(wavelength_orig)))
spectra_orig_np = np.zeros((numspec, len(wavelength_orig)))
# spectra_active_df.head()

numspec = max_num
for idx in tqdm.trange(numspec):
    # 57: is only to remove the directory name
    file_name = data_catalog['filename'][idx][57:]

    dateinfo = re.sub('RASSINE_Stacked_spectrum_B1.00_D', '', file_name)
    year = int(dateinfo[:4])
    month = int(dateinfo[5:7])
    day = int(dateinfo[8:10])
    hour = int(dateinfo[11:13])
    minutes = int(dateinfo[14:16])
    seconds = int(dateinfo[17:19])
    msec = int(dateinfo[20:23])
    time_row = pd.Series([year, month, day, hour, minutes, seconds, msec], index=time_df.columns)
    time_df.loc[idx] = time_row

    data = pd.read_pickle(data_dir + 'WORKSPACE/' + file_name)
    # plot data['flux']
    # plt.plot(data['flux'])
    # plt.title('Flux before correction')
    # plt.savefig('flux.png', dpi=100)
    spec_harps_orig = data['flux'] / continuum_cube[idx, :] * linfo['correction_factor'].to_numpy()

    spec_harps_active = spec_harps_orig.copy() + corr_map_matching_activity_daily[idx, :]
    spectra_orig_np[idx, :] = spec_harps_orig
    spectra_active_np[idx, :] = spec_harps_active
    # spec_orig_row = pd.Series(spec_harps_orig, index=spectra_orig_df.columns)
    # spec_active_row = pd.Series(spec_harps_active, index=spectra_active_df.columns)
    # spectra_orig_df.loc[idx] = spec_orig_row
    # spectra_active_df.loc[idx] = spec_active_row
    # break

time_df.to_csv('data/time_df.csv', index=False)
np.save("data/spectra_orig.npy", spectra_orig_np)
np.save("data/spectra_active.npy", spectra_active_np)
print(spectra_active_np.shape)
print(len(time_df))

