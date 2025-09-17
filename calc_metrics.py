from os.path import join
from glob import glob
from argparse import ArgumentParser
import pandas as pd
import librosa
from pesq import pesq
from pystoi import stoi
from sgmse.util.other import energy_ratios, mean_std
import numpy as np
from tqdm import tqdm
import time
from sgmse.util.semp import composite
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True, help='Directory containing the clean data')
    parser.add_argument("--noisy_dir", type=str, required=True, help='Directory containing the noisy data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the noisy data')
    args = parser.parse_args()

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [],  "si_sar": [],
            "ssnr": [], "pesq_mos": [], "csig": [], "cbak": [], "covl": [], "stoi": []}


    noisy_files = []
    noisy_files += sorted(glob(join(args.noisy_dir, '*.wav')))
    noisy_files += sorted(glob(join(args.noisy_dir, '**', '*.wav')))
    
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(args.noisy_dir, "")[1:]
        if 'dB' in filename:
            clean_filename = filename.split("_")[0] + ".wav"
        else:
            clean_filename = filename

        # The audio was loaded using librosa and a sampling rate of 16000 was specified
        x, sr_x = librosa.load(join(args.clean_dir, clean_filename), sr=16000)
        y, sr_y = librosa.load(join(args.noisy_dir, filename), sr=16000)
        x_hat, sr_x_hat = librosa.load(join(args.enhanced_dir, filename), sr=16000)

        # Residual noise
        n = y - x
        semp = composite(x,x_hat, sr_x)
        _ssnr, _pesq, _csig, _cbak, _covl, _stoi = semp
        # Ensure all sample rates are the same
        assert sr_x == sr_x_hat, "Sample rates should be equal"

        # calculate metrics
        data["filename"].append(filename)
        data["pesq"].append(pesq(16000, x, x_hat, 'wb'))
        data["estoi"].append(stoi(x, x_hat, sr_x, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
        data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
        data["si_sar"].append(energy_ratios(x_hat, x, n)[2])
        data["ssnr"].append(0)#(_ssnr)
        data["pesq_mos"].append(_pesq)
        data["csig"].append(_csig)
        data["cbak"].append(_cbak)
        data["covl"].append(_covl)
        data["stoi"].append(0)#(_stoi)
    # Save results as DataFrame    
    df = pd.DataFrame(data)

    # Print results
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())))
    print("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())))
    print("PESQ MOS: {:.2f} ± {:.2f}".format(*mean_std(df["pesq_mos"].to_numpy())))
    print("CSIG: {:.2f} ± {:.2f}".format(*mean_std(df["csig"].to_numpy())))
    print("CBAK: {:.2f} ± {:.2f}".format(*mean_std(df["cbak"].to_numpy())))
    print("COVL: {:.2f} ± {:.2f}".format(*mean_std(df["covl"].to_numpy())))

   
    # Save results to CSV file
    df.to_csv(join(args.enhanced_dir, "_results.csv"), index=False)
