import os
import pandas
import boto3
from botocore import UNSIGNED
from botocore.client import Config

def download_file_from_aws():
    # NOTE: Change this to the S3 location when finalized
    ocmr_data_attributes_location = './ocmr_data_attributes.csv'

    df = pandas.read_csv('./ocmr_data_attributes.csv')
    # Cleanup empty rows and columns
    df.dropna(how='all', axis=0, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)

    # Show the first 10 items in the list
    print(df.head(10))

    # This is a sample query that filters on file names that contain "fs_", scn equals "15avan", and viw equals "lax"
    # (i.e. fully sampled, LAX view, collected on 1.5T Avanto)
    selected_df = df.query ('`file name`.str.contains("fs_") and scn=="15avan" and viw=="sax"', engine='python')
    print(selected_df)

    # The local path where the files will be downloaded to
    download_path = './ocmr_data'

    # Replace this with the name of the OCMR S3 bucket
    bucket_name = 'ocmr'

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    count = 1
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Iterate through each row in the filtered DataFrame and download the file from S3.
    # Note: Test after finalizing data in S3 bucket
    for index, row in selected_df.iterrows():
        if os.path.exists('{}/{}'.format(download_path, row['file name'])):
            print('{} already exit in {} (File {} of {})'.format(row['file name'], download_path, count, len(selected_df)))
        else:
            print('Downloading {} to {} (File {} of {})'.format(row['file name'], download_path, count, len(selected_df)))
            s3_client.download_file(bucket_name, 'data/{}'.format(row['file name']),
                                    '{}/{}'.format(download_path, row['file name']))
        count += 1


def read_kspace_from_ocmr_data(file_path='./ocmr_data/us_0143_pt_1_5T.h5'):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    from ismrmrdtools import show, transform
    import read_ocmr as read

    # Load the data, display size of kData and scan parmaters
    kData, param = read.read_ocmr(file_path);

    print('Dimension of kData: ', kData.shape)
    print('Scan paramters:')
    import pprint;
    pprint.pprint(param)

    # Show the sampling Pattern
    # kData_tmp-[kx,ky,kz,coil,phase,set,slice,rep], samp-[kx,ky,kz,phase,set,slice,rep]
    dim_kData = kData.shape;
    CH = dim_kData[3];
    SLC = dim_kData[6];
    kData_tmp = np.mean(kData, axis=8);  # average the k-space if average > 1
    samp = (abs(np.mean(kData_tmp, axis=3)) > 0).astype(np.int)  # kx ky kz phase set slice

    slc_idx = math.floor(SLC / 2);
    fig1 = plt.figure(1);
    fig1.suptitle("Sampling Pattern", fontsize=14);
    plt.subplot2grid((1, 8), (0, 0), colspan=6);
    tmp = plt.imshow(np.transpose(np.squeeze(samp[:, :, 0, 0, 0, slc_idx])), aspect='auto');
    plt.xlabel('kx');
    plt.ylabel('ky');
    tmp.set_clim(0.0, 1.0)  # ky by kx
    plt.subplot2grid((1, 9), (0, 7), colspan=2);
    tmp = plt.imshow(np.squeeze(samp[int(dim_kData[0] / 2), :, 0, :, 0, slc_idx]), aspect='auto');
    plt.xlabel('frame');
    plt.yticks([]);
    tmp.set_clim(0.0, 1.0)  # ky by frame

    # Average the k-sapce along phase(time) dimension
    kData_sl = kData_tmp[:, :, :, :, :, :, slc_idx, 0];
    samp_avg = np.repeat(np.sum(samp[:, :, :, :, :, slc_idx, 0], 3), CH, axis=3) + np.finfo(float).eps
    kData_sl_avg = np.divide(np.squeeze(np.sum(kData_sl, 4)), np.squeeze(samp_avg));

    im_avg = transform.transform_kspace_to_image(kData_sl_avg, [0, 1]);  # IFFT (2D image)
    im = np.sqrt(np.sum(np.abs(im_avg) ** 2, 2))  # Sum of Square
    fig2 = plt.figure(2);
    plt.imshow(np.transpose(im), cmap='gray');
    plt.axis('off');  # Show the image

    plt.show()


def reconstruct_fully_sampled_ocmr_datasets(filename='./ocmr_data/fs_0005_1_5T.h5'):
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    from ismrmrdtools import show, transform
    # import ReadWrapper
    import read_ocmr as read

    # Load the data, display size of kData and scan parmaters
    kData, param = read.read_ocmr(filename);
    print('Dimension of kData: ', kData.shape)

    # Image reconstruction (SoS)
    dim_kData = kData.shape;
    CH = dim_kData[3];
    SLC = dim_kData[6];
    kData_tmp = np.mean(kData, axis=8);  # average the k-space if average > 1

    im_coil = transform.transform_kspace_to_image(kData_tmp, [0, 1]);  # IFFT (2D image)
    im_sos = np.sqrt(np.sum(np.abs(im_coil) ** 2, 3));  # Sum of Square
    print('Dimension of Image (with ReadOut ovesampling): ', im_sos.shape)
    RO = im_sos.shape[0];
    image = im_sos[math.floor(RO / 4):math.floor(RO / 4 * 3), :, :];  # Remove RO oversampling
    print('Dimension of Image (without ReadOout ovesampling): ', image.shape)

    # Show the reconstructed cine image
    from IPython.display import clear_output
    import time

    slc_idx = math.floor(SLC / 2);
    print(slc_idx)
    image_slc = np.squeeze(image[:, :, :, :, :, :, slc_idx]);
    for rep in range(5):  # repeate the movie for 5 times
        for frame in range(image_slc.shape[2]):
            clear_output(wait=True)
            plt.imshow(image_slc[:, :, frame], cmap='gray');
            plt.axis('off');
            plt.show()
            time.sleep(0.03)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # step 1: download h5 file from the aws server
    # download_file_from_aws()

    # step 2: an example of reading k-space from OCMR data
    # read_kspace_from_ocmr_data()

    # step 3: reconstruct fully sampled OCMR datasets
    reconstruct_fully_sampled_ocmr_datasets()